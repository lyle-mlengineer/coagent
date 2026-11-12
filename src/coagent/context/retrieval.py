"""BM25 retrieval over function-sized chunks (DESIGN §5.3, PRD FR-CTX-2).

What this is
------------
Given a query (the task, a failing test name, a symbol the model asked about),
return the few source excerpts most likely to be relevant, formatted as citable
``file:start-end`` blocks that fit the retrieved-excerpts slot of the §5.1
budget.

Why BM25 and not embeddings
---------------------------
On 2 vCPU beside a GPU, an embedding index is a memory and latency tax for a
ranking gain BM25 mostly covers in a repo of this size. ``all-MiniLM-L6-v2`` is
an optional future upgrade, not a dependency and not part of v1.

Why chunks come from the repo map
---------------------------------
The repo map already parses every file into ``Signature`` objects carrying line
spans. Re-parsing for retrieval would double the only real cost in the context
layer, so :class:`Retriever` consumes a :class:`RepoMap` and derives chunks
from it. Two consequences worth knowing:

* A chunk is a *function-sized* unit with its signature attached, so the model
  always sees the name and parameters of what it is being shown.
* Files the map skipped (vendored, generated, non-Python) are not retrievable.
  That is intentional: they are not editable either.

Ranking is BM25 plus a symbol-graph adjacency bonus: if a chunk's symbols are
defined in, or referenced from, a file already in context, that chunk gets
pushed up. This is the "callers/callees of the files already in context" part of
§5.3, and it is what lets the agent pull a helper into view without being told.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .repo_map import RepoMap
from .tokens import estimate_tokens

__all__ = ["Chunk", "RetrievalConfig", "Retriever", "Recalled"]

#: §5.1 gives retrieved excerpts 3000 tokens, BM25, 5 chunks max.
DEFAULT_MAX_TOKENS = 3000
DEFAULT_MAX_CHUNKS = 5

#: A chunk longer than this is a class or a huge function. We cap the body so a
#: single 800-line method cannot consume the whole excerpt slot.
MAX_CHUNK_LINES = 80

#: BM25 parameters. k1 controls term-frequency saturation and b controls
#: length normalisation. 1.2 / 0.75 are the standard defaults and are fine for
#: source code, which has short documents and repetitive vocabulary.
BM25_K1 = 1.2
BM25_B = 0.75

#: Identifier-ish tokens. Code is not English: we keep snake_case and camelCase
#: whole, then split them, so `load_csv` matches a query for "csv loader".
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

#: Words that appear in almost every file and carry no signal. Kept small on
#: purpose: an aggressive stoplist on code drops real identifiers.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "if", "in", "is", "it", "of", "to", "for",
        "self", "cls", "def", "class", "return", "import", "from", "as", "with",
        "not", "none", "true", "false", "this", "that", "be", "on", "at", "by",
        "pass", "else", "elif", "while", "try", "except", "raise", "lambda",
    }
)


def _tokenize(text: str) -> list[str]:
    """Split source into lowercase terms, exploding snake_case and camelCase."""
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        low = word.lower()
        if low not in _STOPWORDS and len(low) > 1:
            out.append(low)
        # `load_csv` -> load, csv; `loadCsv` -> load, csv. The joined form is
        # kept above, so both an exact and a split query match.
        if "_" in word or any(c.isupper() for c in word[1:]):
            parts = [p for p in re.split(r"_+", word) if p]
            expanded: list[str] = []
            for part in parts:
                expanded.extend(_CAMEL_RE.findall(part) or [part])
            for part in expanded:
                p = part.lower()
                if p not in _STOPWORDS and len(p) > 1:
                    out.append(p)
    return out


@dataclass
class Chunk:
    """One retrievable unit: a function, method, class head, or module prologue."""

    rel: str
    start_line: int
    end_line: int
    text: str
    #: Signature-derived name, used for adjacency and for the excerpt header.
    symbol: str | None = None
    kind: str = "def"
    #: Terms for BM25, computed once per chunk rather than per query.
    terms: list[str] = field(default_factory=list)

    @property
    def locator(self) -> str:
        return f"{self.rel}:{self.start_line}-{self.end_line}"

    def header(self) -> str:
        name = f" {self.symbol}" if self.symbol else ""
        return f"--- {self.locator}{name} ---"

    def render(self) -> str:
        return f"{self.header()}\n{self.text}"


@dataclass
class RetrievalConfig:
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_chunks: int = DEFAULT_MAX_CHUNKS
    max_chunk_lines: int = MAX_CHUNK_LINES
    #: Weight of the adjacency bonus relative to a normalised BM25 score.
    adjacency_weight: float = 0.35


@dataclass(frozen=True)
class Recalled:
    """A scored chunk, returned so callers can log or re-rank."""

    chunk: Chunk
    score: float
    bm25: float
    adjacency: float

    @property
    def locator(self) -> str:
        return self.chunk.locator


class Retriever:
    """BM25 + symbol adjacency over chunks derived from a :class:`RepoMap`.

    Usage from the loop::

        retriever = Retriever(repo_map)
        excerpts = retriever.render("chunked csv loader dtype", in_context={"src/frame.py"})

    The index is built once per ``RepoMap`` and reused across queries; a query
    costs one pass over the chunk list.
    """

    def __init__(
        self,
        repo_map: RepoMap,
        *,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.repo_map = repo_map
        self.config = config or RetrievalConfig()
        self._chunks: list[Chunk] | None = None
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    # ------------------------------------------------------------------ index

    @property
    def avgdl(self) -> float:
        """Mean chunk length, after ensuring the index exists."""
        self.chunks()
        return self._avgdl

    @property
    def doc_freq(self) -> Counter[str]:
        """Document frequencies, after ensuring the index exists."""
        self.chunks()
        return self._df

    def chunks(self) -> list[Chunk]:
        """Build (once) and return the chunk corpus."""
        if self._chunks is None:
            self._chunks = self._build_chunks()
            self._df = Counter()
            for chunk in self._chunks:
                for term in set(chunk.terms):
                    self._df[term] += 1
            total = sum(len(c.terms) for c in self._chunks)
            self._avgdl = total / len(self._chunks) if self._chunks else 0.0
        return self._chunks

    def _build_chunks(self) -> list[Chunk]:
        """Cut every file into function-sized chunks using repo map spans."""
        chunks: list[Chunk] = []
        for fm in self.repo_map.files():
            try:
                source = (self.repo_map.root / fm.rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = source.splitlines()
            if not lines:
                continue
            spans = [(s.start_line, s.end_line, s) for s in fm.signatures if s.start_line > 0]
            spans.sort()

            # The module prologue — imports and module docstring — is often the
            # most informative part of a file about what it depends on.
            first = spans[0][0] if spans else len(lines) + 1
            if first > 1 and first <= 40:
                chunks.append(
                    _make_chunk(fm.rel, 1, first - 1, lines, "<module>", "module", self.config)
                )

            for start, end, sig in spans:
                if sig.kind == "const":
                    continue  # a bare `X = ...` line is not worth an excerpt slot
                chunks.append(
                    _make_chunk(fm.rel, start, end or start, lines, sig.name, sig.kind, self.config)
                )
        return chunks

    # ------------------------------------------------------------------ query

    def search(
        self,
        query: str,
        *,
        in_context: set[str] | None = None,
        limit: int | None = None,
    ) -> list[Recalled]:
        """Rank chunks against ``query``, boosted by adjacency to ``in_context``.

        ``in_context`` holds repo-relative paths already shown to the model
        (files it read, or the file the task names). Chunks defined in those
        files are excluded — re-showing them wastes the slot — while chunks that
        *reference* their symbols get the adjacency bonus.
        """
        chunks = self.chunks()
        if not chunks:
            return []

        terms = _tokenize(query)
        if not terms:
            return []

        context = {_norm(p) for p in (in_context or set())}
        context_symbols: set[str] = set()
        for chunk in chunks:
            if chunk.rel in context and chunk.symbol and chunk.symbol != "<module>":
                context_symbols.add(chunk.symbol.lower())

        n = len(chunks)
        scores: list[Recalled] = []
        for chunk in chunks:
            if chunk.rel in context:
                continue  # already on screen
            bm25 = self._bm25(terms, chunk, n)
            adjacency = self._adjacency(chunk, context, context_symbols)
            total = bm25 + self.config.adjacency_weight * adjacency
            if total > 0:
                scores.append(Recalled(chunk=chunk, score=total, bm25=bm25, adjacency=adjacency))

        # Deterministic order: score, then locator. Without the tiebreak a
        # re-query can shuffle equal-scoring chunks and invalidate a resumed
        # session's cached context.
        scores.sort(key=lambda r: (-r.score, r.chunk.rel, r.chunk.start_line))
        cap = limit if limit is not None else self.config.max_chunks
        return scores[:cap]

    def _bm25(self, terms: list[str], chunk: Chunk, n: int) -> float:
        """Okapi BM25, computed per-query (the corpus is small enough)."""
        if not self._avgdl:
            return 0.0
        tf = Counter(chunk.terms)
        dl = len(chunk.terms)
        score = 0.0
        for term in set(terms):
            freq = tf.get(term, 0)
            if not freq:
                continue
            df = self._df.get(term, 0)
            # Standard BM25 IDF with the +1 inside the log so a term present in
            # every document scores ~0 rather than going negative.
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avgdl)
            score += idf * (freq * (BM25_K1 + 1)) / denom
        return score

    def _adjacency(self, chunk: Chunk, context: set[str], context_symbols: set[str]) -> float:
        """Symbol-graph bonus: does this chunk touch what is already in context?

        Two edges count — the chunk defines a symbol used in a context file, or
        the chunk references a symbol defined in one. Both are 1.0; the caller
        scales the result with ``adjacency_weight``.
        """
        if not context and not context_symbols:
            return 0.0
        chunk_terms = set(chunk.terms)
        if not chunk_terms:
            return 0.0
        # Import of a module already in context, e.g. "from .frame import ...".
        for rel in context:
            stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if len(stem) > 2 and stem in chunk_terms:
                return 1.0
        if chunk.symbol and chunk.symbol.lower() in context_symbols:
            return 1.0
        if context_symbols & chunk_terms:
            return 1.0
        return 0.0

    # ----------------------------------------------------------------- render

    def render(
        self,
        query: str,
        *,
        in_context: set[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the excerpts for ``query``, packed to the §5.1 slot.

        Chunks are added whole until the budget is hit. A chunk that does not fit
        is skipped rather than truncated: half a function is worse than a
        different whole one, and the excerpt header would lie about the range.
        """
        budget = self.config.max_tokens if max_tokens is None else max_tokens
        recalled = self.search(query, in_context=in_context)
        if not recalled:
            return ""

        out: list[str] = []
        used = 0
        for hit in recalled:
            block = hit.chunk.render()
            cost = estimate_tokens(block)
            if used + cost > budget:
                continue
            out.append(block)
            used += cost
            if len(out) >= self.config.max_chunks:
                break
        return "\n\n".join(out)


def _make_chunk(
    rel: str,
    start: int,
    end: int,
    lines: list[str],
    symbol: str | None,
    kind: str,
    config: RetrievalConfig,
) -> Chunk:
    """Slice a chunk, capping the body so one huge function cannot hog the slot."""
    end = min(end, start + config.max_chunk_lines - 1, len(lines))
    body = "\n".join(lines[start - 1 : end])
    return Chunk(
        rel=rel,
        start_line=start,
        end_line=end,
        text=body,
        symbol=symbol,
        kind=kind,
        terms=_tokenize(body),
    )


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")
