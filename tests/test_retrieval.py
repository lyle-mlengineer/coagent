"""BM25 retrieval tests (PRD FR-CTX-2; DESIGN §5.3).

Retrieval earns its slot only if it is *right* and *deterministic*: a weak model
cannot recover from being shown the wrong function, and a resumed session cannot
tolerate a re-query that shuffles equal-scoring chunks. These tests pin both,
plus the two structural guarantees — chunks come from the repo map's parse (no
second parse), and excerpts never exceed their budget.
"""

from __future__ import annotations

import textwrap

import pytest

from coagent.context.repo_map import RepoMap
from coagent.context.retrieval import (
    Chunk,
    RetrievalConfig,
    Retriever,
    _make_chunk,
    _tokenize,
)
from coagent.context.tokens import estimate_tokens


def _write(root, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """A small repo with an obvious retrieval target and decoys."""
    _write(
        tmp_path,
        "src/frame.py",
        """
        class Frame:
            def __init__(self, rows):
                self.rows = rows

            def head(self, n=5):
                return self.rows[:n]
        """,
    )
    _write(
        tmp_path,
        "src/loader.py",
        """
        from .frame import Frame


        def load_csv(path, chunk_size=None):
            rows = []
            with open(path) as fh:
                for line in fh:
                    rows.append(line.split(","))
            return Frame(rows)
        """,
    )
    _write(
        tmp_path,
        "src/unrelated.py",
        """
        def send_email(address, subject):
            return f"{address}: {subject}"
        """,
    )
    return tmp_path


# ------------------------------------------------------------------ tokenize


def test_tokenize_explodes_snake_case() -> None:
    terms = _tokenize("load_csv")
    assert "load_csv" in terms
    assert "csv" in terms
    assert "load" in terms


def test_tokenize_explodes_camel_case() -> None:
    terms = _tokenize("loadCsvFile")
    assert "csv" in terms
    assert "file" in terms


def test_tokenize_drops_stopwords_and_short_terms() -> None:
    terms = _tokenize("def f(self, return_value): pass")
    assert "def" not in terms
    assert "self" not in terms
    assert "return" not in terms
    assert "return_value" in terms


# -------------------------------------------------------------------- chunks


def test_chunks_derive_from_repo_map_parse(repo) -> None:
    """The whole point: retrieval must not trigger a second parse."""
    mapper = RepoMap(repo)
    retriever = Retriever(mapper)
    chunks = retriever.chunks()
    assert chunks
    # Every chunk's line span must match a signature the map already found.
    spans = {(f.rel, s.start_line) for f in mapper.files() for s in f.signatures}
    for chunk in chunks:
        if chunk.symbol != "<module>":
            assert (chunk.rel, chunk.start_line) in spans


def test_chunk_text_matches_source_lines(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    chunk = next(c for c in retriever.chunks() if c.symbol == "load_csv")
    lines = (repo / "src/loader.py").read_text(encoding="utf-8").splitlines()
    expected = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
    assert chunk.text == expected
    assert "def load_csv" in chunk.text


def test_module_prologue_is_a_chunk(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    prologue = [c for c in retriever.chunks() if c.symbol == "<module>" and c.rel.endswith("loader.py")]
    assert prologue, "the import block should be retrievable"
    assert "from .frame import Frame" in prologue[0].text


def test_chunk_locator_is_citable(repo) -> None:
    chunk = Chunk(rel="src/a.py", start_line=10, end_line=20, text="x", symbol="fn")
    assert chunk.locator == "src/a.py:10-20"
    assert "src/a.py:10-20 fn" in chunk.header()


def test_long_chunk_is_capped(tmp_path) -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(500))
    _write(tmp_path, "long.py", f"def huge():\n{body}\n")
    retriever = Retriever(RepoMap(tmp_path), config=RetrievalConfig(max_chunk_lines=20))
    chunk = next(c for c in retriever.chunks() if c.symbol == "huge")
    assert chunk.end_line - chunk.start_line + 1 <= 20


def test_constants_are_not_chunks(tmp_path) -> None:
    _write(tmp_path, "c.py", "MAX_SIZE = 10\n\n\ndef fn():\n    return 1\n")
    retriever = Retriever(RepoMap(tmp_path))
    assert not any(c.symbol == "MAX_SIZE" for c in retriever.chunks())


# ------------------------------------------------------------------ ranking


def test_relevant_chunk_ranks_first(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    hits = retriever.search("load csv rows into a frame")
    assert hits
    top = hits[0].chunk
    assert top.rel == "src/loader.py"
    assert top.symbol in {"load_csv", "<module>"}


def test_irrelevant_code_is_not_returned(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    hits = retriever.search("csv chunk_size")
    rels = {h.chunk.rel for h in hits}
    assert "src/unrelated.py" not in rels


def test_empty_query_returns_nothing(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.render("") == ""


def test_query_with_no_matches_returns_nothing(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    assert retriever.search("zzzznotpresentzzzz") == []


def test_scores_are_monotonically_ordered(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    hits = retriever.search("frame rows")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_is_deterministic(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    a = [(h.locator, round(h.score, 6)) for h in retriever.search("frame rows csv")]
    b = [(h.locator, round(h.score, 6)) for h in retriever.search("frame rows csv")]
    assert a == b


def test_limit_is_respected(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    assert len(retriever.search("frame rows csv loader", limit=1)) <= 1


def test_max_chunks_from_config(repo) -> None:
    retriever = Retriever(RepoMap(repo), config=RetrievalConfig(max_chunks=2))
    assert len(retriever.search("frame rows csv loader")) <= 2


# ----------------------------------------------------------------- adjacency


def test_context_symbols_boost_callers(repo) -> None:
    """With the frame module on screen, its callers should surface."""
    retriever = Retriever(RepoMap(repo))
    hits = retriever.search("rows", in_context={"src/frame.py"})
    assert any(h.adjacency > 0 for h in hits), "adjacency bonus never applied"
    assert hits[0].adjacency > 0


def test_files_in_context_are_not_reshown(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    hits = retriever.search("frame rows csv", in_context={"src/frame.py"})
    assert all(h.chunk.rel != "src/frame.py" for h in hits)


def test_adjacency_absent_without_context(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    hits = retriever.search("frame rows csv")
    assert all(h.adjacency == 0 for h in hits)


# ------------------------------------------------------------------ budget


def test_render_respects_token_budget(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    out = retriever.render("frame rows csv loader", max_tokens=40)
    assert estimate_tokens(out) <= 45


def test_render_headers_carry_citable_ranges(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    out = retriever.render("load csv")
    assert "--- src/loader.py:" in out
    # file:start-end
    assert ":" in out.splitlines()[0]


def test_render_never_truncates_a_function(repo) -> None:
    """Half a function with a full-looking header would mislead the model."""
    retriever = Retriever(RepoMap(repo))
    out = retriever.render("load csv", max_tokens=500)
    assert "return Frame(rows)" in out


def test_render_default_budget_is_the_design_slot(repo) -> None:
    retriever = Retriever(RepoMap(repo))
    assert retriever.config.max_tokens == 3000
    assert retriever.config.max_chunks == 5


# ------------------------------------------------------------------- ledgers


def test_empty_repo_is_safe(tmp_path) -> None:
    retriever = Retriever(RepoMap(tmp_path))
    assert retriever.chunks() == []
    assert retriever.search("anything") == []
    assert retriever.render("anything") == ""


def test_make_chunk_clamps_to_file_end() -> None:
    lines = ["def f():", "    return 1"]
    chunk = _make_chunk("a.py", 1, 99, lines, "f", "def", RetrievalConfig())
    assert chunk.end_line == 2
    assert chunk.text == "def f():\n    return 1"


def test_idf_does_not_go_negative_for_universal_terms(tmp_path) -> None:
    """A term in every document must score ~0, not a negative that inverts rank."""
    for i in range(6):
        _write(tmp_path, f"m{i}.py", f"def common_{i}(value):\n    return value\n")
    retriever = Retriever(RepoMap(tmp_path))
    hits = retriever.search("common_3")
    assert hits
    assert all(h.score >= 0 for h in hits)
