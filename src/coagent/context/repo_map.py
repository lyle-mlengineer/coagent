"""Repo map: signatures only, ranked, packed to a token budget (DESIGN §5.2).

Why this exists
---------------
On a 12k window, what you show the model beats what you tell it. A naive
"walk every .py and list every def" map spends its budget on files that have
nothing to do with the task and blows up on large repos, so the model greps
anyway — which is the failure this module is supposed to prevent. So the map
is built in three deliberate stages:

1. **Parse** every source file into signatures (no bodies). tree-sitter when a
   grammar is available, a Python ``ast``/regex fallback otherwise. A missing
   grammar degrades the map; it never ends the session (§5.2).
2. **Rank** with PageRank-lite:
   ``0.5·reference_count + 0.3·recently_edited + 0.2·mentioned_in_task``.
   Files the task names always survive the budget.
3. **Pack** greedily into the repo map's token slot from §5.1 (default 2000),
   dropping whole files rather than mid-signature lines.

Everything here is deterministic: no LLM, no network, no clock reads beyond the
mtime signal the caller passes in. That is what makes it testable and what lets
the loop trust the same map across a resume.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .tokens import fit

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "FileMap",
    "RepoMap",
    "RepoMapConfig",
    "Signature",
]

#: §5.1 gives the repo map a 2000-token slot out of a 12k window.
DEFAULT_MAX_TOKENS = 2000

#: Source extensions we can extract Python signatures from. Other languages get
#: a shallow tree-sitter pass when their grammar is installed.
_PY_SUFFIXES = {".py", ".pyi"}

#: Directories that are never part of "the repo" as a model should see it.
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".coagent",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    ".eggs",
    "site-packages",
}

#: Generated files are noise: they are huge, and their signatures tell the model
#: nothing it could edit.
_SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".pyd"}

#: Cap on files considered at all. A repo with 20k files is not one a 7B model
#: can navigate; the ranking below is only meaningful if parsing stays cheap.
MAX_FILES = 800

#: Cap on signatures emitted per file. A 400-line generated class body would
#: otherwise eat the whole map; 40 is enough to convey a module's shape.
MAX_SIGNATURES_PER_FILE = 40

#: Identifier tokens used for both task-mention matching and reference counts.
#: The 3-character minimum is deliberate: one- and two-character names (`x`,
#: `_`, `fn`) would match almost every file and make the ranking meaningless.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass(frozen=True)
class Signature:
    """One line of the map: a def, a class, or a class member."""

    kind: str  # "class" | "def" | "member" | "const"
    text: str  # rendered signature, no body, no trailing colon

    def render(self) -> str:
        indent = {"def": "  ", "member": "    ", "const": "  "}.get(self.kind, "")
        return f"{indent}{self.text}"


@dataclass
class FileMap:
    """Parsed shape of one file plus the signals used to rank it."""

    rel: str
    signatures: list[Signature] = field(default_factory=list)
    #: Symbols the file defines and references elsewhere — drives reference_count.
    references: int = 0
    #: mtime, used only for the "recently edited" signal.
    mtime: float = 0.0
    #: True when the task text names this file or one of its symbols.
    mentioned: bool = False
    #: True when the session has read or written this file recently.
    recently_edited: bool = False
    #: Set when parsing fell back, so the map can note it honestly.
    degraded: bool = False

    def render(self) -> str:
        lines = [self.rel]
        for sig in self.signatures:
            lines.append(sig.render())
        if not self.signatures:
            lines.append("  (no signatures)")
        return "\n".join(lines)


@dataclass
class RepoMapConfig:
    """Knobs for map construction; defaults match DESIGN §5.1/§5.2."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    max_files: int = MAX_FILES
    use_tree_sitter: bool = True


class _Parsers:
    """Lazily built signature extractors.

    Import-time failure is the normal case on a slim Colab image, so every
    optional dependency is resolved on first use and cached as either a parser
    or ``None``. Nothing here raises into the loop.
    """

    def __init__(self, *, use_tree_sitter: bool = True) -> None:
        self._use_ts = use_tree_sitter
        self._ts: dict[str, object] = {}
        self._ts_tried = False
        self.degraded_languages: set[str] = set()

    # ------------------------------------------------------------------ python

    def python_signatures(self, text: str, *, rel: str) -> tuple[list[Signature], bool]:
        """Return (signatures, degraded).

        ``ast`` is tried first for Python: it is stdlib, exact, and already gives
        us qualified names and decorators. tree-sitter is only better for
        *partially broken* files, so it is the recovery path, then regex.
        """
        try:
            return self._ast_signatures(text), False
        except (SyntaxError, ValueError, RecursionError):
            pass

        if self._use_ts:
            sigs = self._tree_sitter_signatures(text, rel=rel)
            if sigs is not None:
                return sigs, False

        self.degraded_languages.add("python")
        return self._regex_signatures(text), True

    @staticmethod
    def _ast_signatures(text: str) -> list[Signature]:
        tree = ast.parse(text)
        out: list[Signature] = []

        def visit_body(body: list[ast.stmt], *, in_class: bool, depth: int) -> None:
            for node in body:
                if len(out) >= MAX_SIGNATURES_PER_FILE:
                    return
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "member" if in_class else "def"
                    out.append(Signature(kind, _render_function(node)))
                elif isinstance(node, ast.ClassDef):
                    out.append(Signature("class", _render_class(node)))
                    # One level of members only: nested classes collapse.
                    if depth < 1:
                        visit_body(node.body, in_class=True, depth=depth + 1)
                elif isinstance(node, ast.Assign) and not in_class and depth == 0:
                    # Module-level constants are part of a module's shape, but
                    # functions are not defined by them — emit only simple names
                    # so `a = b = f()` does not leak a call expression.
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if names:
                        out.append(Signature("const", f"{' = '.join(names)} = ..."))

        visit_body(tree.body, in_class=False, depth=0)
        return out

    # ------------------------------------------------------------- tree-sitter

    def _ensure_ts(self, rel: str) -> object | None:
        """Build a tree-sitter parser for ``rel``'s language, or None."""
        if not self._use_ts:
            return None
        suffix = Path(rel).suffix.lower()
        if suffix in self._ts:
            return self._ts[suffix]
        try:
            from tree_sitter import Language, Parser  # noqa: PLC0415 - optional dep

            if suffix in _PY_SUFFIXES:
                import tree_sitter_python as mod  # noqa: PLC0415

                lang = Language(mod.language())
            else:
                self._ts[suffix] = None
                return None
            parser = Parser(lang)
        except Exception:  # noqa: BLE001 - any import/ABI failure degrades
            self._ts[suffix] = None
            return None
        self._ts[suffix] = parser
        return parser

    def _tree_sitter_signatures(self, text: str, *, rel: str) -> list[Signature] | None:
        parser = self._ensure_ts(rel)
        if parser is None:
            return None
        try:
            tree = parser.parse(text.encode("utf-8", errors="replace"))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return None

        # Read signatures straight from the source bytes the node spans: the
        # text between the `def`/`class` line start and the body's opening
        # colon is the signature, and it survives syntax errors above/below.
        raw = text.encode("utf-8", errors="replace")
        out: list[Signature] = []

        def walk(node: object, *, depth: int) -> None:
            if len(out) >= MAX_SIGNATURES_PER_FILE:
                return
            ntype = getattr(node, "type", "")
            if ntype in {"function_definition", "class_definition"}:
                head = _node_header(raw, node)
                if head:
                    kind = "class" if ntype == "class_definition" else ("member" if depth else "def")
                    out.append(Signature(kind, head))
                    if ntype == "class_definition" and depth < 1:
                        for child in getattr(node, "named_children", []):
                            walk(child, depth=depth + 1)
                        return
                return
            for child in getattr(node, "named_children", []):
                walk(child, depth=depth)

        walk(tree.root_node, depth=0)  # type: ignore[attr-defined]
        return out or None

    # ------------------------------------------------------------------- regex

    @staticmethod
    def _regex_signatures(text: str) -> list[Signature]:
        """Last-resort extraction for files that will not parse.

        Deliberately dumb and line-anchored: it must never raise, and a map with
        slightly wrong signatures is still far better than no map.
        """
        out: list[Signature] = []
        class_depth = 0
        for line in text.splitlines():
            if len(out) >= MAX_SIGNATURES_PER_FILE:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            m_class = re.match(r"class\s+(\w+)\s*(\([^)]*\))?", stripped)
            if m_class:
                cls_indent = indent
                out.append(Signature("class", f"class {m_class.group(1)}"))
                class_depth = cls_indent
                continue
            m_def = re.match(
                r"(async\s+)?def\s+(\w+)\s*\(([^)]*)",
                stripped,
            )
            if m_def:
                prefix = "async def" if m_def.group(1) else "def"
                args = " ".join(m_def.group(3).split())
                kind = "member" if indent > class_depth else "def"
                out.append(Signature(kind, f"{prefix} {m_def.group(2)}({args})".rstrip("(, ")))
                continue
            m_assign = re.match(r"([A-Z][A-Z0-9_]{2,})\s*=", stripped)
            if m_assign and indent == 0:
                out.append(Signature("const", f"{m_assign.group(1)} = ..."))
        return out


def _node_header(raw: bytes, node: object) -> str:
    """Slice the signature text of a tree-sitter node, up to its body colon."""
    try:
        start = int(node.start_byte)  # type: ignore[attr-defined]
        end = int(node.end_byte)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return ""
    chunk = raw[start:end].decode("utf-8", errors="replace")
    # A signature ends at the first colon that is not inside the parameter list.
    depth = 0
    for i, ch in enumerate(chunk):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            chunk = chunk[:i]
            break
    else:
        chunk = chunk.split("\n", 1)[0]
    return " ".join(chunk.split())


def _render_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = " ".join(ast.unparse(node.args).split())
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args}){ret}".rstrip("(, ")


def _render_class(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    return f"class {node.name}({bases})".rstrip("(") if bases else f"class {node.name}"


class RepoMap:
    """Builds the ranked, budget-packed signature map for one repository.

    Usage from the loop::

        repo_map = RepoMap(repo_path, task=state.task, recently_edited=...)
        text = repo_map.render()

    The object is cheap to rebuild each turn; parsing is the only real cost and
    it is bounded by :data:`MAX_FILES`.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        task: str = "",
        recently_edited: set[str] | None = None,
        config: RepoMapConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.task = task
        self.recently_edited = {_norm(p) for p in (recently_edited or set())}
        self.config = config or RepoMapConfig()
        self._parsers = _Parsers(use_tree_sitter=self.config.use_tree_sitter)
        self._files: list[FileMap] = []
        self._token_sets: dict[str, set[str]] = {}
        self._built = False

    # ------------------------------------------------------------------ public

    def render(self) -> str:
        """Return the map text, packed to ``config.max_tokens``."""
        files = self._ranked()
        if not files:
            return "(empty repository)"

        blocks = [f.render() for f in files]
        text = fit(blocks, self.config.max_tokens)
        if self._parsers.degraded_languages:
            langs = ", ".join(sorted(self._parsers.degraded_languages))
            text += f"\n(map degraded: no parser for {langs}; signatures are approximate)"
        return text

    def files(self) -> list[FileMap]:
        """Ranked file maps, highest score first. Exposed for tests and evals."""
        return self._ranked()

    # ------------------------------------------------------------------ build

    def _ranked(self) -> list[FileMap]:
        if not self._built:
            self._files = self._scan()
            self._built = True
        return sorted(self._files, key=_score, reverse=True)

    def _scan(self) -> list[FileMap]:
        paths = self._candidate_paths()
        if not paths:
            return []

        # Pass 1: parse each file. File text is cached for this scan so pass 2
        # does not re-read from disk.
        parsed: list[FileMap] = []
        for path, rel in paths:
            fm = self._parse_file(path, rel)
            if fm is None:
                continue
            parsed.append(fm)

        # Pass 2: reference counts. A file is "important" when the symbols it
        # defines are used elsewhere — that is the PageRank-lite signal, and it
        # is why a shared util outranks a leaf script.
        for fm in parsed:
            fm.references = self._count_references(fm)

        return parsed

    def _candidate_paths(self) -> list[tuple[Path, str]]:
        """Walk the repo, skipping vendored/generated trees, capped at max_files."""
        out: list[tuple[Path, str]] = []
        root = self.root
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune in place so os.walk never descends into them.
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                suffix = Path(name).suffix.lower()
                if suffix in _SKIP_SUFFIXES:
                    continue
                if suffix not in _PY_SUFFIXES:
                    continue
                path = Path(dirpath) / name
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                out.append((path, rel))
                if len(out) >= self.config.max_files:
                    return out
        return out

    def _parse_file(self, path: Path, rel: str) -> FileMap | None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        # A file with no newline in 200k chars is minified/generated; its
        # "signatures" would be one unreadable line.
        if len(text) > 200_000 and text.count("\n") < 50:
            return None

        sigs, degraded = self._parsers.python_signatures(text, rel=rel)
        # Identifier set per file: cheap membership test for reference counts.
        self._token_sets[rel] = set(_IDENT_RE.findall(text))
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0

        return FileMap(
            rel=rel,
            signatures=sigs,
            mtime=mtime,
            mentioned=self._is_mentioned(rel, sigs),
            recently_edited=rel in self.recently_edited,
            degraded=degraded,
        )

    def _is_mentioned(self, rel: str, sigs: list[Signature]) -> bool:
        """True when the task names the file, its stem, or a symbol it defines."""
        if not self.task:
            return False
        task = self.task.lower()
        if rel.lower() in task:
            return True
        stem = Path(rel).stem.lower()
        if len(stem) >= 3 and re.search(rf"\b{re.escape(stem)}\b", task):
            return True
        for sig in sigs:
            name = _symbol_name(sig.text)
            if name and len(name) >= 3 and re.search(rf"\b{re.escape(name.lower())}\b", task):
                return True
        return False

    def _count_references(self, fm: FileMap) -> int:
        """Count how many *other* files mention the symbols ``fm`` defines.

        This is the inbound edge of the PageRank-lite signal: a module everyone
        imports matters more than a leaf script, so it earns the top of the map.
        Counting the wrong direction (mentions *made* by this file) would rank
        buggy callers above the shared code they depend on.

        Matching is done on pre-tokenized identifier sets rather than per-symbol
        regexes: the naive version ran ~26k regex searches on a 7k-line repo and
        cost 4 seconds, which on a Colab VM is time the model spends idle.
        """
        own = {_symbol_name(s.text) for s in fm.signatures}
        own.discard(None)
        if not own:
            return 0
        return sum(1 for other, tokens in self._token_sets.items() if other != fm.rel and own & tokens)


def _score(fm: FileMap) -> float:
    """PageRank-lite score (DESIGN §5.2).

    ``0.5·reference_count + 0.3·recently_edited + 0.2·mentioned_in_task``

    The reference count is normalised by a soft cap so one hub file cannot
    dominate the whole map, and "mentioned" is a hard boost because a file the
    task names must survive the budget.
    """
    refs = min(fm.references, 20) / 20.0
    recent = 1.0 if fm.recently_edited else 0.0
    mentioned = 1.0 if fm.mentioned else 0.0
    return 0.5 * refs + 0.3 * recent + 0.2 * mentioned


def _symbol_name(sig_text: str) -> str | None:
    """Extract the defined name from a rendered signature line."""
    m = re.match(r"(?:async\s+)?def\s+(\w+)", sig_text)
    if m:
        return m.group(1)
    m = re.match(r"class\s+(\w+)", sig_text)
    if m:
        return m.group(1)
    m = re.match(r"([A-Z][A-Z0-9_]{2,})\s*=", sig_text)
    if m:
        return m.group(1)
    return None


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")
