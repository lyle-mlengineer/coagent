"""Filesystem tools: read, write, list, grep, and SEARCH/REPLACE editing.

Design constraints honoured here:

* **Path jail on reads and writes** (DESIGN §6.2, FR-TOL-2) — every path is
  resolved and checked against `Policy` before it is touched.
* **Read before write** (FR-AGT-4) — `edit_file`/`write_file` (rewrite mode)
  refuse a path the session has not read. This kills the hallucinated-edit
  class of failure for the price of a dict lookup.
* **Unique SEARCH match** (DESIGN §6.1) — zero matches returns the closest
  region with line numbers; more than one match is rejected with the offsets.
* **Full rewrite only for new files or files under 200 lines** — large rewrites
  are the biggest source of catastrophic diffs with small models.
* **Deterministic truncation** (FR-CTX-3) — head/tail with line numbers, never a
  silent mid-file drop.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..safety.policy import Policy, Verdict
from .base import ArgSpec, BaseTool, SafetyClass, ToolError, ToolResult

__all__ = [
    "EditFileTool",
    "GrepTool",
    "ListDirTool",
    "ReadFileTool",
    "WriteFileTool",
    "FileSession",
    "truncate_head_tail",
    "apply_search_replace",
    "MAX_REWRITE_LINES",
    "MAX_READ_LINES",
]

MAX_REWRITE_LINES = 200
MAX_READ_LINES = 2000
MAX_READ_BYTES = 512 * 1024


@dataclass
class FileSession:
    """Per-session bookkeeping shared by the filesystem tools."""

    repo_root: Path
    policy: Policy
    read: set[Path] = field(default_factory=set)
    written: set[Path] = field(default_factory=set)
    #: Set by `build_registry`; the notebook exec tool needs the run envelope
    #: (timeouts, scrubbed env, scratch dir) that `ExecConfig` carries.
    exec_cfg: Any = None
    #: Set by `build_registry`: the shared store that applies the §3.5 size policy.
    artifact_store: Any = None
    #: Optional `(kind, payload, mime) -> None` renderer, injected by the notebook
    #: UI. `None` means headless: artifacts are referenced, never drawn.
    render: Any = None

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        try:
            return candidate.resolve()
        except OSError as exc:
            raise ToolError(f"cannot resolve path {path!r}: {exc}") from exc

    def rel(self, path: Path) -> str:
        """Repo-relative POSIX path. Colab is Linux; a stable separator keeps the
        model's `path` arguments consistent across dev machines."""
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    def mark_read(self, path: Path) -> None:
        self.read.add(path)

    def mark_written(self, path: Path) -> None:
        self.written.add(path)
        self.read.add(path)

    def was_read(self, path: Path) -> bool:
        return path in self.read or path in self.written


def truncate_head_tail(text: str, *, max_lines: int = MAX_READ_LINES, start_line: int = 1) -> tuple[str, bool]:
    """Head/tail truncation with line numbers; never a silent mid-file cut."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        numbered = [f"{start_line + i:>6}\t{line}" for i, line in enumerate(lines)]
        return "\n".join(numbered), False
    head_n = max_lines * 6 // 10
    tail_n = max_lines - head_n
    head = lines[:head_n]
    tail = lines[-tail_n:]
    omitted = len(lines) - head_n - tail_n
    out = [f"{start_line + i:>6}\t{line}" for i, line in enumerate(head)]
    out.append(f"      …  [{omitted} lines omitted]")
    out.extend(
        f"{start_line + len(lines) - tail_n + i:>6}\t{line}" for i, line in enumerate(tail)
    )
    return "\n".join(out), True


def _normalize_ws(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines())


def apply_search_replace(source: str, search: str, replace: str, *, path: str = "") -> tuple[str, int]:
    """Apply one SEARCH/REPLACE block, returning new text and the 1-based line.

    Exact match first; then a whitespace/indentation-insensitive fuzzy pass. Any
    ambiguity is an error, never a guess.
    """
    if not search:
        raise ToolError("the `search` block is empty; provide the exact text to replace.")

    occurrences = source.count(search)
    if occurrences == 1:
        idx = source.index(search)
        return source[:idx] + replace + source[idx + len(search) :], source[:idx].count("\n") + 1
    if occurrences > 1:
        starts = [m.start() for m in re.finditer(re.escape(search), source)]
        lines = [source[:s].count("\n") + 1 for s in starts]
        raise ToolError(
            f"the `search` block matches {occurrences} times in {path or 'the file'} "
            f"(lines {', '.join(map(str, lines))}). Add surrounding context so it is unique."
        )

    # Fuzzy pass: compare whitespace-normalized lines.
    src_lines = source.splitlines(keepends=True)
    search_norm = _normalize_ws(search)
    search_len = len(search.strip().splitlines())
    best: tuple[float, int, int] | None = None
    for start in range(len(src_lines)):
        window = "".join(src_lines[start : start + search_len])
        if not window:
            break
        ratio = difflib.SequenceMatcher(None, search_norm, _normalize_ws(window)).ratio()
        if best is None or ratio > best[0]:
            best = (ratio, start, start + search_len)

    if best is not None and best[0] >= 0.75:
        ratio, start, end = best
        region_lines = [
            f"{i + 1:>6}\t{src_lines[i].rstrip()}" for i in range(start, min(end, len(src_lines)))
        ]
        raise ToolError(
            f"no exact match in {path or 'the file'}, but a {ratio:.0%} similar region exists at "
            f"lines {start + 1}-{end}. Re-issue the edit with this exact text:\n"
            + "\n".join(region_lines)
        )
    raise ToolError(
        f"the `search` block does not appear in {path or 'the file'}. "
        "Read the file first and copy the text exactly, including indentation."
    )


class _PolicyTool(BaseTool):
    """Shared plumbing: a `FileSession`-backed resolver with jail checks."""

    session: FileSession

    def __init__(self, session: FileSession) -> None:
        super().__init__(session)
        self.session = session

    def _read_path(self, path: str) -> Path:
        resolved = self.session.resolve(path)
        decision = self.session.policy.check_read(resolved)
        if decision.verdict is Verdict.DENY:
            raise ToolError(decision.reason)
        return resolved

    def _write_path(self, path: str) -> Path:
        resolved = self.session.resolve(path)
        decision = self.session.policy.check_write(resolved)
        if decision.verdict is Verdict.DENY:
            raise ToolError(decision.reason)
        if decision.verdict is Verdict.ASK:
            # Approvals are resolved by the loop before the tool runs; reaching
            # here means the loop skipped the gate.
            raise ToolError(decision.reason)
        return resolved


class ReadFileTool(_PolicyTool):
    name = "read_file"
    description = (
        "Read a text file from the repository. Returns line-numbered content. "
        "Files over 2000 lines are head/tail truncated. Always read before editing."
    )
    safety = SafetyClass.READ
    args = (
        ArgSpec("path", "string", "Repo-relative path to the file."),
        ArgSpec("start", "integer", "1-based first line to read.", required=False, default=1),
        ArgSpec("end", "integer", "Last line to read (inclusive).", required=False, default=0),
    )

    def run(self, path: str, start: int = 1, end: int = 0) -> ToolResult:
        resolved = self._read_path(path)
        if not resolved.exists():
            raise ToolError(f"{path} does not exist. Use list_dir to find the right path.")
        if resolved.is_dir():
            raise ToolError(f"{path} is a directory; use list_dir.")
        if resolved.suffix.lower() == ".ipynb":
            # JSON-notebook text is unreadable as line numbers and every byte of it
            # is escaped source. Point at the tool that shows cells instead.
            raise ToolError(
                f"{path} is a notebook; use read_notebook to see its cells by id, "
                "or edit_notebook to change one."
            )
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            raise ToolError(f"cannot read {path}: {exc}") from exc
        if b"\x00" in raw[:4096]:
            raise ToolError(f"{path} looks binary; use a shell command or an artifact tool.")
        if len(raw) > MAX_READ_BYTES:
            raise ToolError(
                f"{path} is {len(raw) / 1024:.0f} KB, over the {MAX_READ_BYTES // 1024} KB read cap. "
                "Read a line range with start/end, or grep for what you need."
            )
        text = raw.decode("utf-8", errors="replace")
        self.session.mark_read(resolved)

        lines = text.splitlines()
        offset = max(start, 1) - 1
        if end and end >= start:
            lines = lines[offset:end]
        else:
            text = "\n".join(lines[offset:])
            body, truncated = truncate_head_tail(text, start_line=offset + 1)
            return ToolResult.success(body, path=self.session.rel(resolved), truncated=truncated)
        numbered = "\n".join(f"{offset + i + 1:>6}\t{ln}" for i, ln in enumerate(lines))
        return ToolResult.success(numbered, path=self.session.rel(resolved))


class ListDirTool(_PolicyTool):
    name = "list_dir"
    description = "List a directory in the repository, one entry per line with sizes."
    safety = SafetyClass.READ
    args = (
        ArgSpec("path", "string", "Repo-relative directory path.", required=False, default="."),
        ArgSpec("depth", "integer", "Recursion depth (1-3).", required=False, default=1),
    )

    def run(self, path: str = ".", depth: int = 1) -> ToolResult:
        resolved = self._read_path(path)
        if not resolved.is_dir():
            raise ToolError(f"{path} is not a directory.")
        depth = max(1, min(depth, 3))
        out: list[str] = []
        self._walk(resolved, "", depth, out, cap=400)
        self.session.mark_read(resolved)
        if not out:
            return ToolResult.success(f"{path} is empty.")
        return ToolResult.success("\n".join(out), path=self.session.rel(resolved))

    def _walk(self, root: Path, prefix: str, depth: int, out: list[str], cap: int) -> None:
        if len(out) >= cap:
            return
        try:
            entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError as exc:
            out.append(f"{prefix}[unreadable: {exc}]")
            return
        for entry in entries:
            if len(out) >= cap:
                out.append("… [listing truncated]")
                return
            if entry.name == ".git":
                continue
            if entry.is_dir():
                out.append(f"{prefix}{entry.name}/")
                if depth > 1:
                    self._walk(entry, prefix + "  ", depth - 1, out, cap)
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = -1
                out.append(f"{prefix}{entry.name}  ({_human(size)})")


class GrepTool(_PolicyTool):
    name = "grep"
    description = (
        "Search file contents with a Python regular expression. Returns matching "
        "lines with line numbers. Use this instead of reading whole files."
    )
    safety = SafetyClass.READ
    args = (
        ArgSpec("pattern", "string", "Python regular expression."),
        ArgSpec("path", "string", "File or directory to search.", required=False, default="."),
        ArgSpec("glob", "string", "Only search files matching this glob, e.g. '*.py'.", required=False, default=""),
        ArgSpec("max_matches", "integer", "Maximum matching lines to return.", required=False, default=80),
    )

    def run(self, pattern: str, path: str = ".", glob: str = "", max_matches: int = 80) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regex {pattern!r}: {exc}") from exc
        resolved = self._read_path(path)
        max_matches = max(1, min(max_matches, 300))
        files = [resolved] if resolved.is_file() else _iter_files(resolved, glob)

        hits: list[str] = []
        scanned = 0
        for file in files:
            if len(hits) >= max_matches:
                break
            decision = self.session.policy.check_read(file)
            if decision.verdict is not Verdict.ALLOW:
                continue
            scanned += 1
            try:
                with file.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if regex.search(line):
                            rel = self.session.rel(file)
                            hits.append(f"{rel}:{lineno}: {line.rstrip()[:300]}")
                            if len(hits) >= max_matches:
                                break
            except OSError:
                continue
        self.session.mark_read(resolved)
        if not hits:
            return ToolResult.success(
                f"no matches for {pattern!r} in {path} ({scanned} files scanned)", matches=0
            )
        body = "\n".join(hits)
        if len(hits) >= max_matches:
            body += f"\n… [stopped at {max_matches} matches]"
        return ToolResult.success(body, matches=len(hits))


class EditFileTool(_PolicyTool):
    name = "edit_file"
    description = (
        "Replace one exact, unique block of text in an existing file (SEARCH/REPLACE). "
        "The file must have been read this session. Prefer this over write_file."
    )
    safety = SafetyClass.WRITE
    args = (
        ArgSpec("path", "string", "Repo-relative path to edit."),
        ArgSpec("search", "string", "Exact existing text to replace. Must match exactly once."),
        ArgSpec("replace", "string", "Replacement text."),
        ArgSpec("cite", "string", "Optional file:line hint you are anchoring on (advisory).", required=False, default=""),
    )

    def run(self, path: str, search: str, replace: str, cite: str = "") -> ToolResult:
        resolved = self._write_path(path)
        if not self.session.was_read(resolved):
            raise ToolError(
                f"{path} has not been read this session. Call read_file on it first, "
                "then re-issue the edit with the exact text."
            )
        if not resolved.is_file():
            raise ToolError(f"{path} does not exist; use write_file to create it.")
        try:
            original = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"cannot read {path}: {exc}") from exc

        updated, line = apply_search_replace(original, search, replace, path=path)
        syntax_error = _syntax_check(resolved, updated)
        if syntax_error:
            raise ToolError(
                f"the edit would leave {path} unparseable ({syntax_error}). "
                "Re-check the indentation and syntax in your `replace` block."
            )
        try:
            resolved.write_text(updated, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"cannot write {path}: {exc}") from exc
        self.session.mark_written(resolved)

        diff = _unified_diff(original, updated, self.session.rel(resolved))
        return ToolResult.success(
            f"edited {path} at line {line}\n{diff}",
            path=self.session.rel(resolved),
            diff=diff,
            line=line,
            added=max(0, diff.count("\n+") - diff.count("\n+++")),
        )


class WriteFileTool(_PolicyTool):
    name = "write_file"
    description = (
        "Create a new file, or fully rewrite a file under 200 lines. "
        "Large rewrites are refused; use edit_file for existing files."
    )
    safety = SafetyClass.WRITE
    args = (
        ArgSpec("path", "string", "Repo-relative path to write."),
        ArgSpec("content", "string", "Full file content."),
    )

    def run(self, path: str, content: str) -> ToolResult:
        resolved = self._write_path(path)
        exists = resolved.exists()
        if exists:
            if not self.session.was_read(resolved):
                raise ToolError(
                    f"{path} already exists and has not been read this session. "
                    "Read it and use edit_file, or read it first before rewriting."
                )
            try:
                current_lines = len(resolved.read_text(encoding="utf-8").splitlines())
            except OSError as exc:
                raise ToolError(f"cannot read {path}: {exc}") from exc
            if current_lines > MAX_REWRITE_LINES:
                raise ToolError(
                    f"{path} is {current_lines} lines; full rewrites are only allowed under "
                    f"{MAX_REWRITE_LINES}. Use edit_file."
                )
        syntax_error = _syntax_check(resolved, content)
        if syntax_error and not syntax_error.startswith("unknown"):
            raise ToolError(f"{path} would not parse ({syntax_error}); fix the content first.")

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"cannot write {path}: {exc}") from exc
        self.session.mark_written(resolved)
        return ToolResult.success(
            f"wrote {path} ({len(content.splitlines())} lines)",
            path=self.session.rel(resolved),
            created=not exists,
        )


# ----------------------------------------------------------------- helpers


def _iter_files(root: Path, glob: str) -> list[Path]:
    pattern = glob or "*"
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob(pattern)) if p.is_file() and ".git" not in p.parts]


def _syntax_check(path: Path, text: str) -> str:
    """Parse the file if we know how. Returns an error string, or '' if fine."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        import ast

        try:
            ast.parse(text)
        except SyntaxError as exc:
            return f"{exc.msg} at line {exc.lineno}"
        return ""
    if suffix == ".json":
        import json

        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return f"{exc.msg} at line {exc.lineno}"
        return ""
    if suffix in {".toml", ".tml"}:
        try:
            from ..config import tomllib
        except ImportError:  # pragma: no cover
            return ""
        try:
            tomllib.loads(text)
        except Exception as exc:  # noqa: BLE001 - tomllib raises variants by version
            return str(exc)
        return ""
    return ""  # unknown language: the configured linter is the gate (FR-VER-4)


def _unified_diff(before: str, after: str, name: str, context: int = 3) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{name}",
        tofile=f"b/{name}",
        n=context,
        lineterm="",
    )
    return "\n".join(diff)


def _human(size: int) -> str:
    if size < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"
