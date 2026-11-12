"""Notebook tools: read, edit, and execute `.ipynb` files (PRD FR-TOL-6).

Why notebooks get their own module
----------------------------------
A notebook is JSON, not text, so SEARCH/REPLACE on the raw file is a trap: the
model would be editing escaped `\n` inside a string, and a half-matched cell
would produce a file that opens in Colab as a corrupt notebook. Three
consequences shape this module.

**Cells are addressed by id, never by index.** Indices shift under insert and
delete, so "edit cell 3" silently edits the wrong code the moment an earlier cell
is removed. `nbformat` 4.5+ gives every cell an `id`; `rust_cell()` assigns
stable ones on first read, so an old notebook acquires ids before it is edited
rather than after a wrong edit.

**Read-before-write applies here too** (DESIGN §5.4). `edit_notebook` refuses a
path the session has not read, exactly like `edit_file`.

**Cell outputs never enter context** (FR-CTX-4). A single `display(df)` can blow
the whole 12k window. Execution reads outputs from a second notebook file, folds
them into a compact report — the last traceback, or a capped tail of stdout — and
hands the full outputs to the loop as artifacts.
"""

from __future__ import annotations

import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..context.artifacts import ArtifactError
from ..safety.policy import Verdict
from .base import ArgSpec, SafetyClass, ToolError, ToolResult
from .fs import _PolicyTool
from .shell import ExecConfig, run_subprocess

__all__ = [
    "CellView",
    "NotebookDoc",
    "ReadNotebookTool",
    "EditNotebookTool",
    "ExecNotebookTool",
    "load_notebook",
    "render_cells",
    "summarise_execution",
    "MAX_CELL_LINES",
    "MAX_TAIL_LINES",
    "MAX_LINE_CHARS",
]

#: Per-cell source cap in the read view. A 600-line cell is a file, not a cell;
#: the model can still read the notebook as text if it truly needs all of it.
MAX_CELL_LINES = 60

#: FR-CTX-4: stdout/stderr that does enter context is capped at 40 lines.
MAX_TAIL_LINES = 40

#: How much of a traceback to keep. The last frame is what names the bug.
MAX_TRACEBACK_LINES = 30

_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# --------------------------------------------------------------------- document


@dataclass
class CellView:
    """One cell as the model sees it: id, type, source. Never outputs."""

    id: str
    index: int  # 0-based position, shown only as a hint
    cell_type: str
    source: str
    execution_count: int | None = None
    outputs: int = 0

    @property
    def line_count(self) -> int:
        return len(self.source.splitlines())


@dataclass
class NotebookDoc:
    """A parsed notebook plus the ids we normalised.

    Held separately from `nbformat` so the reader and the renderer stay testable
    without the dependency; `nbformat` is only required for reading/writing the
    real file, and is imported inside the io helpers.
    """

    path: Path
    rel: str
    nb: Any  # nbformat.NotebookNode
    assigned_ids: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    @property
    def cells(self) -> list[Any]:
        return list(self.nb.cells)

    @property
    def kernel(self) -> str:
        name = (self.nb.metadata or {}).get("kernelspec", {}).get("name")
        return str(name) if name else "python3"

    @property
    def language(self) -> str:
        meta = (self.nb.metadata or {}).get("language_info", {})
        return str(meta.get("name") or "python")

    def views(self) -> list[CellView]:
        return [self.view(i) for i in range(len(self.nb.cells))]

    def view(self, index: int) -> CellView:
        cell = self.nb.cells[index]
        source = cell_source(cell)
        outputs = cell.get("outputs", []) or []
        return CellView(
            id=str(cell.get("id") or ""),
            index=index,
            cell_type=str(cell.get("cell_type", "code")),
            source=source,
            execution_count=cell.get("execution_count"),
            outputs=len(outputs),
        )

    def find(self, cell_id: str) -> tuple[int, Any]:
        """Locate a cell by id, with a corrective message listing real ids.

        Ids are the contract. An unknown id is a model mistake that costs a turn,
        so the error names every valid id rather than saying "not found".
        """
        cells = self.nb.cells
        for index, cell in enumerate(cells):
            if str(cell.get("id") or "") == cell_id:
                return index, cell
        known = _id_list(cells)
        if cell_id.strip().isdigit():
            raise ToolError(
                f"cells are addressed by id, not index. {_valid_ids_note(known)}"
            )
        raise ToolError(f"no cell with id {cell_id!r}. {_valid_ids_note(known)}")

    def markdown_heading(self, cell_id: str) -> str:
        _, cell = self.find(cell_id)
        return cell_source(cell).strip().splitlines()[0].lstrip("# ").strip() if cell else ""


def _id_list(cells: list[Any]) -> list[str]:
    return [str(c.get("id") or "") for c in cells]


def _valid_ids_note(known: list[str]) -> str:
    shown = ", ".join(k for k in known if k) or "(none)"
    return f"Valid cell ids: {shown}."


def cell_source(cell: Any) -> str:
    """Notebook cell source is a str or a list of lines; normalise to str."""
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(str(s) for s in src)
    return str(src)


def set_cell_source(cell: Any, text: str) -> None:
    """Store source as a list of lines, which is what nbformat and git diffs like."""
    cell["source"] = text.splitlines(keepends=True)


def load_notebook(path: Path, rel: str) -> NotebookDoc:
    """Read and normalise a notebook, assigning missing cell ids.

    `nbformat` validation is deliberately *not* run here: Colab writes notebooks
    that are missing the odd required field, and refusing to read them would make
    the common case the broken case. Validation runs on write, where it protects
    the file we are about to hand back to Colab.
    """
    try:
        import nbformat
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ToolError(
            "reading notebooks needs nbformat. Install it with: pip install 'coagent[ui]'"
        ) from exc
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot read {rel}: {exc}") from exc
    try:
        nb = nbformat.reads(raw, as_version=4)
    except Exception as exc:  # noqa: BLE001 - nbformat raises unrelated types
        raise ToolError(
            f"{rel} is not a valid notebook ({exc}). Fix the JSON, or use read_file "
            "to inspect it as text."
        ) from exc

    doc = NotebookDoc(path=path, rel=rel, nb=nb, fingerprint=_fingerprint(raw))
    doc.assigned_ids = assign_cell_ids(nb)
    doc.metadata = dict(nb.get("metadata") or {})
    return doc


def assign_cell_ids(nb: Any) -> int:
    """Give every cell a valid id, returning how many were assigned.

    Colab still emits notebooks under nbformat 4.4, where cells have no id at
    all. Assigning on read means the model always has a handle, and the id does
    not change later because of an unrelated edit.
    """
    assigned = 0
    used: set[str] = set()
    for index, cell in enumerate(nb.cells):
        existing = str(cell.get("id") or "")
        if _CELL_ID_RE.match(existing) and existing not in used:
            used.add(existing)
            continue
        candidate = f"c{index:02d}"
        suffix = 0
        while candidate in used:
            suffix += 1
            candidate = f"c{index:02d}-{suffix}"
        cell["id"] = candidate
        used.add(candidate)
        assigned += 1
    return assigned


def _fingerprint(raw: str) -> str:
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------- render


def render_cells(doc: NotebookDoc, *, max_cell_lines: int = MAX_CELL_LINES, header: bool = True) -> str:
    """The model's view of a notebook: one block per cell, sources only.

    Outputs are summarised as a count, not printed, because FR-CTX-4 says cell
    outputs are artifacts and a `display(df)` would evict the window. The model
    sees *that* a cell ran and *where* it failed, never the dataframe.

    ``header=False`` drops the notebook-level line so an edit diff shows only the
    cell change; nbformat bumps its minor version on save, which would otherwise
    appear as a spurious first-line change on every edit.
    """
    cells = doc.cells
    if not cells:
        return f"{doc.rel}: empty notebook (no cells)"

    out: list[str] = []
    if header:
        head = (
            f"{doc.rel} — {len(cells)} cells, kernel {doc.kernel}, "
            f"nbformat {doc.nb.get('nbformat', '?')}.{doc.nb.get('nbformat_minor', '?')}"
        )
        if doc.assigned_ids:
            head += f" ({doc.assigned_ids} cell id(s) assigned)"
        out.extend([head, ""])

    for view in doc.views():
        label = f"cell {view.id} [{view.cell_type}"
        if view.cell_type == "code":
            label += f", exec {view.execution_count if view.execution_count is not None else '-'}"
            if view.outputs:
                label += f", {view.outputs} output(s) hidden"
        label += f"] lines {view.index + 1}"
        out.append(f"--- {label} ---")

        body = view.source.rstrip("\n")
        if not body.strip():
            out.append("(empty)")
        else:
            lines = body.splitlines()
            if len(lines) > max_cell_lines:
                head = lines[: max_cell_lines - 5]
                tail = lines[-4:]
                out.extend(head)
                out.append(f"… [{len(lines) - max_cell_lines} lines omitted in this cell]")
                out.extend(tail)
            else:
                out.extend(lines)
        out.append("")
    return "\n".join(out).rstrip()


def write_notebook(doc: NotebookDoc) -> None:
    """Serialise back to disk, validating first so Colab always opens it."""
    try:
        import nbformat
    except ImportError as exc:  # pragma: no cover
        raise ToolError("writing notebooks needs nbformat") from exc

    nb = doc.nb
    # Writing sets a minimum version that supports ids; without this, a notebook
    # read as 4.4 would serialize ids it is not allowed to have.
    nb.nbformat = max(int(nb.get("nbformat", 4)), 4)
    if nb.nbformat == 4:
        nb.nbformat_minor = max(int(nb.get("nbformat_minor", 0)), 5)
    try:
        nbformat.validate(nb)
    except Exception as exc:  # noqa: BLE001 - ValidationError needs jsonschema
        raise ToolError(f"refusing to write an invalid notebook: {exc}") from exc

    text = nbformat.writes(nb, version=4)
    doc.path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


# ------------------------------------------------------------------ python gate


def _python_syntax_error(cell: Any, language: str) -> str:
    """Reject a cell that will not parse as Python (FR-VER-4, DESIGN §6.1).

    IPython magics (`%matplotlib`, `!pip`) and `%%` cells are not parseable as
    Python but are perfectly valid notebook cells, so they are skipped rather
    than flagged. Anything else that fails `ast.parse` is a real error and is
    caught before it reaches the file — the alternative is a model that only
    learns about the typo after a 300 s execution.
    """
    if language.lower() not in {"python", "python3", ""}:
        return ""
    if cell.get("cell_type") != "code":
        return ""
    source = cell_source(cell)
    if not source.strip():
        return ""
    stripped = source.lstrip()
    if stripped.startswith("%%") or stripped.startswith("!"):
        return ""
    # Line magics and shell escapes are valid in IPython only; neutralise them so
    # the rest of the cell can still be checked.
    neutralised = "\n".join(
        "" if line.lstrip().startswith(("%", "!", "?")) else line
        for line in source.splitlines()
    )
    import ast

    try:
        ast.parse(neutralised)
    except SyntaxError as exc:
        return f"{exc.msg} at line {exc.lineno}"
    return ""


# --------------------------------------------------------------------- output


@dataclass
class ExecutionOutput:
    """One output from an executed notebook, as an artifact candidate."""

    cell_id: str
    kind: str  # "stream" | "error" | "image" | "html" | "text"
    mime: str = ""
    data: str = ""


def parse_outputs(doc: NotebookDoc) -> list[ExecutionOutput]:
    """Flatten every cell output into artifacts (never into context verbatim)."""
    collected: list[ExecutionOutput] = []
    for view in doc.views():
        _, cell = doc.find(view.id)
        for output in cell.get("outputs", []) or []:
            kind = str(output.get("output_type", ""))
            if kind == "stream":
                collected.append(
                    ExecutionOutput(
                        view.id, "stream", str(output.get("name", "stdout")), "".join(output.get("text", ""))
                    )
                )
            elif kind == "error":
                collected.append(
                    ExecutionOutput(
                        view.id,
                        "error",
                        "traceback",
                        "\n".join(output.get("traceback", []))
                        or f"{output.get('ename')}: {output.get('evalue')}",
                    )
                )
            elif kind in {"display_data", "execute_result"}:
                data = output.get("data", {}) or {}
                for mime in ("image/png", "image/jpeg", "text/html", "text/plain"):
                    if mime in data:
                        value = data[mime]
                        if isinstance(value, list):
                            value = "".join(value)
                        kind_name = "image" if mime.startswith("image/") else (
                            "html" if mime == "text/html" else "text"
                        )
                        collected.append(ExecutionOutput(view.id, kind_name, mime, str(value)))
                        break
    return collected


def summarise_execution(doc: NotebookDoc, *, tail_lines: int = MAX_TAIL_LINES) -> tuple[str, bool, dict[str, Any]]:
    """Condense outputs into what may enter context (FR-CTX-4).

    Returns `(summary, ok, meta)`. Only the last traceback and a capped tail of
    stdout are included; the rest is in `meta["outputs"]` for the artifact tools.
    """
    outputs = parse_outputs(doc)
    errors = [o for o in outputs if o.kind == "error"]
    streams = [o for o in outputs if o.kind == "stream"]
    images = [o for o in outputs if o.kind == "image"]
    html = [o for o in outputs if o.kind == "html"]

    ok = not errors
    sections: list[str] = []
    if errors:
        last = errors[-1]
        tb_lines = [_cap_line(line) for line in _strip_ansi(last.data).splitlines()]
        kept = tb_lines[-MAX_TRACEBACK_LINES:]
        sections.append(f"Traceback (cell {last.cell_id}, last {len(kept)} lines):\n" + "\n".join(kept))
    text_out = "\n".join(_strip_ansi(o.data) for o in streams).strip()
    if text_out:
        lines = [_cap_line(line) for line in text_out.splitlines()]
        if len(lines) > tail_lines:
            omitted = len(lines) - tail_lines
            sections.append(
                f"stdout/stderr (tail {tail_lines} of {len(lines)} lines, {omitted} omitted):\n"
                + "\n".join(lines[-tail_lines:])
            )
        else:
            # Join the capped lines, not `text_out`: a single huge line must not
            # bypass the 40-line budget (FR-CTX-4).
            sections.append("stdout/stderr:\n" + "\n".join(lines))

    if images or html:
        bits = []
        if images:
            bits.append(f"{len(images)} image(s)")
        if html:
            bits.append(f"{len(html)} HTML output(s)")
        sections.append("rendered: " + ", ".join(bits) + " (sent to artifacts, not context)")

    summary = "\n\n".join(sections) if sections else "executed with no output"
    meta = {
        "ok": ok,
        "cells_with_output": len({o.cell_id for o in outputs}),
        "images": len(images),
        "errors": len(errors),
        "outputs": outputs,
    }
    return summary, ok, meta


#: A line cap as well as a line count. `print(x * 200)` is one line and would
#: otherwise walk straight past the 40-line budget (FR-CTX-4).
MAX_LINE_CHARS = 300


def _cap_line(line: str, limit: int = MAX_LINE_CHARS) -> str:
    return line if len(line) <= limit else line[: limit - 20] + " …[truncated]"


def _decode_base64(text: str) -> bytes | None:
    """Decode an image payload, tolerating the newlines nbformat keeps."""
    import base64
    import binascii

    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError):
        return None


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# --------------------------------------------------------------------- tools


class ReadNotebookTool(_PolicyTool):
    name = "read_notebook"
    description = (
        "Read a Jupyter notebook (.ipynb) as cells: every cell is shown with its "
        "id, type, and source. Cell outputs are NOT shown (they go to artifacts). "
        "Read before editing, exactly like read_file."
    )
    safety = SafetyClass.READ
    args = (
        ArgSpec("path", "string", "Repo-relative path to the .ipynb file."),
        ArgSpec("max_cell_lines", "integer", "Truncate each cell to this many lines.", required=False, default=MAX_CELL_LINES),
    )

    def run(self, path: str, max_cell_lines: int = MAX_CELL_LINES) -> ToolResult:
        resolved = self._read_path(path)
        if not resolved.is_file():
            raise ToolError(f"{path} does not exist. Use list_dir to find it.")
        doc = load_notebook(resolved, self.session.rel(resolved))
        self.session.mark_read(resolved)
        body = render_cells(doc, max_cell_lines=max(5, min(int(max_cell_lines), 400)))
        return ToolResult.success(
            body,
            path=doc.rel,
            cells=len(doc.cells),
            ids_assigned=doc.assigned_ids,
            fingerprint=doc.fingerprint,
        )


class EditNotebookTool(_PolicyTool):
    name = "edit_notebook"
    description = (
        "Edit a notebook cell by id. Modes: `replace_source` (rewrite the cell), "
        "`insert` (add a cell after a given id, or at the top with id \"\"), "
        "`delete` (remove the cell), `replace_metadata` (set kernelspec name). "
        "Read the notebook first. The file must stay a valid notebook."
    )
    safety = SafetyClass.WRITE
    args = (
        ArgSpec("path", "string", "Repo-relative path to the .ipynb file."),
        ArgSpec(
            "mode",
            "string",
            "One of: replace_source, insert, delete, replace_metadata.",
            enum=("replace_source", "insert", "delete", "replace_metadata"),
        ),
        ArgSpec("cell_id", "string", "Target cell id. Use \"\" with mode=insert to insert at the top.", required=False, default=""),
        ArgSpec("cell_type", "string", "Cell type for insert: code or markdown.", required=False, default="code", enum=("code", "markdown")),
        ArgSpec("source", "string", "New cell source (replace_source, insert).", required=False, default=""),
        ArgSpec("kernel", "string", "Kernel name for replace_metadata, e.g. python3.", required=False, default=""),
        ArgSpec("cite", "string", "Optional hint about why you are editing (advisory).", required=False, default=""),
    )

    def run(
        self,
        path: str,
        mode: str,
        cell_id: str = "",
        cell_type: str = "code",
        source: str = "",
        kernel: str = "",
        cite: str = "",
    ) -> ToolResult:
        resolved = self._write_path(path)
        if not self.session.was_read(resolved):
            raise ToolError(
                f"{path} has not been read this session. Call read_notebook on it first, "
                "then edit by cell id."
            )
        if not resolved.is_file():
            raise ToolError(f"{path} does not exist; create it with write_file (JSON) or `coagent init`.")

        doc = load_notebook(resolved, self.session.rel(resolved))
        before = render_cells(doc, header=False)
        language = doc.language
        detail = ""

        if mode == "replace_source":
            if not cell_id:
                raise ToolError("replace_source needs `cell_id`. Use read_notebook to get the ids.")
            index, cell = doc.find(cell_id)
            if not source.strip() and cell.get("cell_type") == "code":
                raise ToolError(
                    f"the new source for cell {cell_id} is empty. To remove a cell use mode=delete."
                )
            set_cell_source(cell, source)
            error = _python_syntax_error(cell, language)
            if error:
                raise ToolError(
                    f"cell {cell_id} would not parse as Python ({error}). Fix the source and retry; "
                    "the notebook was not changed."
                )
            cell["outputs"] = []  # stale outputs would describe code that no longer exists
            cell["execution_count"] = None
            detail = f"replaced source of cell {cell_id} (index {index + 1})"

        elif mode == "insert":
            if not source.strip():
                raise ToolError("insert needs non-empty `source`.")
            new_cell = self._new_cell(cell_type, source)
            error = _python_syntax_error(new_cell, language)
            if error:
                raise ToolError(
                    f"the new cell would not parse as Python ({error}). The notebook was not changed."
                )
            if not cell_id:
                doc.nb.cells.insert(0, new_cell)
                detail = f"inserted {cell_type} cell at the top"
            else:
                index, _ = doc.find(cell_id)
                doc.nb.cells.insert(index + 1, new_cell)
                detail = f"inserted {cell_type} cell after {cell_id} (new index {index + 2})"
            detail += f" with id {new_cell['id']}"

        elif mode == "delete":
            if not cell_id:
                raise ToolError("delete needs `cell_id`.")
            index, _ = doc.find(cell_id)
            if len(doc.nb.cells) <= 1:
                raise ToolError(
                    "refusing to delete the only cell: an empty notebook is usually not what "
                    "you mean. Use replace_source instead."
                )
            doc.nb.cells.pop(index)
            detail = f"deleted cell {cell_id} (was index {index + 1})"

        elif mode == "replace_metadata":
            if not kernel:
                raise ToolError("replace_metadata needs `kernel`, e.g. \"python3\".")
            meta = dict(doc.nb.get("metadata") or {})
            spec = dict(meta.get("kernelspec") or {})
            spec["name"] = kernel
            spec.setdefault("display_name", kernel)
            spec.setdefault("language", language or "python")
            meta["kernelspec"] = spec
            doc.nb["metadata"] = meta
            detail = f"set kernelspec.name to {kernel}"
        else:  # pragma: no cover - the enum blocks this
            raise ToolError(f"unknown mode {mode!r}")

        try:
            write_notebook(doc)
        except OSError as exc:
            raise ToolError(f"cannot write {path}: {exc}") from exc
        self.session.mark_written(resolved)

        after = render_cells(doc, header=False)
        from .fs import _unified_diff

        diff = _unified_diff(before, after, doc.rel)
        return ToolResult.success(
            f"{detail} in {path}\n{diff}",
            path=doc.rel,
            diff=diff,
            mode=mode,
            cell_id=cell_id,
            cite=cite,
        )

    def _new_cell(self, cell_type: str, source: str) -> Any:
        from nbformat.v4 import new_code_cell, new_markdown_cell

        maker = new_markdown_cell if cell_type == "markdown" else new_code_cell
        return maker(source=source)


class ExecNotebookTool(_PolicyTool):
    name = "exec_notebook"
    description = (
        "Execute a notebook in a subprocess via nbconvert, in place, with a "
        "300 s timeout. Returns only the last traceback or a capped tail of stdout; "
        "images and the full outputs become artifacts. Use this to prove a cell runs."
    )
    safety = SafetyClass.EXEC
    args = (
        ArgSpec("path", "string", "Repo-relative path to the .ipynb file."),
        ArgSpec("timeout_s", "integer", "Timeout override in seconds.", required=False, default=0),
    )

    def __init__(self, session: Any) -> None:
        super().__init__(session)
        self.session = session
        self.exec_cfg: ExecConfig = session.exec_cfg

    def run(self, path: str, timeout_s: int = 0) -> ToolResult:
        resolved = self._write_path(path)
        if not resolved.is_file():
            raise ToolError(f"{path} does not exist; nothing to execute.")
        doc = load_notebook(resolved, self.session.rel(resolved))
        self.session.mark_written(resolved)  # execution rewrites the file in place

        limit = int(timeout_s) or self.exec_cfg_notebook_timeout()
        outcome = self._execute_in_subprocess(resolved, doc, timeout_s=limit)

        # nbconvert writes into the file itself. Reading it back is how we get
        # outputs without trusting stdout, which nbconvert keeps empty.
        try:
            executed = load_notebook(resolved, doc.rel)
        except ToolError as exc:
            return ToolResult.failure(
                f"the notebook was rewritten by nbconvert but cannot be re-read ({exc}). "
                "Restore it with git checkout and re-run."
            )

        summary, ok, meta = summarise_execution(executed)
        # Two ways to exceed the budget: nbconvert's own `ExecutePreprocessor`
        # timeout (which exits 1 with a CellTimeoutError and keeps the notebook),
        # or our process-group kill when even nbconvert hangs. Both are the same
        # thing to the model, so they are reported the same way.
        cell_timeout = outcome.output.strip() and _looks_like_cell_timeout(outcome.output)
        timed_out = outcome.timed_out or bool(cell_timeout)
        if cell_timeout:
            ok = False

        if timed_out:
            status = "timeout"
        elif not ok:
            # nbconvert exits 0 even with failing cells (`allow_errors`), so the
            # status line must report the cell verdict, not the exit code.
            status = f"{meta['errors']} failing cell(s)"
        else:
            status = f"exit {outcome.returncode}"
        header = f"$ jupyter nbconvert --execute {doc.rel}\n[{status}, {outcome.duration_s:.1f}s, limit {limit}s]"
        body = f"{header}\n\n{summary}"
        if timed_out:
            killed = " and its process group was killed" if outcome.timed_out else ""
            body += (
                f"\n\n[execution exceeded {limit}s{killed}. A long-running cell is the "
                "usual cause; narrow the work or raise `tools.notebook_timeout_s`.]"
            )
        elif outcome.output.strip() and not ok and not meta["errors"]:
            # nbconvert's own log is only useful when the notebook came back
            # without a traceback (e.g. it never started). When a cell failed, the
            # traceback above already says everything and the log is 20 lines of
            # ANSI-coloured noise competing for the observation slot.
            body += f"\n\nnbconvert output (tail):\n{_tail(outcome.output, 12)}"

        images = meta.pop("outputs", [])
        # Cell outputs are artifacts (FR-CTX-4). Saving them here, rather than
        # asking the model to call save_artifact again, is what makes "the cell
        # ran and the figure saved" one action instead of two — and the model is
        # not in the loop for bytes it never sees.
        saved = self._store_outputs(images, doc.rel)
        if saved:
            body += "\n\nartifacts: " + ", ".join(r.path for r in saved)

        return ToolResult(
            ok=ok and not timed_out,
            content=body,
            truncated=outcome.truncated,
            meta={
                **meta,
                "path": doc.rel,
                "timed_out": timed_out,
                "returncode": outcome.returncode,
                "outputs": images,
                "artifacts": [r.path for r in saved],
            },
        )

    def _store_outputs(self, outputs: list[Any], notebook_rel: str) -> list[Any]:
        """Store images and HTML from a run; streams are not artifacts.

        A stream output is the cell's own logging and belongs in the tail; a
        figure or a rendered table is a product. Each is saved under a name
        derived from the cell id, so re-running replaces the previous file
        instead of piling up `output-1.png`, `output-2.png`, …
        """
        store = getattr(self.session, "artifact_store", None)
        if store is None or not outputs:
            return []
        stem = Path(notebook_rel).stem or "notebook"
        saved = []
        for output in outputs:
            if output.kind not in {"image", "html"}:
                continue
            ext = ".png" if output.mime == "image/png" else (
                ".jpg" if output.mime == "image/jpeg" else ".html"
            )
            if output.kind == "image":
                data = _decode_base64(output.data)
                if data is None:
                    continue
            else:
                data = output.data.encode("utf-8", errors="replace")
            try:
                saved.append(
                    store.save_bytes(
                        data,
                        f"{stem}-{output.cell_id}{ext}",
                        mime=output.mime or "image/png",
                        subdir="notebooks",
                    )
                )
            except ArtifactError:
                # A failed artifact save must not fail the execution: the cell
                # ran, and that is the fact the verify gate needs.
                continue
        return saved

    def exec_cfg_notebook_timeout(self) -> int:
        return int(getattr(self.exec_cfg, "notebook_timeout_s", 300) or 300)

    def _execute_in_subprocess(self, path: Path, doc: NotebookDoc, *, timeout_s: int) -> Any:
        """Run nbconvert against a *copy* so a crash cannot corrupt the repo file.

        Copy-execute-move is the safe order: if nbconvert dies mid-write, the
        tracked notebook is untouched and the session keeps a clean tree.
        """
        workdir = Path(tempfile.mkdtemp(prefix="coagent-nb-", dir=str(self._scratch())))
        copy = workdir / path.name
        try:
            self._gate_execution(doc, timeout_s)
            copy.write_bytes(path.read_bytes())
            argv = [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                f"--ExecutePreprocessor.timeout={timeout_s}",
                f"--ExecutePreprocessor.kernel_name={doc.kernel}",
                "--ExecutePreprocessor.allow_errors=True",
                str(copy),
            ]
            outcome = run_subprocess(
                argv,
                cfg=self.exec_cfg,
                cwd=workdir,
                timeout_s=timeout_s + 30,
            )
            if copy.is_file() and not outcome.timed_out:
                path.write_bytes(copy.read_bytes())
            return outcome
        finally:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)

    def _gate_execution(self, doc: NotebookDoc, timeout_s: int) -> None:
        """Execution is gated like shell: the policy sees every code cell.

        A notebook that shells out or installs a package must be approved by the
        same matrix that governs `run_shell`, otherwise `exec_notebook` is a way
        around the approval system.
        """
        policy = self.session.policy
        for view in doc.views():
            if view.cell_type != "code":
                continue
            shell_lines = [
                line for line in view.source.splitlines() if line.lstrip().startswith("!")
            ]
            decision = policy.check_shell(" ".join(shell_lines) if shell_lines else _SHELL_NEUTRAL)
            if decision.verdict is Verdict.DENY:
                raise ToolError(f"cell {view.id} is denied by policy: {decision.reason}")
            if decision.verdict is Verdict.ASK:
                raise ToolError(
                    f"cell {view.id} needs approval before it can run: {decision.reason}. "
                    "Ask the user, or remove the command."
                )

    def _scratch(self) -> Path:
        scratch = self.session.policy.scratch
        if scratch is None:
            scratch = self.session.repo_root / ".coagent" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        return scratch


#: A string no deny rule matches; used when a cell has no shell escapes so the
#: policy check is a no-op rather than a false positive.
_SHELL_NEUTRAL = "python-notebook-cell"


def _tail(text: str, lines: int) -> str:
    parts = _strip_ansi(text).splitlines()
    return "\n".join(parts[-lines:])


#: nbconvert's `ExecutePreprocessor` timeout surfaces as a `CellTimeoutError`
#: traceback and exit 1; it does not hang. Both that and our own kill are a
#: timeout to the model, so both are detected here.
_TIMEOUT_MARKERS = ("CellTimeoutError", "Timeout waiting for execute reply", "CellExecutionTimeout")


def _looks_like_cell_timeout(output: str) -> bool:
    return any(marker in output for marker in _TIMEOUT_MARKERS)
