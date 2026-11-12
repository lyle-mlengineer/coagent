"""Artifact tools: save to the repo/Drive and render in the transcript (FR-TOL-7).

Two tools, two jobs, one store:

* `save_artifact` — takes bytes the agent produced (a plot written by a cell, a
  CSV from a script, a text report) and puts it through the §3.5 size policy. The
  model never chooses the branch: it names a file, the store decides whether the
  bytes are committed or pointed at by hash.
* `display_artifact` — renders an artifact inline in the notebook cell, so the
  user sees the figure rather than a file path (FR-UX-4, US-3). The rendering is
  done by an optional callback: on Colab it is `IPython.display`, on the CLI it is
  a printed path. The loop must work with neither available, so a missing UI is a
  normal result, not an error.

Why `save_artifact` refuses to invent content
---------------------------------------------
The tool takes a *path* to something already on disk, or explicit text. It
deliberately cannot fabricate bytes from a description: an agent that can invent
a plot it never drew is worse than one that says "the cell produced nothing".
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..context.artifacts import ArtifactError, ArtifactRecord, ArtifactStore
from ..safety.policy import Verdict
from .base import ArgSpec, SafetyClass, ToolError, ToolResult
from .fs import _PolicyTool

__all__ = [
    "SaveArtifactTool",
    "DisplayArtifactTool",
    "MAX_SAVE_BYTES",
    "RenderCallback",
]

#: Hard ceiling for one artifact. Matches the store's copy limit; anything larger
#: is a dataset that belongs where it already is.
MAX_SAVE_BYTES = 2 * 1024 * 1024 * 1024

RenderCallback = Callable[[str, str, str], None]


class _ArtifactTool(_PolicyTool):
    """Shared plumbing: the store comes from the session, like the policy does."""

    store: ArtifactStore

    def __init__(self, session: Any) -> None:
        super().__init__(session)
        self.session = session
        self.store = session.artifact_store

    def _require_store(self) -> ArtifactStore:
        if self.store is None:  # pragma: no cover - build_registry always sets it
            raise ToolError(
                "no artifact store is configured for this session; artifact tools are "
                "unavailable. Everything else still works."
            )
        return self.store


class SaveArtifactTool(_ArtifactTool):
    name = "save_artifact"
    description = (
        "Save a produced file (plot, CSV, report, notebook output) as an artifact. "
        "Give a `source` path for a file that already exists, or `text` for content "
        "you have inline. Files under the commit cap are committed under "
        "artifacts/<run>/; larger files go to Drive and are recorded by sha256. "
        "Use this — not write_file — for anything a tool generated."
    )
    safety = SafetyClass.WRITE
    args = (
        ArgSpec("name", "string", "Artifact filename, e.g. 'loss.png' or 'bench.csv'."),
        ArgSpec("source", "string", "Repo-relative or /tmp path of an existing file.", required=False, default=""),
        ArgSpec("text", "string", "Inline text content, used only when `source` is absent.", required=False, default=""),
        ArgSpec("mime", "string", "MIME type, e.g. image/png. Helps pick the extension.", required=False, default=""),
        ArgSpec("subdir", "string", "Subdirectory inside the run dir, e.g. 'plots'.", required=False, default=""),
    )

    def run(
        self,
        name: str,
        source: str = "",
        text: str = "",
        mime: str = "",
        subdir: str = "",
    ) -> ToolResult:
        store = self._require_store()
        if not source and not text:
            raise ToolError(
                "save_artifact needs either `source` (a file that already exists) or "
                "`text`. It will not invent content: run the cell or script that "
                "produces the file first."
            )
        if source and text:
            raise ToolError("give `source` or `text`, not both.")

        infer_mime = mime or _mime_for(name)
        try:
            if source:
                path = self._artifact_source(source)
                record = store.save_file(path, name=name, mime=infer_mime, subdir=subdir)
            else:
                record = store.save_text(text, name, mime=infer_mime or "text/plain", subdir=subdir)
        except ArtifactError as exc:
            raise ToolError(str(exc)) from exc

        self.session.mark_written(store.dir / record.name)
        return ToolResult.success(
            self._describe(record),
            artifact=record.path,
            sha256=record.sha256,
            store=record.store,
            bytes=record.bytes,
            committed=record.in_repo,
        )

    def _artifact_source(self, source: str) -> Path:
        """A produced file may live in the repo or the scratch dir, nowhere else."""
        resolved = self.session.resolve(source)
        decision = self.session.policy.check_read(resolved)
        if decision.verdict is Verdict.DENY:
            raise ToolError(
                f"cannot read {source} to save it: {decision.reason}. Write the file "
                "inside the repo or .coagent/tmp first."
            )
        if not resolved.is_file():
            raise ToolError(
                f"{source} does not exist, so there is nothing to save. Produce the file "
                "first (run the cell or script), then save it."
            )
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise ToolError(f"cannot stat {source}: {exc}") from exc
        if size > MAX_SAVE_BYTES:
            raise ToolError(
                f"{source} is {size / 1024**3:.1f} GB, over the 2 GB artifact limit. "
                "Keep it where it is and record the path instead."
            )
        if size == 0:
            raise ToolError(
                f"{source} is empty. An empty artifact is almost always a failed job; "
                "check the cell or script that was meant to write it."
            )
        return resolved

    @staticmethod
    def _describe(record: ArtifactRecord) -> str:
        where = (
            "committed"
            if record.in_repo
            else f"not committed ({record.store})"
        )
        body = (
            f"saved {record.path} ({record.human_size}, sha256 {record.sha256[:12]}, {where})"
        )
        if record.note:
            body += f"\n{record.note}"
        if not record.in_repo and record.location:
            body += f"\nstored at: {record.location}"
        return body


class DisplayArtifactTool(_ArtifactTool):
    name = "display_artifact"
    description = (
        "Render an artifact inline in the notebook/transcript so the user can see it "
        "(an image, HTML, or text). Use this after save_artifact when the user asked "
        "for something visual. Passing nothing renders the whole run manifest."
    )
    safety = SafetyClass.UI
    args = (
        ArgSpec("path", "string", "Artifact path or an in-repo image path to render.", required=False, default=""),
        ArgSpec("mime", "string", "Override the MIME type when it cannot be inferred.", required=False, default=""),
    )

    def run(self, path: str = "", mime: str = "") -> ToolResult:
        store = self._require_store()
        if not path:
            return ToolResult.success(
                store.manifest.render(),
                shown=0,
                manifest=str(store.manifest_path),
            )

        target, resolved_mime = self._locate(path, mime)
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise ToolError(f"cannot read {path} to display it: {exc}") from exc

        kind = _display_kind(resolved_mime)
        renderer: RenderCallback | None = getattr(self.session, "render", None)
        payload = _payload(data, kind)

        if renderer is None:
            # Headless (CLI, tests): the artifact is still on disk and referenced,
            # which is all FR-UX-4 can promise without a notebook.
            return ToolResult.success(
                f"{path} ({resolved_mime or 'unknown type'}) is ready but there is no notebook "
                "surface to render it in; open it at the path above.",
                shown=0,
                path=str(target),
                mime=resolved_mime,
                kind=kind,
            )

        try:
            renderer(kind, payload, resolved_mime)
        except Exception as exc:  # noqa: BLE001 - a UI failure must not end a step
            return ToolResult.success(
                f"{path} was saved but the notebook could not render it ({exc}). "
                "The file is still available at that path.",
                shown=0,
                path=str(target),
                mime=resolved_mime,
                error=str(exc),
            )

        return ToolResult.success(
            f"rendered {path} ({resolved_mime or kind}) in the transcript",
            shown=1,
            path=str(target),
            mime=resolved_mime,
            kind=kind,
        )

    def _locate(self, path: str, mime: str) -> tuple[Path, str]:
        """Find the artifact's bytes: in the run dir, or anywhere readable in jail."""
        store = self._require_store()
        candidate = Path(path)
        resolved = candidate if candidate.is_absolute() else self.session.repo_root / candidate
        resolved = resolved.resolve()

        # A manifest `path` is repo-relative; a Drive artifact's bytes live at
        # `location`. Prefer real bytes wherever they are.
        for record in store.manifest.artifacts:
            if path in {record.path, record.name}:
                if record.location:
                    location = Path(record.location)
                    if location.is_file():
                        return location, mime or record.mime
                break

        decision = self.session.policy.check_read(resolved)
        if decision.verdict is Verdict.DENY:
            raise ToolError(f"cannot display {path}: {decision.reason}")
        if not resolved.is_file():
            raise ToolError(
                f"{path} is not a saved artifact and does not exist. Use save_artifact "
                "first, or pass a path inside the run's artifact directory."
            )
        return resolved, mime or _mime_for(resolved.name)


def _display_kind(mime: str) -> str:
    if mime.startswith("image/"):
        return "image"
    if mime == "text/html":
        return "html"
    return "text"


def _payload(data: bytes, kind: str) -> str:
    """Images travel as base64 so the renderer can be written as a string only."""
    if kind == "image":
        return base64.b64encode(data).decode("ascii")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError(
            "this artifact is binary and its type is not an image; it cannot be rendered "
            "inline. The manifest still records it."
        ) from None


_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".csv": "text/csv",
    ".json": "application/json",
    ".txt": "text/plain",
    ".md": "text/plain",
}


def _mime_for(name: str) -> str:
    return _MIME_BY_EXT.get(Path(name).suffix.lower(), "")


def decode_base64_lenient(text: str) -> bytes | None:
    """Decode a data payload a cell reported, tolerating whitespace/newlines."""
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError):
        return None
