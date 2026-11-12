"""The artifact store: size policy, hashing, and the manifest (DESIGN §3.5).

What an artifact is
-------------------
Anything a task *produced* rather than wrote by hand: a plot from a cell, a CSV
from a scrape, a notebook's rendered output. The design gives them one manifest
and two storage classes:

===========================  =========================================  =========
Size                         Where                                      In repo
===========================  =========================================  =========
< ``commit_under_mb``        ``artifacts/<run-id>/<name>``              committed
``commit_under_mb`` – 2 GB   Drive ``coagent/artifacts/<sha256>``       pointer
> 2 GB / datasets            Drive or original URL                     pointer
===========================  =========================================  =========

The rule the design states and this module enforces is: **never commit more than
the cap.** A 20 MB PNG in a git repo is the kind of mistake that cannot be undone
without a rewrite; AT-7 exists to prove it does not happen.

Why the manifest is the source of truth
---------------------------------------
The committed bytes and the Drive blob are the same artifact in two places, and
a resumed session on a fresh VM has to be able to tell which is which. So the
manifest records ``{path, sha256, bytes, store, created_by_step}`` for every
entry: the hash is what survives a move to Drive, so a pointer with no bytes in
the repo is still verifiable.

Why Drive is optional
---------------------
FR-BOOT-4: mounting Drive is an explicit user action, and declining must degrade
gracefully. If Drive is not mounted, an over-cap artifact is *not* silently
dropped or forced into git — it stays where it is and the manifest records a
pointer with nowhere to point yet, so the user can collect it. That is a
degraded capability, and the report says so.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "ARTIFACT_ROOT",
    "ArtifactError",
    "ArtifactRecord",
    "ArtifactStore",
    "Manifest",
    "MANIFEST_NAME",
    "MAX_DRIVE_BYTES",
]

#: `artifacts/<run-id>/` at the repo root, per §3.5.
ARTIFACT_ROOT = "artifacts"
MANIFEST_NAME = "manifest.json"

#: 2 GB: past this the design says "Drive or original URL", never a copy we own.
MAX_DRIVE_BYTES = 2 * 1024 * 1024 * 1024

#: A filesystem- and git-safe name. Artifacts come from cell outputs and model
#: arguments, so they are untrusted: no `..`, no separators, no absolute paths.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_IMAGE_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/svg+xml": ".svg"}
_TEXT_EXT = {"text/html": ".html", "text/plain": ".txt", "text/csv": ".csv", "application/json": ".json"}


class ArtifactError(RuntimeError):
    """An artifact could not be stored. The message is the user's next step."""


@dataclass
class ArtifactRecord:
    """One entry in the manifest (§3.5 shape, plus what the UI needs)."""

    path: str
    sha256: str
    bytes: int
    store: str  # "repo" | "drive" | "external"
    created_by_step: int = 0
    #: Where the bytes actually are. For repo artifacts this is the repo-relative
    #: path; for Drive it is the absolute path or the sha-keyed pointer.
    location: str = ""
    mime: str = ""
    name: str = ""
    #: Set when a large artifact could not reach Drive (no mount). The bytes stay
    #: in place and this explains where.
    note: str = ""

    @property
    def in_repo(self) -> bool:
        return self.store == "repo"

    @property
    def human_size(self) -> str:
        size = float(self.bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} GB"

    def render(self) -> str:
        where = "committed" if self.in_repo else f"not committed ({self.store})"
        line = f"{self.path}  {self.human_size}  {self.sha256[:12]}  {where}"
        if self.note:
            line += f"\n    {self.note}"
        return line


@dataclass
class Manifest:
    """`artifacts/<run-id>/manifest.json` — every artifact, sorted for stable diffs."""

    run_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    artifacts: list[ArtifactRecord] = field(default_factory=list)

    def add(self, record: ArtifactRecord) -> ArtifactRecord:
        """Insert, replacing any earlier record for the same path.

        Re-running a cell overwrites the same plot file; keeping both records
        would make the manifest describe a file that no longer exists.
        """
        self.artifacts = [a for a in self.artifacts if a.path != record.path]
        self.artifacts.append(record)
        self.artifacts.sort(key=lambda a: a.path)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "artifacts": [asdict(a) for a in self.artifacts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        records = [ArtifactRecord(**item) for item in data.get("artifacts", [])]
        return cls(
            run_id=str(data.get("run_id", "")),
            created_at=str(data.get("created_at", "")),
            artifacts=records,
        )

    def render(self) -> str:
        if not self.artifacts:
            return f"{self.run_id}: no artifacts"
        lines = [f"{self.run_id}: {len(self.artifacts)} artifact(s)"]
        lines.extend(f"  {a.render()}" for a in self.artifacts)
        return "\n".join(lines)


class ArtifactStore:
    """Applies the size policy and maintains the manifest.

    The store owns no UI and no model: it is the one place that decides *where* a
    produced file lives. Tools hand it bytes, the loop asks it for the committed
    paths that go into the `Coagent-Artifacts` trailer.
    """

    def __init__(
        self,
        repo_root: Path,
        run_id: str,
        *,
        commit_under_mb: int = 5,
        drive_dir: str = "coagent/artifacts",
        drive_root: Path | None = None,
        step: int = 0,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.run_id = _safe_segment(run_id) or "run"
        self.commit_limit = max(0, int(commit_under_mb)) * 1024 * 1024
        self.drive_dir = drive_dir
        self.drive_root = Path(drive_root).resolve() if drive_root else _detect_drive()
        self.step = step
        self.manifest = Manifest(run_id=self.run_id)

    # ----------------------------------------------------------------- paths

    @property
    def dir(self) -> Path:
        return self.repo_root / ARTIFACT_ROOT / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST_NAME

    @property
    def run_dir_rel(self) -> str:
        return f"{ARTIFACT_ROOT}/{self.run_id}"

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _drive_target(self, sha: str, name: str) -> Path | None:
        if self.drive_root is None:
            return None
        return self.drive_root / self.drive_dir / sha[:2] / f"{sha}{Path(name).suffix}"

    # ------------------------------------------------------------------ save

    def save_bytes(
        self,
        data: bytes,
        name: str,
        *,
        mime: str = "",
        step: int | None = None,
        subdir: str = "",
    ) -> ArtifactRecord:
        """Store `data` under the §3.5 policy and record it in the manifest."""
        sha = hashlib.sha256(data).hexdigest()
        safe = _safe_name(name, mime)
        use_step = self.step if step is None else step
        size = len(data)

        if size > MAX_DRIVE_BYTES:
            # Never a copy we own at this size; the caller keeps the original.
            record = ArtifactRecord(
                path=f"{self.run_dir_rel}/{safe}",
                sha256=sha,
                bytes=size,
                store="external",
                created_by_step=use_step,
                mime=mime,
                name=safe,
                note=(
                    f"{size / 1024**3:.1f} GB is over the {MAX_DRIVE_BYTES // 1024**3} GB copy "
                    "limit; keep the original file or URL and record it here by hand."
                ),
            )
            return self.manifest.add(record)

        if size <= self.commit_limit:
            record = self._save_to_repo(data, safe, sha, use_step, mime, subdir)
        else:
            record = self._save_to_drive(data, safe, sha, use_step, mime, subdir)

        self._write_manifest()
        return record

    def save_file(self, source: Path, *, name: str = "", mime: str = "", subdir: str = "") -> ArtifactRecord:
        """Store an existing file by copying its bytes through the same policy."""
        source = Path(source)
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"cannot read {source}: {exc}") from exc
        return self.save_bytes(data, name or source.name, mime=mime, subdir=subdir)

    def save_text(self, text: str, name: str, *, mime: str = "text/plain", subdir: str = "") -> ArtifactRecord:
        return self.save_bytes(text.encode("utf-8"), name, mime=mime, subdir=subdir)

    def _save_to_repo(
        self,
        data: bytes,
        safe: str,
        sha: str,
        step: int,
        mime: str,
        subdir: str,
    ) -> ArtifactRecord:
        target = self.dir / (_safe_segment(subdir) + "/" if subdir else "") / safe
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            raise ArtifactError(f"cannot write artifact {self.rel(target)}: {exc}") from exc
        return self.manifest.add(
            ArtifactRecord(
                path=self.rel(target),
                sha256=sha,
                bytes=len(data),
                store="repo",
                created_by_step=step,
                location=self.rel(target),
                mime=mime,
                name=safe,
            )
        )

    def _save_to_drive(
        self,
        data: bytes,
        safe: str,
        sha: str,
        step: int,
        mime: str,
        subdir: str,
    ) -> ArtifactRecord:
        """Over the commit cap: Drive if mounted, otherwise an honest pointer.

        The in-repo `path` is where the manifest says it *would* live, so a report
        and the `Coagent-Artifacts` trailer name a stable identity either way.
        """
        path = f"{self.run_dir_rel}/{_safe_segment(subdir) + '/' if subdir else ''}{safe}"
        target = self._drive_target(sha, safe)
        if target is None:
            return self.manifest.add(
                ArtifactRecord(
                    path=path,
                    sha256=sha,
                    bytes=len(data),
                    store="drive",
                    created_by_step=step,
                    location="",
                    mime=mime,
                    name=safe,
                    note=(
                        f"{len(data) / 1024**2:.1f} MB is over the "
                        f"{self.commit_limit // 1024**2} MB commit cap and Drive is not mounted, "
                        "so it was not committed. Mount Drive and save it again, or raise "
                        "[artifacts] commit_under_mb deliberately."
                    ),
                )
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            raise ArtifactError(f"cannot write to Drive at {target}: {exc}") from exc
        return self.manifest.add(
            ArtifactRecord(
                path=path,
                sha256=sha,
                bytes=len(data),
                store="drive",
                created_by_step=step,
                location=str(target),
                mime=mime,
                name=safe,
            )
        )

    # -------------------------------------------------------------- manifest

    def _write_manifest(self) -> Path:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(
                json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(f"cannot write the artifact manifest: {exc}") from exc
        return self.manifest_path

    def flush(self) -> Path:
        """Write the manifest even when no artifact was added this step.

        The manifest is committed alongside the run, so it has to exist as soon as
        the run directory does — even if every artifact went to Drive.
        """
        return self._write_manifest()

    def committed_paths(self) -> list[str]:
        """Repo-relative paths for the `Coagent-Artifacts` trailer.

        Only committed bytes and the manifest are named: a Drive pointer is not a
        path in this commit, and claiming it is would make the trailer lie.
        """
        paths = [a.path for a in self.manifest.artifacts if a.in_repo]
        if paths:
            paths.append(f"{self.run_dir_rel}/{MANIFEST_NAME}")
        return sorted(set(paths))

    def load(self) -> Manifest:
        """Re-read the manifest from disk, for a resumed session (FR-GIT-5)."""
        if not self.manifest_path.is_file():
            return self.manifest
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read {self.rel(self.manifest_path)}: {exc}") from exc
        self.manifest = Manifest.from_dict(data)
        return self.manifest


# --------------------------------------------------------------------- helpers


def _detect_drive() -> Path | None:
    """The Colab Drive mount, if it is actually there.

    `drive.mount()` is the user's explicit action (FR-BOOT-4); this only *notices*
    it. On a laptop there is no `/content/drive`, so large artifacts degrade to a
    pointer rather than being forced into git.
    """
    candidate = Path("/content/drive/MyDrive")
    return candidate if candidate.is_dir() else None


def _safe_segment(text: str, *, limit: int = 40) -> str:
    """A single path segment with no traversal and no separators."""
    cleaned = _SAFE_NAME.sub("-", str(text)).strip("-.")
    return cleaned[:limit].strip("-.") or ""


def _safe_name(name: str, mime: str) -> str:
    """A filename that cannot escape the artifact directory.

    Cell outputs and model arguments are untrusted input, so `../../etc/passwd`
    must not survive this function. An extension is inferred from the MIME type
    when the name has none (a PNG from a cell is usually called `output`).
    """
    base = _safe_segment(Path(str(name)).name, limit=80)
    if not base:
        base = "artifact"
    if not Path(base).suffix:
        base += _IMAGE_EXT.get(mime) or _TEXT_EXT.get(mime) or ".bin"
    return base


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
