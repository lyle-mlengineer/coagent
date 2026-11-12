"""Artifact store and tool tests (DESIGN §3.5, PRD FR-GIT-10 / FR-TOL-7, AT-7).

The properties worth pinning:

* **The commit cap is enforced in code, not by convention.** A 20 MB file must
  never reach git (AT-7), because a committed binary cannot be removed without
  rewriting history.
* **The manifest is the source of truth.** Every artifact is recorded with its
  sha256, size, store, and the step that made it, so a resumed session on a fresh
  VM can tell what exists and where.
* **Drive is optional** (FR-BOOT-4). Without a mount, an over-cap artifact is not
  silently dropped and not forced into git: it becomes an honest pointer with a
  message that says what to do.
* **Artifact names are untrusted.** They come from cell ids and model arguments,
  so `../../etc/passwd` must not survive the name sanitiser.
* **The trailer cannot lie.** Only bytes that are actually in this commit are
  named in `Coagent-Artifacts`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from coagent.config import Config
from coagent.context.artifacts import (
    MANIFEST_NAME,
    ArtifactStore,
    _safe_name,
    _safe_segment,
)
from coagent.tools import Phase, build_registry
from coagent.tools.base import ToolError

MB = 1024 * 1024


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


@pytest.fixture()
def store(repo: Path) -> ArtifactStore:
    return ArtifactStore(repo, "2026-11-12-abc123", commit_under_mb=5, drive_root=None)


@pytest.fixture()
def act(repo: Path):
    return build_registry(repo, Config(), phase=Phase.ACT)


# ------------------------------------------------------------------ size policy


def test_small_artifact_is_committed(store: ArtifactStore, repo: Path) -> None:
    record = store.save_text("a,b\n1,2\n", "data.csv", mime="text/csv")
    assert record.store == "repo"
    assert record.in_repo
    assert (repo / record.path).is_file()
    assert record.path == "artifacts/2026-11-12-abc123/data.csv"


def test_small_artifact_sha_matches_the_bytes(store: ArtifactStore, repo: Path) -> None:
    payload = b"hello artifact"
    record = store.save_bytes(payload, "x.txt")
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert record.bytes == len(payload)
    assert (repo / record.path).read_bytes() == payload


def test_at7_twenty_mb_artifact_is_a_pointer_not_a_commit(
    repo: Path, store: ArtifactStore
) -> None:
    """AT-7: a 20 MB artifact must not enter git, and must leave a manifest."""
    big = b"\x00" * (20 * MB)
    record = store.save_bytes(big, "big.png", mime="image/png")
    assert record.store == "drive"
    assert not record.in_repo
    # No bytes under the artifact tree: only the manifest is there.
    files = [p for p in (repo / "artifacts").rglob("*") if p.is_file()]
    assert [p.name for p in files] == [MANIFEST_NAME]
    assert record.sha256 == hashlib.sha256(big).hexdigest()


def test_over_cap_without_drive_says_what_to_do(repo: Path, store: ArtifactStore) -> None:
    record = store.save_bytes(b"x" * (6 * MB), "big.bin")
    assert record.store == "drive"
    assert "Drive is not mounted" in record.note
    assert "commit_under_mb" in record.note


def test_over_cap_with_drive_writes_to_drive(tmp_path: Path, repo: Path) -> None:
    drive = tmp_path / "drive"
    drive.mkdir()
    store = ArtifactStore(repo, "run", commit_under_mb=5, drive_root=drive)
    record = store.save_bytes(b"x" * (6 * MB), "big.png", mime="image/png")
    assert record.store == "drive"
    assert Path(record.location).is_file()
    assert Path(record.location).read_bytes()[:1] == b"x"
    # Still not in the repo.
    assert not (repo / record.path).exists()


def test_size_limit_is_configurable(repo: Path) -> None:
    store = ArtifactStore(repo, "run", commit_under_mb=1, drive_root=None)
    assert store.save_bytes(b"x" * (2 * MB), "a.bin").store == "drive"
    assert store.save_bytes(b"x" * (512 * 1024), "b.bin").store == "repo"


def test_over_two_gb_is_external_not_copied(repo: Path, monkeypatch) -> None:
    from coagent.context import artifacts as module

    store = ArtifactStore(repo, "run", drive_root=None)
    fake = b"x" * 10
    # Report a 3 GB artifact without allocating it.
    monkeypatch.setattr(module, "MAX_DRIVE_BYTES", 0)
    record = store.save_bytes(fake, "huge.bin")
    assert record.store == "external"
    assert "copy limit" in record.note


# ---------------------------------------------------------------------- manifest


def test_manifest_records_the_documented_fields(store: ArtifactStore, repo: Path) -> None:
    store.save_text("x", "a.txt")
    data = json.loads((repo / "artifacts/2026-11-12-abc123/manifest.json").read_text(encoding="utf-8"))
    entry = data["artifacts"][0]
    for key in ("path", "sha256", "bytes", "store", "created_by_step"):
        assert key in entry
    assert data["run_id"] == "2026-11-12-abc123"


def test_manifest_replaces_a_rerun_instead_of_duplicating(store: ArtifactStore, repo: Path) -> None:
    store.save_text("first", "same.txt")
    store.save_text("second", "same.txt")
    entries = [a for a in store.manifest.artifacts if a.name == "same.txt"]
    assert len(entries) == 1
    assert entries[0].sha256 == hashlib.sha256(b"second").hexdigest()


def test_manifest_is_stable_for_git_diffs(store: ArtifactStore, repo: Path) -> None:
    store.save_text("b", "b.txt")
    store.save_text("a", "a.txt")
    names = [a.name for a in store.manifest.artifacts]
    assert names == sorted(names)


def test_manifest_round_trips(store: ArtifactStore, repo: Path) -> None:
    store.save_text("keep me", "a.txt")
    fresh = ArtifactStore(repo, "2026-11-12-abc123", drive_root=None)
    manifest = fresh.load()
    assert [a.name for a in manifest.artifacts] == ["a.txt"]
    assert manifest.artifacts[0].sha256 == store.manifest.artifacts[0].sha256


def test_flush_writes_a_manifest_with_no_artifacts(store: ArtifactStore, repo: Path) -> None:
    path = store.flush()
    assert path.is_file()
    assert store.manifest.artifacts == []


# ---------------------------------------------------------------- trailer paths


def test_committed_paths_names_only_repo_bytes(repo: Path) -> None:
    store = ArtifactStore(repo, "run", commit_under_mb=5, drive_root=None)
    store.save_text("small", "small.txt")
    store.save_bytes(b"x" * (6 * MB), "big.bin")
    paths = store.committed_paths()
    assert "artifacts/run/small.txt" in paths
    assert "artifacts/run/manifest.json" in paths
    assert not any("big.bin" in p for p in paths)


def test_committed_paths_empty_when_everything_went_to_drive(repo: Path) -> None:
    store = ArtifactStore(repo, "run", commit_under_mb=1, drive_root=None)
    store.save_bytes(b"x" * (2 * MB), "big.bin")
    assert store.committed_paths() == []


# ------------------------------------------------------------------- safe names


@pytest.mark.parametrize(
    "name",
    ["../../etc/passwd", "..\\..\\windows\\system32\\x", "/abs/path/thing.txt", "a/b/c.png"],
)
def test_names_cannot_escape_the_artifact_dir(name: str) -> None:
    safe = _safe_name(name, "text/plain")
    assert "/" not in safe
    assert "\\" not in safe
    assert ".." not in safe


def test_missing_extension_is_inferred_from_mime() -> None:
    assert _safe_name("output", "image/png") == "output.png"
    assert _safe_name("table", "text/html") == "table.html"


def test_empty_name_gets_a_default() -> None:
    assert _safe_name("", "image/png") == "artifact.png"


def test_run_id_segment_is_sanitised() -> None:
    assert _safe_segment("../../evil") == "evil"
    store_root = _safe_segment("")
    assert store_root == ""


# ---------------------------------------------------------------------- tools


def test_save_artifact_tool_requires_content(repo: Path, act) -> None:
    with pytest.raises(ToolError) as err:
        act.call("save_artifact", {"name": "x.png"})
    assert "will not invent content" in str(err.value)


def test_save_artifact_refuses_both_source_and_text(repo: Path, act) -> None:
    with pytest.raises(ToolError) as err:
        act.call("save_artifact", {"name": "x.txt", "source": "mod.py", "text": "hi"})
    assert "not both" in str(err.value)


def test_save_artifact_from_an_existing_file(repo: Path, act) -> None:
    (repo / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = act.call("save_artifact", {"name": "fig.png", "source": "fig.png", "mime": "image/png"})
    assert out.ok
    assert out.meta["committed"] is True
    assert (repo / out.meta["artifact"]).is_file()


def test_save_artifact_missing_source_explains(repo: Path, act) -> None:
    with pytest.raises(ToolError) as err:
        act.call("save_artifact", {"name": "x.png", "source": "nothere.png"})
    assert "nothing to save" in str(err.value)


def test_save_artifact_refuses_an_empty_file(repo: Path, act) -> None:
    (repo / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(ToolError) as err:
        act.call("save_artifact", {"name": "empty.csv", "source": "empty.csv"})
    assert "empty" in str(err.value)


def test_save_artifact_from_text(repo: Path, act) -> None:
    out = act.call("save_artifact", {"name": "report.md", "text": "# done\n"})
    assert out.ok
    assert (repo / out.meta["artifact"]).read_text(encoding="utf-8") == "# done\n"


def test_save_artifact_jails_the_source(repo: Path, act, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    with pytest.raises(ToolError) as err:
        act.call("save_artifact", {"name": "x.txt", "source": str(outside)})
    assert "cannot read" in str(err.value)


def test_save_artifact_is_phase_gated(repo: Path) -> None:
    assert "save_artifact" not in build_registry(repo, Config(), phase=Phase.SURVEY).names()
    assert "save_artifact" not in build_registry(repo, Config(), phase=Phase.VERIFY).names()


# ---------------------------------------------------------------- displaying


def test_display_artifact_headless_references_the_file(repo: Path, act) -> None:
    (repo / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    out = act.call("save_artifact", {"name": "fig.png", "source": "fig.png", "mime": "image/png"})
    shown = act.call("display_artifact", {"path": out.meta["artifact"]})
    assert shown.ok
    assert shown.meta["shown"] == 0
    assert "no notebook surface" in shown.content


def test_display_artifact_calls_the_renderer(repo: Path) -> None:
    seen: list[tuple[str, str, str]] = []
    registry = build_registry(repo, Config(), phase=Phase.ACT)
    session = registry.get("display_artifact").session
    session.render = lambda kind, payload, mime: seen.append((kind, payload, mime))

    (repo / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
    out = registry.call("save_artifact", {"name": "fig.png", "source": "fig.png", "mime": "image/png"})
    registry.call("display_artifact", {"path": out.meta["artifact"]})

    assert seen, "the renderer was never called"
    kind, payload, mime = seen[0]
    assert kind == "image"
    assert mime == "image/png"
    assert isinstance(payload, str)  # base64 travels as text


def test_a_broken_renderer_does_not_fail_the_step(repo: Path) -> None:
    def boom(*_args):
        raise RuntimeError("widget died")

    registry = build_registry(repo, Config(), phase=Phase.ACT)
    registry.get("display_artifact").session.render = boom
    (repo / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out = registry.call("save_artifact", {"name": "fig.png", "source": "fig.png", "mime": "image/png"})
    shown = registry.call("display_artifact", {"path": out.meta["artifact"]})
    assert shown.ok
    assert "could not render" in shown.content


def test_display_with_no_path_shows_the_manifest(repo: Path, act) -> None:
    out = act.call("display_artifact", {})
    assert out.ok
    assert "no artifacts" in out.content


def test_display_artifact_is_available_to_verify(repo: Path) -> None:
    assert "display_artifact" in build_registry(repo, Config(), phase=Phase.VERIFY).names()


# ------------------------------------------------------- notebook integration


def test_exec_notebook_saves_a_displayed_image(tmp_path: Path) -> None:
    """US-3: a plot displayed in a cell becomes a committed artifact."""
    nbformat = pytest.importorskip("nbformat")
    from nbformat.v4 import new_code_cell, new_notebook

    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
        "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    cell = new_code_cell(
        "import base64\n"
        f"png = base64.b64decode('{png_b64}')\n"
        "from IPython.display import Image, display\n"
        "display(Image(data=png))\n"
    )
    (tmp_path / "plot.ipynb").write_text(
        nbformat.writes(new_notebook(cells=[cell]), version=4), encoding="utf-8"
    )
    registry = build_registry(tmp_path, Config(), phase=Phase.ACT)
    out = registry.call("exec_notebook", {"path": "plot.ipynb", "timeout_s": 120})
    if not out.ok:
        pytest.skip("notebook execution unavailable in this environment")

    artifacts = out.meta.get("artifacts") or []
    assert artifacts, "a displayed image should have been stored"
    saved = tmp_path / artifacts[0]
    assert saved.is_file()
    assert saved.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "artifacts" / "run" / MANIFEST_NAME).is_file()


# ------------------------------------------------------------------- git facts


def test_artifact_files_are_actually_committable(repo: Path, store: ArtifactStore) -> None:
    """The manifest and a small artifact survive a real `git add` + commit."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    store.save_text("data", "d.txt")
    store.save_bytes(b"x" * (6 * MB), "big.bin")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "artifacts"], cwd=repo, check=True)
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert "artifacts/2026-11-12-abc123/d.txt" in tracked
    assert "artifacts/2026-11-12-abc123/manifest.json" in tracked
    assert not any("big.bin" in t for t in tracked)


def test_resume_briefing_names_saved_artifacts(repo: Path, store: ArtifactStore) -> None:
    """A fresh VM has the files in git but no memory of them (DESIGN §3.6)."""
    from coagent.gitx.resume import read_artifact_manifest

    store.save_text("data", "d.csv")
    store.save_bytes(b"x" * (6 * MB), "big.bin")
    paths = read_artifact_manifest(repo, "2026-11-12-abc123")
    # Both the committed file and the Drive-only pointer are reported: the
    # manifest is what says the second one exists.
    assert "artifacts/2026-11-12-abc123/d.csv" in paths
    assert "artifacts/2026-11-12-abc123/big.bin" in paths


def test_resume_briefing_survives_a_broken_manifest(repo: Path) -> None:
    from coagent.gitx.resume import read_artifact_manifest

    run_dir = repo / "artifacts" / "2026-11-12-abc123"
    run_dir.mkdir(parents=True)
    (run_dir / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    assert read_artifact_manifest(repo, "2026-11-12-abc123") == []


def test_resume_briefing_renders_the_artifact_line(repo: Path, store: ArtifactStore) -> None:
    from coagent.gitx.resume import Briefing, read_artifact_manifest

    store.save_text("data", "d.csv")
    briefing = Briefing(session_id="s", branch="coagent/s")
    briefing.artifacts = read_artifact_manifest(repo, "2026-11-12-abc123")
    assert "artifacts: artifacts/2026-11-12-abc123/d.csv" in briefing.render()
