"""gitx tests with real temp repos: dirty refusal, branch, trailers, push, lock."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from coagent.gitx import (
    CommitMeta,
    DirtyTreeError,
    GitError,
    GitRepo,
    LockHeldError,
    SessionError,
    SessionLock,
    build_briefing,
    checkpoint,
    commit_step,
    new_session_id,
    open_session,
    parse_trailers,
    sync_bundle,
)
from coagent.gitx.session import SESSION_BRANCH_PREFIX
from coagent.safety.policy import Policy

# --------------------------------------------------------------------- fixtures


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture()
def repo_path(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "dev@example.com")
    _git(root, "config", "user.name", "Dev")
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-q", "-m", "initial")
    return root


@pytest.fixture()
def repo(repo_path: Path) -> GitRepo:
    r = GitRepo(repo_path)
    r.ensure_identity()
    return r


# ------------------------------------------------------------------ identity


def test_ensure_identity_respects_existing_identity(repo_path: Path) -> None:
    """An identity the user already set must never be overwritten."""
    r = GitRepo(repo_path)
    r.ensure_identity()
    assert _git(repo_path, "config", "user.name").strip() == "Dev"
    assert _git(repo_path, "config", "user.email").strip() == "dev@example.com"


def test_ensure_identity_fills_missing_name(repo_path: Path, monkeypatch) -> None:
    """With no identity anywhere, coAgent sets its own."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(repo_path / "no-such-global-config"))
    _git(repo_path, "config", "--unset", "user.name")
    r = GitRepo(repo_path)
    r.ensure_identity()
    assert _git(repo_path, "config", "user.name").strip() == "coAgent"


def test_commits_are_attributed(repo: GitRepo, repo_path: Path) -> None:
    (repo_path / "app.py").write_text("x = 2\n", encoding="utf-8")
    repo.add_all_tracked()
    repo.commit("chore: bump")
    assert "coAgent" in _git(repo_path, "log", "-1", "--format=%an")


# -------------------------------------------------------------------- commit


def test_commit_trailers_are_queryable(repo: GitRepo, repo_path: Path) -> None:
    (repo_path / "app.py").write_text("x = 2\n", encoding="utf-8")
    sha = commit_step(
        repo,
        CommitMeta(
            session="2026-11-12-a1b2c3",
            step=7,
            task="add CSV loader with tests",
            verify="pytest -q (12 passed)",
            model="Qwen2.5-Coder-7B-Instruct@nf4",
            artifacts=["artifacts/bench/load.csv"],
            subject="coagent: add chunked CSV loader",
            body="Streams rows without loading the whole file.",
        ),
    )
    message = _git(repo_path, "log", "-1", "--format=%B")
    trailers = parse_trailers(message)
    assert trailers["Coagent-Session"] == "2026-11-12-a1b2c3"
    assert trailers["Coagent-Step"] == "7"
    assert trailers["Coagent-Task"] == "add CSV loader with tests"
    assert trailers["Coagent-Verify"] == "pytest -q (12 passed)"
    assert trailers["Coagent-Model"] == "Qwen2.5-Coder-7B-Instruct@nf4"
    assert trailers["Coagent-Artifacts"] == "artifacts/bench/load.csv"
    assert _git(repo_path, "log", "-1", "--format=%H").strip() == sha


def test_every_step_appears_in_grep_order(repo: GitRepo, repo_path: Path) -> None:
    for step in (1, 2, 3):
        (repo_path / "app.py").write_text(f"x = {step + 1}\n", encoding="utf-8")
        commit_step(repo, CommitMeta(session="s", step=step, task=f"step {step}"))
    # `git log --grep="Coagent-Step"` must return every step.
    out = _git(repo_path, "log", "--format=%s", "--grep=Coagent-Step")
    assert len([line for line in out.splitlines() if line.strip()]) == 3


def test_trailer_values_are_single_line(repo: GitRepo, repo_path: Path) -> None:
    (repo_path / "app.py").write_text("x = 5\n", encoding="utf-8")
    commit_step(
        repo,
        CommitMeta(session="s", step=1, task="line one\nline two\nline three", subject="multi"),
    )
    message = _git(repo_path, "log", "-1", "--format=%B")
    assert "Coagent-Task: line one line two line three" in message


def test_empty_commit_is_refused(repo: GitRepo) -> None:
    with pytest.raises(ValueError, match="nothing to commit"):
        commit_step(repo, CommitMeta(session="s", step=1, task="noop"))


def test_untracked_files_are_not_committed_by_with_trackers(repo: GitRepo, repo_path: Path) -> None:
    (repo_path / "new.py").write_text("y = 1\n", encoding="utf-8")
    (repo_path / "app.py").write_text("x = 9\n", encoding="utf-8")
    commit_step(repo, CommitMeta(session="s", step=1, task="tracked only"))
    tracked = _git(repo_path, "ls-files").split()
    assert "app.py" in tracked
    # Untracked stays untracked: the loop stages what it deliberately wrote.
    assert "new.py" not in tracked


# ------------------------------------------------------------------- session


def test_session_creates_branch(repo_path: Path) -> None:
    session = open_session(repo_path, session_id="2026-11-12-abc123")
    assert session.branch == f"{SESSION_BRANCH_PREFIX}2026-11-12-abc123"
    assert not session.resumed
    assert _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD").strip() == session.branch
    session.close()


def test_dirty_tree_is_refused(repo_path: Path) -> None:
    (repo_path / "app.py").write_text("x = 99\n", encoding="utf-8")
    with pytest.raises(DirtyTreeError) as exc:
        open_session(repo_path)
    msg = str(exc.value)
    assert "app.py" in msg
    assert "Commit, stash, or discard" in msg
    # The user's change must be intact.
    assert "x = 99" in (repo_path / "app.py").read_text()


def test_untracked_file_also_refused(repo_path: Path) -> None:
    (repo_path / "scratch.txt").write_text("hi\n", encoding="utf-8")
    with pytest.raises(DirtyTreeError, match="scratch.txt"):
        open_session(repo_path)


def test_session_reuses_existing_branch_as_resume(repo_path: Path) -> None:
    first = open_session(repo_path, session_id="2026-11-12-abc123")
    first.close()
    _git(repo_path, "switch", "-q", "main")
    second = open_session(repo_path, session_id="2026-11-12-abc123")
    assert second.resumed is True
    assert second.branch == first.branch
    second.close()


def test_session_never_works_on_main(repo_path: Path) -> None:
    """Asking for `main` must not put the session on `main`."""
    with pytest.raises(SessionError, match="refusing to work on 'main'"):
        open_session(repo_path, branch="main")
    assert _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    # Nothing was created or locked.
    assert not (repo_path / ".coagent" / "lock.json").exists()


def test_session_branches_off_a_feature_branch(repo_path: Path) -> None:
    """Starting from a non-coagent branch is fine: coAgent branches off it."""
    _git(repo_path, "switch", "-q", "-c", "feature/x")
    session = open_session(repo_path, session_id="2026-11-12-abc123")
    assert session.branch == "coagent/2026-11-12-abc123"
    session.close()


def test_session_does_not_carry_untracked_lock_into_dirty_check(repo_path: Path) -> None:
    """A lock file from a previous session must not block the next one."""
    first = open_session(repo_path, session_id="2026-11-12-abc123")
    first.close()
    _git(repo_path, "switch", "-q", "main")
    # lock.json is still on disk (release removed it, so recreate to be sure)
    (repo_path / ".coagent" / "lock.json").write_text("{}", encoding="utf-8")
    second = open_session(repo_path, session_id="2026-11-12-def456")
    assert second.session_id == "2026-11-12-def456"
    second.close()


def test_session_not_a_repo(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(SessionError, match="not a git repository"):
        open_session(plain)


def test_session_id_shape() -> None:
    sid = new_session_id()
    assert len(sid) == 17  # YYYY-MM-DD-shortid
    assert sid[4] == "-" and sid[7] == "-"


# ---------------------------------------------------------------------- lock


def test_lock_prevents_second_session(repo_path: Path, monkeypatch) -> None:
    """A lock from another process/host blocks a new session."""
    from coagent.gitx import lock as lock_mod

    session = open_session(repo_path, session_id="2026-11-12-abc123")
    _git(repo_path, "switch", "-q", "main")
    # Pretend the lock came from a different process on another VM.
    monkeypatch.setattr(lock_mod.socket, "gethostname", lambda: "other-vm")
    with pytest.raises(LockHeldError, match="holds the lock"):
        open_session(repo_path, session_id="2026-11-12-def456")
    monkeypatch.undo()
    session.close()


def test_lock_force_takes_over(repo_path: Path, monkeypatch) -> None:
    from coagent.gitx import lock as lock_mod

    first = open_session(repo_path, session_id="2026-11-12-abc123")
    _git(repo_path, "switch", "-q", "main")
    monkeypatch.setattr(lock_mod.socket, "gethostname", lambda: "other-vm")
    second = open_session(repo_path, session_id="2026-11-12-def456", force=True)
    assert second.session_id == "2026-11-12-def456"
    # The old session must not be able to release the new lock.
    assert first.lock.release() is False if first.lock else True
    monkeypatch.undo()
    second.close()


def test_expired_lock_is_stale(repo_path: Path) -> None:
    lock = SessionLock(repo_path, "old", ttl_s=0)
    lock.acquire()
    other = SessionLock(repo_path, "new", ttl_s=600)
    info = other.acquire()  # no exception: the old lock expired
    assert info.session_id == "new"


def test_heartbeat_extends_expiry(repo_path: Path) -> None:
    lock = SessionLock(repo_path, "s1", ttl_s=60)
    first = lock.acquire()
    second = lock.heartbeat()
    assert second.expires_at >= first.expires_at
    assert second.started_at == first.started_at


def test_release_only_own_lock(repo_path: Path) -> None:
    lock = SessionLock(repo_path, "s1")
    lock.acquire()
    intruder = SessionLock(repo_path, "s2")
    assert intruder.release() is False
    assert lock.release() is True


# ---------------------------------------------------------------- push guard


def test_push_denied_for_main(repo: GitRepo) -> None:
    with pytest.raises(GitError, match="only .*coagent"):
        repo.push("origin", "main")


def test_push_denied_for_unconfigured_remote(repo: GitRepo) -> None:
    policy = Policy(repo_root=repo.root, allowed_remotes=("origin",))
    with pytest.raises(GitError, match="not in"):
        repo.push("backdoor", "coagent/x", policy=policy)


def test_push_without_token_is_actionable(repo: GitRepo) -> None:
    with pytest.raises(GitError, match="no GitHub token"):
        repo.push("origin", "coagent/x")


def test_clone_url_with_credentials_is_refused(tmp_path: Path) -> None:
    from coagent.gitx.session import clone_repo

    with pytest.raises(GitError, match="embedded credentials"):
        clone_repo("https://user:ghp_secret@github.com/a/b.git", tmp_path / "x")


def test_clone_does_not_pollute_the_destination_before_cloning(repo_path: Path, tmp_path: Path) -> None:
    """Regression: the askpass helper must not be written into the target.

    It used to live at `<target>/.coagent/tmp/askpass.sh` and was installed before
    `git clone` ran, so the destination was never empty and every URL clone died
    with exit 128 "already exists and is not an empty directory".
    """
    from coagent.gitx.session import clone_repo

    dest = tmp_path / "workspace" / "src"

    clone_repo(str(repo_path), dest, depth=1)

    assert (dest / ".git").is_dir()
    # installed inside the repo only *after* the clone succeeded
    assert (dest / ".coagent" / "tmp" / "askpass.sh").is_file()
    # the staging helper outside the target is cleaned up
    assert not (dest.parent / ".coagent-askpass").exists()


def test_clone_into_a_fresh_nonexistent_path_works(repo_path: Path, tmp_path: Path) -> None:
    from coagent.gitx.session import clone_repo

    dest = tmp_path / "a" / "b" / "src"
    clone_repo(str(repo_path), dest, depth=1)
    assert (dest / ".git").is_dir()


def test_askpass_script_written_not_embedded(repo_path: Path) -> None:
    r = GitRepo(repo_path, token="ghp_1234567890ABCDEFGHIJ")
    path = r.install_askpass()
    assert path.is_file()
    assert "ghp_" not in path.read_text()  # the token is never written to disk


# ------------------------------------------------------------------ bundles


def test_bundle_written(repo: GitRepo, tmp_path: Path) -> None:
    (repo.root / "app.py").write_text("x = 42\n", encoding="utf-8")
    commit_step(repo, CommitMeta(session="s", step=1, task="bundled"))
    branch = repo.current_branch()
    result = sync_bundle(repo, branch, tmp_path / "bundles", session_id="s")
    assert result.ok
    assert result.path is not None and result.path.exists()
    assert result.path.stat().st_size > 0


def test_checkpoint_falls_back_to_bundle(repo: GitRepo, tmp_path: Path) -> None:
    (repo.root / "app.py").write_text("x = 43\n", encoding="utf-8")
    commit_step(repo, CommitMeta(session="s", step=1, task="fallback"))
    result = checkpoint(
        repo,
        branch=repo.current_branch(),
        session_id="s",
        mode="push",
        remote="origin",  # does not exist
        bundle_dir=tmp_path / "bundles",
    )
    assert result.ok
    assert result.mode == "bundle"
    assert result.warnings  # the push failure is reported, not swallowed


def test_checkpoint_reports_push_failure_without_bundle_dir(repo: GitRepo) -> None:
    (repo.root / "app.py").write_text("x = 44\n", encoding="utf-8")
    commit_step(repo, CommitMeta(session="s", step=1, task="nofallback"))
    result = checkpoint(repo, branch=repo.current_branch(), session_id="s", mode="push", remote="origin")
    assert not result.ok
    assert "failed" in result.message or "denied" in result.message


# ------------------------------------------------------------------- resume


def test_briefing_reads_trailers_and_plan(repo_path: Path) -> None:
    session = open_session(repo_path)
    plan_dir = repo_path / ".coagent" / "plans"
    plan_dir.mkdir(parents=True)
    (plan_dir / "task.md").write_text(
        "# add CSV loader\n- [x] read src/loader.py\n- [ ] write the test\n- [ ] pytest -q green\n",
        encoding="utf-8",
    )
    (repo_path / "app.py").write_text("x = 7\n", encoding="utf-8")
    commit_step(
        session.repo,
        CommitMeta(
            session=session.session_id,
            step=4,
            task="add CSV loader",
            verify="pytest -q (1 failed)",
        ),
    )
    briefing = build_briefing(session.repo, session.session_id, session.branch)
    assert briefing.committed_steps == 1
    assert briefing.trailers["Coagent-Step"] == "4"
    assert briefing.next_step == 5
    assert len(briefing.open_items) == 2
    text = briefing.render()
    assert "session" in text
    assert "write the test" in text
    assert "next step: 5" in text
    session.close()


def test_briefing_without_log_falls_back_to_trailers(repo_path: Path) -> None:
    session = open_session(repo_path)
    (repo_path / "app.py").write_text("x = 8\n", encoding="utf-8")
    commit_step(session.repo, CommitMeta(session=session.session_id, step=2, task="t"))
    briefing = build_briefing(session.repo, session.session_id, session.branch)
    assert briefing.recent == []
    assert briefing.trailers["Coagent-Step"] == "2"
    session.close()


def test_briefing_warns_on_dirty_tree(repo_path: Path) -> None:
    session = open_session(repo_path)
    (repo_path / "app.py").write_text("x = 11\n", encoding="utf-8")
    briefing = build_briefing(session.repo, session.session_id, session.branch)
    assert briefing.dirty_paths
    assert "WARNING" in briefing.render()
    session.close()


def test_next_step_is_monotonic_across_a_plan_commit(repo_path: Path) -> None:
    """A plan/checkpoint commit with a lower step number must not rewind the counter."""
    session = open_session(repo_path)
    (repo_path / "app.py").write_text("x = 20\n", encoding="utf-8")
    commit_step(session.repo, CommitMeta(session=session.session_id, step=5, task="code"))
    plan_dir = repo_path / ".coagent" / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "p.md").write_text("- [ ] next\n", encoding="utf-8")
    session.repo.add(".coagent/plans/p.md")
    session.repo.commit_with_trailers(
        "coagent: plan", "", [("Coagent-Session", session.session_id), ("Coagent-Step", "0")]
    )
    briefing = build_briefing(session.repo, session.session_id, session.branch)
    assert briefing.max_step == 5
    assert briefing.next_step == 6
    session.close()


def test_session_log_redacts_at_write_time(repo_path: Path) -> None:
    from coagent.gitx import LogRecord, SessionLog

    log = SessionLog(repo_path, "s1")
    log.append(
        LogRecord(
            step=1,
            phase="act",
            tool="run_shell",
            ok=False,
            error="auth failed with ghp_1234567890ABCDEFGHIJ",
        )
    )
    from coagent.safety.redact import MASK

    text = (repo_path / ".coagent" / "sessions" / "s1.jsonl").read_text(encoding="utf-8")
    assert MASK in text
    assert "ghp_1234567890ABCDEFGHIJ" not in text
