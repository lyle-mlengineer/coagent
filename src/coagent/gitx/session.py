"""Session lifecycle: clone, guard, branch (PRD FR-GIT-1/2, FR-GIT-5).

A session is identified by its branch: `coagent/<yyyy-mm-dd>-<short-id>`. If the
branch already exists, opening it is a *resume*, not a new session — that is the
whole recovery story from DESIGN §3.6.

Two guards the tests care about:

* **Refuse a dirty tree.** A user with uncommitted work must get a stop with
  instructions, never a reset (FR-GIT-1, AT-3).
* **Never touch `main`.** The session branch is created from `HEAD` and the
  session refuses to run on a branch outside the coagent namespace.
"""

from __future__ import annotations

import secrets
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .git import GitError, GitRepo
from .lock import SessionLock

__all__ = [
    "Session",
    "SessionError",
    "DirtyTreeError",
    "new_session_id",
    "open_session",
    "short_id",
    "SESSION_BRANCH_PREFIX",
]

SESSION_BRANCH_PREFIX = "coagent/"
SHORT_ID_LEN = 6

# coAgent's own ephemeral state. Never counted as user dirt.
COAGENT_RUNTIME_PATHS: tuple[str, ...] = (
    ".coagent/lock.json",
    ".coagent/tmp/",
    ".coagent/sessions/",
    ".coagent/cache/",
    ".coagent/bundles/",
    ".coagent/secrets",
)


class SessionError(RuntimeError):
    """The session could not be opened. The message is the user's next step."""


class DirtyTreeError(SessionError):
    """The working tree has uncommitted changes; coAgent refuses to clobber them."""

    def __init__(self, paths: list[str]) -> None:
        listed = "\n  ".join(paths[:20])
        more = f"\n  … and {len(paths) - 20} more" if len(paths) > 20 else ""
        super().__init__(
            "the working tree has uncommitted changes, so coAgent will not start "
            "(it never resets your work). Commit, stash, or discard them first:\n"
            f"  {listed}{more}\n"
            "Then re-run. Nothing has been modified."
        )
        self.paths = paths


def short_id(n: int = SHORT_ID_LEN) -> str:
    return secrets.token_hex(max(3, n // 2))[:n]


def new_session_id(*, now: datetime | None = None, short: str | None = None) -> str:
    """`2026-11-12-a1b2c3` — date first so branches sort chronologically."""
    moment = now or datetime.now(UTC)
    return f"{moment:%Y-%m-%d}-{short or short_id()}"


@dataclass(slots=True)
class Session:
    """An open working session: repo, branch, lock, identity."""

    repo: GitRepo
    session_id: str
    branch: str
    resumed: bool = False
    lock: SessionLock | None = None

    @property
    def root(self) -> Path:
        return self.repo.root

    def checkpoint_heartbeat(self) -> None:
        if self.lock is not None:
            self.lock.heartbeat()

    def close(self, *, release_lock: bool = True) -> None:
        if release_lock and self.lock is not None:
            self.lock.release()
            self.lock = None


def _ensure_coagent_ignored(root: Path) -> None:
    """Warn (never silently edit) when `.coagent/` runtime state is untracked.

    coAgent writes `.coagent/lock.json`, `tmp/`, and `sessions/`; a user repo
    without these ignores would see a permanently dirty tree and refuse to
    start its own session. DESIGN §11 is explicit that bootstrap **warns**
    rather than mutating a tracked file, so we record the advice on the session
    and leave the file alone.
    """
    ignore = root / ".gitignore"
    needed = (".coagent/lock.json", ".coagent/tmp/", ".coagent/sessions/")
    try:
        existing = ignore.read_text(encoding="utf-8") if ignore.is_file() else ""
    except OSError:
        return
    missing = [p for p in needed if p not in existing]
    if missing:
        # Docker-style: report once, do not surprise the user by editing a file
        # that is already in their history.
        import sys

        print(
            "coagent: warning — .gitignore does not ignore coAgent runtime state:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n  Add them, or `coagent init` will keep reporting a dirty tree.",
            file=sys.stderr,
        )


def open_session(
    repo_path: str | Path,
    *,
    session_id: str | None = None,
    branch: str | None = None,
    force: bool = False,
    author_name: str = "coAgent",
    author_email: str = "coagent@users.noreply.github.com",
    token: str | None = None,
    ttl_s: int = 30 * 60,
    allow_branch: str | None = None,
) -> Session:
    """Open (or resume) a coAgent session on an existing clone.

    `allow_branch` exists for tests and for the `durable` clone path; in normal
    use the branch must live under `coagent/`.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise SessionError(f"{root} is not a directory.")
    repo = GitRepo(root, author_name=author_name, author_email=author_email, token=token)
    if not repo.is_repo():
        raise SessionError(
            f"{root} is not a git repository. Clone it first, or pass a repo URL to coAgent(repo=...)."
        )

    repo.ensure_identity()
    _ensure_coagent_ignored(root)

    # Guard: never start on top of the user's uncommitted work. coAgent's own
    # runtime state is excluded (DESIGN §11: warn, never mutate a tracked file).
    if repo.is_dirty(ignore=COAGENT_RUNTIME_PATHS):
        raise DirtyTreeError(repo.changed_paths(ignore=COAGENT_RUNTIME_PATHS))

    current = repo.current_branch()

    if branch is not None:
        target = branch
    elif current.startswith(SESSION_BRANCH_PREFIX):
        # Already on a session branch: this is a resume.
        target = current
    else:
        target = f"{SESSION_BRANCH_PREFIX}{session_id or new_session_id()}"

    resumed = repo.branch_exists(target)

    if not resumed:
        try:
            repo.switch(target, create=True)
        except GitError as exc:
            raise SessionError(
                f"could not create the session branch {target}: {exc}. "
                "If you started from a detached HEAD, check out a branch first."
            ) from exc
    else:
        if current != target:
            try:
                repo.switch(target)
            except GitError as exc:
                raise SessionError(f"could not switch to the existing session branch {target}: {exc}") from exc

    if allow_branch is None and not target.startswith(SESSION_BRANCH_PREFIX):
        raise SessionError(
            f"refusing to work on {target!r}: coAgent only works on {SESSION_BRANCH_PREFIX}* "
            "branches so that main is never touched."
        )

    sid = target.removeprefix(SESSION_BRANCH_PREFIX)
    lock = SessionLock(root, sid, ttl_s=ttl_s)
    lock.acquire(force=force)

    return Session(repo=repo, session_id=sid, branch=target, resumed=resumed, lock=lock)


def clone_repo(
    url: str,
    dest: str | Path,
    *,
    depth: int = 1,
    token: str | None = None,
    author_name: str = "coAgent",
    author_email: str = "coagent@users.noreply.github.com",
    overwrite: bool = False,
) -> GitRepo:
    """Clone `url` into `dest` (`--depth 1` by default, DESIGN §3.1)."""
    target = Path(dest).resolve()
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise SessionError(
                f"{target} already exists and is not empty. coAgent will not overwrite it; "
                "pass overwrite=True to replace it, or point at a different directory."
            )
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    repo = GitRepo(target, author_name=author_name, author_email=author_email, token=token)
    repo.install_askpass()
    repo.clone(url, target, depth=depth)
    return repo
