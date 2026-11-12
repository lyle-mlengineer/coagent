"""Single-writer session lock (PRD FR-GIT-6).

`.coagent/lock.json` holds `{session_id, pid, started_at, heartbeat_at,
expires_at, host}`. A live lock means a second session refuses; an expired one
is stale and can be taken over. `--force` takes over even a live lock after
reporting which session held it.

This is advisory and deliberately simple: coAgent is a single-user tool, and a
stateful lock protocol on an ephemeral VM is more failure modes than it buys.
The real duplicate-work guard is that the session branch lives in git.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["LockInfo", "SessionLock", "LockHeldError"]

LOCK_RELPATH = Path(".coagent") / "lock.json"
DEFAULT_TTL_S = 30 * 60


class LockHeldError(RuntimeError):
    """Another session holds the lock."""

    def __init__(self, other: LockInfo) -> None:
        self.other = other
        minutes = max(0, int((other.expires_at - time.time()) / 60))
        super().__init__(
            f"session {other.session_id} holds the lock on this repo "
            f"(host {other.host}, expires in {minutes} min). "
            "Use --force to take over, or `coagent resume` to continue that session."
        )


@dataclass(slots=True)
class LockInfo:
    session_id: str
    pid: int
    host: str
    started_at: float
    heartbeat_at: float
    expires_at: float

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at

    def same_process(self) -> bool:
        """True when this lock is held by the current interpreter."""
        return self.host == socket.gethostname() and self.pid == os.getpid()


class SessionLock:
    def __init__(self, repo_root: str | Path, session_id: str, *, ttl_s: int = DEFAULT_TTL_S) -> None:
        self.root = Path(repo_root).resolve()
        self.path = self.root / LOCK_RELPATH
        self.session_id = session_id
        self.ttl_s = ttl_s
        self.info: LockInfo | None = None

    # ------------------------------------------------------------------ read

    def read(self) -> LockInfo | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        required = set(LockInfo.__dataclass_fields__)
        if not required <= set(data):
            return None
        return LockInfo(**{k: data[k] for k in required})

    # ----------------------------------------------------------------- write

    def acquire(self, *, force: bool = False) -> LockInfo:
        existing = self.read()
        if existing is not None and not existing.is_expired() and not existing.same_process() and not force:
            # A live lock from another process/host blocks us unless forced.
            raise LockHeldError(existing)
        now = time.time()
        info = LockInfo(
            session_id=self.session_id,
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=now,
            heartbeat_at=now,
            expires_at=now + self.ttl_s,
        )
        self._write(info)
        self.info = info
        return info

    def heartbeat(self) -> LockInfo:
        """Refresh the expiry. Called from the checkpoint loop (FR-REL-1)."""
        current = self.read()
        if current is not None and current.session_id != self.session_id:
            raise LockHeldError(current)
        base = current or (self.info or self._fresh())
        now = time.time()
        info = LockInfo(
            session_id=self.session_id,
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=base.started_at,
            heartbeat_at=now,
            expires_at=now + self.ttl_s,
        )
        self._write(info)
        self.info = info
        return info

    def release(self) -> bool:
        """Remove the lock, but never someone else's."""
        current = self.read()
        if current is None:
            return False
        if current.session_id != self.session_id and not current.is_expired():
            return False
        try:
            self.path.unlink()
        except OSError:
            return False
        self.info = None
        return True

    def __enter__(self) -> SessionLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    # ---------------------------------------------------------------- helpers

    def _write(self, info: LockInfo) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on POSIX and Windows

    def _fresh(self) -> LockInfo:
        now = time.time()
        return LockInfo(
            session_id=self.session_id,
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=now,
            heartbeat_at=now,
            expires_at=now + self.ttl_s,
        )
