"""gitx — git as the agent's database (DESIGN §3)."""

from __future__ import annotations

from .commit import TRAILER_ORDER, CommitMeta, build_message, commit_step, parse_trailers
from .durable import SyncResult, checkpoint, sync_bundle, sync_push
from .git import GitError, GitRepo, GitResult
from .lock import LockHeldError, LockInfo, SessionLock
from .log import LogRecord, SessionLog
from .resume import Briefing, PlanItem, build_briefing, read_plan_items
from .session import (
    SESSION_BRANCH_PREFIX,
    DirtyTreeError,
    Session,
    SessionError,
    clone_repo,
    new_session_id,
    open_session,
    short_id,
)

__all__ = [
    "SESSION_BRANCH_PREFIX",
    "TRAILER_ORDER",
    "Briefing",
    "CommitMeta",
    "DirtyTreeError",
    "GitError",
    "GitRepo",
    "GitResult",
    "LockHeldError",
    "LockInfo",
    "LogRecord",
    "PlanItem",
    "Session",
    "SessionError",
    "SessionLock",
    "SessionLog",
    "SyncResult",
    "build_briefing",
    "build_message",
    "checkpoint",
    "clone_repo",
    "commit_step",
    "new_session_id",
    "open_session",
    "parse_trailers",
    "read_plan_items",
    "short_id",
    "sync_bundle",
    "sync_push",
]
