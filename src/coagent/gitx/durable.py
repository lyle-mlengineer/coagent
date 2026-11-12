"""Durable sync: get history off the VM after every checkpoint (FR-GIT-4).

**A commit is not a checkpoint; the sync is.** `/content` is exactly the disk
that disappears, so `checkpoint()` must run immediately after every commit:

* `mode = "push"` — push `coagent/<id>` to a configured remote. Denied to any
  other ref by the policy allowlist.
* `mode = "bundle"` — write a `git bundle` to Drive (or any durable dir), with
  no network at all. The first bundle carries full history; later ones are
  `A..B` increments, so cost tracks the diff, not the repo.

A failed sync must **warn and continue**: losing a checkpoint attempt is not
worth stopping a task that is otherwise making progress (PRD FR-GIT-4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .git import GitError, GitRepo

__all__ = ["SyncResult", "sync_push", "sync_bundle", "checkpoint"]


@dataclass(slots=True)
class SyncResult:
    ok: bool
    mode: str
    message: str
    ref: str = ""
    path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def sync_push(
    repo: GitRepo,
    remote: str,
    branch: str,
    *,
    policy=None,
    attempts: int = 2,
) -> SyncResult:
    """Push the session branch. Retries once, then reports rather than raises."""
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            result = repo.push(remote, branch, policy=policy)
        except GitError as exc:
            # A denial or a missing token is not a transient failure: report it
            # once and let the caller fall back to a bundle.
            return SyncResult(ok=False, mode="push", message=str(exc), ref=branch)
        if result.ok:
            return SyncResult(ok=True, mode="push", message=f"pushed {branch} to {remote}", ref=branch)
        last = (result.stderr or result.stdout).strip()[:500]
        if attempt < attempts:
            time.sleep(1.5)
    return SyncResult(
        ok=False,
        mode="push",
        message=f"push to {remote} failed: {last or 'unknown error'}",
        ref=branch,
    )


def sync_bundle(
    repo: GitRepo,
    branch: str,
    bundle_dir: str | Path,
    *,
    session_id: str,
    previous_tip: str | None = None,
) -> SyncResult:
    """Write a bundle (full first, then incremental) and return its path."""
    out_dir = Path(bundle_dir) / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = out_dir / f"{stamp}.bundle"

    head = repo.head()
    if previous_tip and repo.run(
        "rev-parse", "--verify", "--quiet", f"{previous_tip}^{{commit}}", check=False
    ).ok:
        # Incremental: only the new commits. `^` makes it A..B (exclusive A).
        spec = f"{previous_tip}..{branch}"
        argv = ["bundle", "create", str(target), spec]
    else:
        argv = ["bundle", "create", str(target), "--all"]

    result = repo.run(*argv, check=False)
    if not result.ok:
        return SyncResult(
            ok=False,
            mode="bundle",
            message=f"git bundle failed: {result.stderr.strip()[:400]}",
            ref=branch,
        )
    return SyncResult(
        ok=True,
        mode="bundle",
        message=f"wrote bundle {target.name} ({target.stat().st_size / 1024:.0f} KB) at {head[:8]}",
        ref=branch,
        path=target,
    )


def checkpoint(
    repo: GitRepo,
    *,
    branch: str,
    session_id: str,
    mode: str = "push",
    remote: str = "origin",
    bundle_dir: str | Path | None = None,
    previous_tip: str | None = None,
    policy=None,
) -> SyncResult:
    """Sync after a commit. Falls back to a bundle if the push fails."""
    warnings: list[str] = []
    if mode == "bundle":
        if bundle_dir is None:
            return SyncResult(
                ok=False,
                mode="bundle",
                message="durability.mode is 'bundle' but no bundle_dir is configured; "
                "mount Drive or set durability.bundle_dir.",
                ref=branch,
            )
        return sync_bundle(
            repo, branch, bundle_dir, session_id=session_id, previous_tip=previous_tip
        )

    pushed = sync_push(repo, remote, branch, policy=policy)
    if pushed.ok:
        return pushed
    warnings.append(pushed.message)
    if bundle_dir is None:
        pushed.warnings = warnings
        return pushed

    # Fallback: checkpoint offline so the work survives eviction.
    bundled = sync_bundle(repo, branch, bundle_dir, session_id=session_id, previous_tip=previous_tip)
    bundled.warnings = warnings
    if bundled.ok:
        bundled.message = f"{pushed.message}; fell back to {bundled.message}"
    return bundled
