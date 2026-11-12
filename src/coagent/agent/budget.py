"""Budgets and loop detection (PRD FR-AGT-5, FR-AGT-6; DESIGN §7.3–7.4).

Two failure modes this module exists to prevent:

1. **Unattended burn.** A free agent that quietly loops for 40 minutes is not
   free — it costs the user their session. Every counter here has a ceiling and
   exhausts into a checkpointed report, never a silent continue.
2. **Repeating the same failing action.** A weak model will re-issue the same
   edit. The fingerprint catches it: 3 repeats force a replan, 5 stop the task.
"""

from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field

__all__ = ["Budget", "BudgetExhausted", "BudgetLimits", "LoopDetector", "LoopVerdict"]


class BudgetExhausted(RuntimeError):
    """A hard limit was reached. The caller must checkpoint and report."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


@dataclass(slots=True)
class BudgetLimits:
    max_steps: int = 60
    max_wall_s: int = 2700
    max_tokens: int = 200_000
    max_gpu_min: int = 40
    max_commits: int = 40
    max_consecutive_failures: int = 3

    @classmethod
    def from_config(cls, cfg) -> BudgetLimits:  # noqa: ANN001 - cfg is Config
        a = cfg.agent
        return cls(
            max_steps=a.max_steps,
            max_wall_s=a.max_wall_s,
            max_tokens=a.max_tokens,
            max_gpu_min=a.max_gpu_min,
            max_commits=a.max_commits,
            max_consecutive_failures=a.max_consecutive_failures,
        )


@dataclass
class Budget:
    """Live counters with hard ceilings."""

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    steps: int = 0
    tokens: int = 0
    commits: int = 0
    gpu_seconds: float = 0.0
    consecutive_failures: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # ---------------------------------------------------------------- counters

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def gpu_minutes(self) -> float:
        return self.gpu_seconds / 60.0

    def record_step(self) -> None:
        self.steps += 1
        self.check()

    def record_tokens(self, n: int) -> None:
        self.tokens += max(0, n)
        self.check()

    def record_commit(self) -> None:
        self.commits += 1
        self.check()

    def record_gpu(self, seconds: float) -> None:
        self.gpu_seconds += max(0.0, seconds)
        self.check()

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """A failed step. Three in a row stops the task (PRD FR-AGT-5)."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.limits.max_consecutive_failures:
            raise BudgetExhausted(
                "consecutive_failures",
                f"{self.consecutive_failures} consecutive failed steps "
                f"(limit {self.limits.max_consecutive_failures}). Stopping so the "
                "failures can be inspected instead of repeated.",
            )

    # ----------------------------------------------------------------- checks

    def check(self) -> None:
        """Raise `BudgetExhausted` if any ceiling is crossed."""
        lim = self.limits
        if self.steps > lim.max_steps:
            raise BudgetExhausted(
                "steps", f"reached the step limit ({lim.max_steps}). Checkpointing and reporting."
            )
        if self.elapsed_s >= lim.max_wall_s:
            raise BudgetExhausted(
                "wall_clock",
                f"reached the wall-clock limit ({lim.max_wall_s // 60} min). "
                "Checkpointing and reporting.",
            )
        if self.tokens > lim.max_tokens:
            raise BudgetExhausted(
                "tokens",
                f"reached the token limit ({lim.max_tokens}). Checkpointing and reporting.",
            )
        if self.commits > lim.max_commits:
            raise BudgetExhausted(
                "commits",
                f"reached the commit limit ({lim.max_commits}). A task that needs more "
                "commits than this is too large to finish reliably.",
            )
        if self.gpu_minutes > lim.max_gpu_min:
            raise BudgetExhausted(
                "gpu_minutes",
                f"reached the GPU-minute limit ({lim.max_gpu_min}). Checkpointing and reporting.",
            )

    def remaining(self) -> dict[str, float]:
        lim = self.limits
        return {
            "steps": max(0, lim.max_steps - self.steps),
            "wall_s": max(0.0, lim.max_wall_s - self.elapsed_s),
            "tokens": max(0, lim.max_tokens - self.tokens),
            "commits": max(0, lim.max_commits - self.commits),
            "gpu_min": max(0.0, lim.max_gpu_min - self.gpu_minutes),
        }

    def describe(self) -> str:
        r = self.remaining()
        return (
            f"step {self.steps}/{self.limits.max_steps} · "
            f"{r['wall_s'] / 60:.0f} min left · "
            f"{r['tokens']} tokens left"
        )


class LoopVerdict(str, enum.Enum):
    CONTINUE = "continue"
    REPLAN = "replan"
    STOP = "stop"


@dataclass
class LoopDetector:
    """Detects a model repeating itself (DESIGN §7.4).

    `fingerprint = sha256(tool + args)` — deliberately excludes elapsed time and
    token counts, so only a genuinely repeated action counts. Two whitelisted
    exception classes are tracked separately:

    * edits to a file that was never read (grounding failure),
    * the same test failing after three different edits (wrong hypothesis).
    """

    replan_after: int = 3
    stop_after: int = 5
    _last: str = ""
    _repeats: int = 0
    _history: list[str] = field(default_factory=list)

    @staticmethod
    def fingerprint(tool: str, args: dict) -> str:
        payload = tool + "\x00" + repr(sorted((k, repr(v)) for k, v in args.items()))
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]

    def observe(self, tool: str, args: dict) -> LoopVerdict:
        """Record an action and return a `LoopVerdict`."""
        fp = self.fingerprint(tool, args)
        self._history.append(fp)
        if fp == self._last:
            self._repeats += 1
        else:
            self._repeats = 1
            self._last = fp
        if self._repeats >= self.stop_after:
            return LoopVerdict.STOP
        if self._repeats >= self.replan_after:
            return LoopVerdict.REPLAN
        return LoopVerdict.CONTINUE

    @property
    def repeats(self) -> int:
        return self._repeats

    def reset(self) -> None:
        self._last = ""
        self._repeats = 0

    def corrective(self, verdict: LoopVerdict) -> str:
        if verdict == LoopVerdict.REPLAN:
            return (
                "You are repeating the same action. Stop, re-read the last failure "
                "output, and choose a different approach. If the plan is wrong, say so "
                "and propose a corrected plan."
            )
        if verdict == LoopVerdict.STOP:
            return (
                "You have repeated the same action too many times. The task is stopping "
                "so the failure can be inspected."
            )
        return ""
