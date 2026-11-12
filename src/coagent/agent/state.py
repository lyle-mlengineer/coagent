"""Session state: the plan file and the observation log (DESIGN §3.3, §5.4).

The plan file is *the* piece of memory that survives both context eviction and
VM death. So:

* it is written before the first code change (FR-AGT-1),
* it is committed with the steps that follow it,
* it is re-read after any eviction and on resume (FR-CTX-5).

The observation log is in-memory only and bounded; it exists so the ACT prompt
can carry the last few tool results without re-reading anything from disk.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .prompts import PlanStep

__all__ = [
    "Observation",
    "PlanFile",
    "SessionState",
    "slugify",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, limit: int = 40) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-")) or "task"


@dataclass(slots=True)
class Observation:
    """One tool result, as the model sees it."""

    tool: str
    ok: bool
    content: str
    step: int = 0
    at: float = field(default_factory=time.time)


class PlanFile:
    """`.coagent/plans/<task-slug>.md` — the durable plan."""

    def __init__(self, repo_root: str | Path, task_slug: str) -> None:
        self.root = Path(repo_root).resolve()
        self.dir = self.root / ".coagent" / "plans"
        self.path = self.dir / f"{task_slug}.md"
        self.title: str = task_slug
        self.steps: list[PlanStep] = []

    # ------------------------------------------------------------------- io

    def write(self, title: str, steps: list[PlanStep]) -> Path:
        self.title = title.strip() or self.path.stem
        self.steps = list(steps)
        self.dir.mkdir(parents=True, exist_ok=True)
        body = [f"# {self.title}", ""]
        body.extend(step.checkbox() for step in self.steps)
        body.append("")
        self.path.write_text("\n".join(body), encoding="utf-8")
        return self.path

    def read(self) -> bool:
        if not self.path.is_file():
            return False
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        self.steps = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
                self.steps.append(PlanStep(stripped[5:].strip(), done=True))
            elif stripped.startswith("- [ ]"):
                self.steps.append(PlanStep(stripped[5:].strip(), done=False))
        # The first H1 is the plan title; fall back to the file stem.
        if lines and lines[0].startswith("# "):
            self.title = lines[0][2:].strip() or self.path.stem
        return True

    # --------------------------------------------------------------- queries

    @property
    def open_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if not s.done]

    @property
    def complete(self) -> bool:
        return bool(self.steps) and not self.open_steps

    def mark_done(self, index: int) -> None:
        if 0 <= index < len(self.steps):
            self.steps[index].done = True

    def tick_matching(self, needle: str) -> int:
        """Tick the first open step whose text contains `needle`."""
        lowered = needle.lower()
        for step in self.steps:
            if not step.done and lowered in step.text.lower():
                step.done = True
                return 1
        return 0

    def render(self) -> str:
        return "\n".join([f"# {self.title}", *(s.checkbox() for s in self.steps)])

    @property
    def progress(self) -> str:
        done = sum(1 for s in self.steps if s.done)
        return f"{done}/{len(self.steps)}"


@dataclass
class SessionState:
    """Everything the loop needs to remember between turns."""

    task: str
    plan: PlanFile
    observations: list[Observation] = field(default_factory=list)
    files_read: set[str] = field(default_factory=set)
    files_written: set[str] = field(default_factory=set)
    next_step: int = 1
    last_verification: str = ""
    artifacts: list[str] = field(default_factory=list)
    max_observations: int = 6

    # ------------------------------------------------------------ observations

    def add_observation(self, tool: str, ok: bool, content: str, *, step: int = 0) -> Observation:
        obs = Observation(tool=tool, ok=ok, content=content, step=step)
        self.observations.append(obs)
        # Keep the window bounded: old results are the first thing to evict.
        if len(self.observations) > self.max_observations:
            del self.observations[: len(self.observations) - self.max_observations]
        return obs

    def observation_pairs(self) -> list[tuple[str, bool, str]]:
        return [(o.tool, o.ok, o.content) for o in self.observations]

    # ------------------------------------------------------------------ files

    def note_read(self, path: str) -> None:
        self.files_read.add(path)

    def note_written(self, path: str) -> None:
        self.files_written.add(path)
        self.files_read.add(path)

    def was_read(self, path: str) -> bool:
        return path in self.files_read

    # ---------------------------------------------------------------- progress

    def bump_step(self) -> int:
        current = self.next_step
        self.next_step += 1
        return current

    def evict(self, *, keep_plan: bool = True) -> None:
        """Drop observations before anything else (DESIGN §5.1 eviction order).

        The plan is re-read afterwards, never evicted — that is the whole point
        of keeping it in a file (FR-CTX-5).
        """
        self.observations.clear()
        if keep_plan:
            self.plan.read()

    def summary(self) -> str:
        return (
            f"task={self.task!r} plan={self.plan.progress} "
            f"reads={len(self.files_read)} writes={len(self.files_written)} "
            f"observations={len(self.observations)}"
        )
