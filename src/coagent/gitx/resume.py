"""Resume: rebuild a briefing from durable state (PRD FR-GIT-5, FR-REL-4).

Deliberately **template extraction, not an LLM call** (DESIGN §3.4). A resume
must be cheap and must work before the model is loaded, because the most common
resume happens on a fresh VM after a preemption.

Sources, in order of authority:

1. the session branch and its `Coagent-*` trailers (always present);
2. `.coagent/plans/*.md` (the memory that survives eviction);
3. `.coagent/sessions/<id>.jsonl` (may be missing after a hard kill).

If the log is missing, the briefing falls back to trailers — which is why the
commit protocol matters more than the log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .commit import parse_trailers
from .git import GitRepo

__all__ = ["Briefing", "build_briefing", "read_plan_items", "read_last_log_records"]


@dataclass(slots=True)
class PlanItem:
    text: str
    done: bool


@dataclass(slots=True)
class Briefing:
    session_id: str
    branch: str
    last_commit: str = ""
    last_subject: str = ""
    trailers: dict[str, str] = field(default_factory=dict)
    committed_steps: int = 0
    max_step: int = 0
    plan_path: Path | None = None
    plan_items: list[PlanItem] = field(default_factory=list)
    open_items: list[str] = field(default_factory=list)
    recent: list[dict] = field(default_factory=list)
    dirty_paths: list[str] = field(default_factory=list)

    @property
    def next_step(self) -> int:
        """One past the highest step ever committed, not just HEAD's.

        A plan or checkpoint commit can carry a lower `Coagent-Step` than the
        last code commit; taking the max keeps step numbers monotonic, which is
        what `git log --grep="Coagent-Step"` ordering depends on.
        """
        return self.max_step + 1

    def render(self, *, max_chars: int = 1500) -> str:
        """A 15-ish line briefing as plain text."""
        lines = [
            f"coAgent resume · session {self.session_id} · branch {self.branch}",
            f"last commit: {self.last_commit[:8]} {self.last_subject}".rstrip(),
            f"committed steps: {self.committed_steps} · next step: {self.next_step}",
        ]
        if self.trailers.get("Coagent-Task"):
            lines.append(f"task: {self.trailers['Coagent-Task']}")
        if self.trailers.get("Coagent-Verify"):
            lines.append(f"last verify: {self.trailers['Coagent-Verify']}")
        if self.dirty_paths:
            lines.append(f"WARNING: uncommitted changes on disk: {', '.join(self.dirty_paths[:5])}")
        if self.plan_path is not None:
            lines.append(f"plan: {self.plan_path}")
        if self.plan_items:
            done = sum(1 for item in self.plan_items if item.done)
            lines.append(f"plan progress: {done}/{len(self.plan_items)}")
        for item in self.open_items[:8]:
            lines.append(f"  [ ] {item}")
        if self.recent:
            failures = [r for r in self.recent if not r.get("ok", True)]
            if failures:
                last = failures[-1]
                lines.append(
                    f"open failure: {last.get('tool', '?')} — "
                    f"{str(last.get('error', ''))[:180]}"
                )
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        return text


def read_plan_items(plan_path: Path) -> list[PlanItem]:
    """Parse `- [x]` / `- [ ]` checkboxes from a plan file."""
    items: list[PlanItem] = []
    try:
        text = plan_path.read_text(encoding="utf-8")
    except OSError:
        return items
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- [x]", "- [X]")):
            items.append(PlanItem(text=stripped[5:].strip(), done=True))
        elif stripped.startswith("- [ ]"):
            items.append(PlanItem(text=stripped[5:].strip(), done=False))
    return items


def read_last_log_records(log_path: Path, *, count: int = 20) -> list[dict]:
    """The last `count` JSONL records, tolerant of a truncated final line."""
    if not log_path.is_file():
        return []
    records: list[dict] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a kill can leave a partial line; skip it
    except OSError:
        return []
    return records[-count:]


def build_briefing(repo: GitRepo, session_id: str, branch: str) -> Briefing:
    """Assemble the resume briefing from whatever survived."""
    root = repo.root
    briefing = Briefing(session_id=session_id, branch=branch)

    log_out = repo.run("log", "--format=%H%x00%s%x00%B%x01", "-n50", branch, check=False)
    commits = [c for c in log_out.stdout.split("\x01") if c.strip()]
    briefing.committed_steps = sum(1 for c in commits if "Coagent-Step:" in c)
    for commit in commits:
        trailers = parse_trailers(commit)
        try:
            briefing.max_step = max(briefing.max_step, int(trailers.get("Coagent-Step", "0")))
        except ValueError:
            continue
    if commits:
        head = commits[0].split("\x00")
        if len(head) >= 3:
            briefing.last_commit = head[0].strip()
            briefing.last_subject = head[1].strip()
            briefing.trailers = parse_trailers(head[2])

    # Plan: the newest plan file for this session, else the newest overall.
    plan_dir = root / ".coagent" / "plans"
    plan_path = None
    if plan_dir.is_dir():
        candidates = sorted(plan_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            plan_path = candidates[0]
    if plan_path is not None:
        briefing.plan_path = plan_path
        briefing.plan_items = read_plan_items(plan_path)
        briefing.open_items = [item.text for item in briefing.plan_items if not item.done]

    briefing.recent = read_last_log_records(root / ".coagent" / "sessions" / f"{session_id}.jsonl")

    if repo.is_dirty():
        briefing.dirty_paths = repo.changed_paths()

    return briefing
