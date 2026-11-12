"""The commit protocol (PRD FR-GIT-3, Appendix B).

One commit per verified step, with six machine-readable trailers. That gives
`git log --grep="Coagent-Task: add CSV"`, `git diff <step-6> <step-7>` as replay,
a blame trail naming the model and step, and `git revert` as a per-action undo.

Trailers are written verbatim (no wrapping, no rewriting), and values are
sanitized to a single line so a newline inside a task description cannot break
the header block that `--grep` depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .git import GitRepo, build_commit_message

__all__ = ["CommitMeta", "TRAILER_ORDER", "build_message", "commit_step", "parse_trailers"]

TRAILER_ORDER: tuple[str, ...] = (
    "Coagent-Session",
    "Coagent-Step",
    "Coagent-Task",
    "Coagent-Verify",
    "Coagent-Model",
    "Coagent-Artifacts",
)

_MAX_TRAILER_LEN = 200


def _one_line(value: str, *, limit: int = _MAX_TRAILER_LEN) -> str:
    """Collapse a value onto one line and cap it — trailers are headers."""
    flat = " ".join(str(value).split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat


@dataclass(slots=True)
class CommitMeta:
    """Everything a step commit records."""

    session: str
    step: int
    task: str
    verify: str = "not run"
    model: str = "unknown"
    artifacts: list[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""

    def trailers(self) -> list[tuple[str, str]]:
        values = {
            "Coagent-Session": self.session,
            "Coagent-Step": str(self.step),
            "Coagent-Task": _one_line(self.task or "unspecified"),
            "Coagent-Verify": _one_line(self.verify or "not run"),
            "Coagent-Model": _one_line(self.model or "unknown"),
            "Coagent-Artifacts": _one_line(
                ", ".join(self.artifacts) if self.artifacts else "none"
            ),
        }
        return [(key, values[key]) for key in TRAILER_ORDER]

    def default_subject(self) -> str:
        return self.subject.strip() or _one_line(self.task or "coagent step", limit=72)


def build_message(meta: CommitMeta) -> str:
    return build_commit_message(meta.default_subject(), meta.body, meta.trailers())


def commit_step(repo: GitRepo, meta: CommitMeta) -> str:
    """Stage tracked changes and commit them with the full trailer set."""
    repo.add_all_tracked()
    if not repo.has_staged_changes():
        # Nothing changed: committing an empty step would corrupt the step
        # numbering that `git log --grep` relies on.
        raise ValueError("nothing to commit: no tracked changes are staged")
    return repo.commit_with_trailers(meta.default_subject(), meta.body, meta.trailers())


def parse_trailers(message: str) -> dict[str, str]:
    """Read the coagent trailers back out of a commit message."""
    found: dict[str, str] = {}
    for line in message.splitlines():
        key, sep, value = line.partition(": ")
        if sep and key.startswith("Coagent-"):
            found[key] = value.strip()
    return found
