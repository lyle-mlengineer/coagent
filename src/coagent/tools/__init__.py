"""Tool packages, assembled per phase (DESIGN §7.1).

`build_registry()` returns the tools available for a phase. Phases exist because
a small model does one job well: SURVEY can only read, PLAN cannot write at all,
ACT gets one write or exec, VERIFY may only run checks.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..context.artifacts import ArtifactStore
from ..safety.denylist import DEFAULT_DENIED
from ..safety.policy import Policy, SafetyClass
from .artifacts import DisplayArtifactTool, SaveArtifactTool
from .base import ArgSpec, BaseTool, ToolError, ToolRegistry, ToolResult, ToolSchemaError
from .fs import (
    EditFileTool,
    FileSession,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    apply_search_replace,
    truncate_head_tail,
)
from .notebooks import EditNotebookTool, ExecNotebookTool, ReadNotebookTool
from .shell import ExecConfig, RunPythonTool, RunShellTool, RunTestsTool

__all__ = [
    "ArgSpec",
    "ArtifactStore",
    "BaseTool",
    "ExecConfig",
    "FileSession",
    "Phase",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSchemaError",
    "apply_search_replace",
    "build_policy",
    "build_registry",
    "truncate_head_tail",
]

# What each phase is allowed to do. PLAN is absent on purpose: it cannot act.
PHASE_SAFETY: dict[str, frozenset[SafetyClass]] = {
    "survey": frozenset({SafetyClass.READ}),
    "plan": frozenset(),
    "act": frozenset(
        {SafetyClass.READ, SafetyClass.WRITE, SafetyClass.EXEC, SafetyClass.VCS, SafetyClass.UI}
    ),
    # VERIFY may render a plot it just produced: that is how "the cell ran"
    # becomes something the user can see (FR-UX-4), and it writes nothing new.
    "verify": frozenset({SafetyClass.READ, SafetyClass.EXEC, SafetyClass.UI}),
    "report": frozenset({SafetyClass.READ, SafetyClass.UI}),
}


class Phase:
    SURVEY = "survey"
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    REPORT = "report"

    ALL = (SURVEY, PLAN, ACT, VERIFY, REPORT)


def build_policy(
    repo_root: Path,
    cfg: Config,
    *,
    interactive: bool = True,
    scratch: Path | None = None,
) -> Policy:
    return Policy(
        repo_root=Path(repo_root),
        interactive=interactive,
        allow_read=[Path(p) for p in cfg.tools.allow_read],
        push_ref_prefix=cfg.git.push_ref_prefix,
        allowed_remotes=tuple(cfg.git.remotes),
        require_approval=tuple(cfg.safety.require_approval),
        deny_rules=tuple(cfg.safety.deny) if cfg.safety.deny else DEFAULT_DENIED,
        scratch=scratch if scratch is not None else Path(repo_root) / ".coagent" / "tmp",
    )


def build_exec_config(repo_root: Path, cfg: Config) -> ExecConfig:
    return ExecConfig(
        repo_root=Path(repo_root),
        max_output_bytes=cfg.tools.max_output_bytes,
        timeout_s=cfg.tools.shell_timeout_s,
        notebook_timeout_s=cfg.tools.notebook_timeout_s,
        hide_gpu=True,  # PRD D4: verification is CPU-only by default
        engine_dir=Path(repo_root) / ".coagent" / "tmp",
    )


def build_registry(
    repo_root: Path,
    cfg: Config,
    *,
    phase: str = Phase.ACT,
    policy: Policy | None = None,
    session: FileSession | None = None,
) -> ToolRegistry:
    """Every tool this phase may use, already filtered by safety class.

    ``session`` is shared across phases when the caller has one, because
    read-before-write and the artifact store are per-*session* state, not
    per-phase: SURVEY reading a file must satisfy ACT's edit guard.
    """
    root = Path(repo_root)
    policy = policy or (session.policy if session is not None else build_policy(root, cfg))
    if session is None:
        session = FileSession(repo_root=root, policy=policy)
    exec_cfg = build_exec_config(root, cfg)
    session.exec_cfg = exec_cfg
    if session.artifact_store is None:
        session.artifact_store = ArtifactStore(
            root,
            getattr(session, "run_id", "") or "run",
            commit_under_mb=cfg.artifacts.commit_under_mb,
            drive_dir=cfg.artifacts.drive_dir,
        )

    all_tools: list[BaseTool] = [
        ListDirTool(session),
        ReadFileTool(session),
        GrepTool(session),
        EditFileTool(session),
        WriteFileTool(session),
        # Notebooks are first-class (PRD D-2): a plot-and-cell task is a task,
        # not a source-file task with a JSON side quest.
        ReadNotebookTool(session),
        EditNotebookTool(session),
        ExecNotebookTool(session),
        SaveArtifactTool(session),
        DisplayArtifactTool(session),
        RunShellTool(exec_cfg, policy),
        RunTestsTool(
            exec_cfg,
            policy,
            verify_commands=list(cfg.verify.commands),
            feedback_max_tokens=cfg.verify.feedback_max_tokens,
        ),
        RunPythonTool(exec_cfg, policy),
    ]
    allowed = PHASE_SAFETY.get(phase)
    if allowed is None:
        raise ValueError(f"unknown phase {phase!r}; expected one of {list(PHASE_SAFETY)}")
    return ToolRegistry([t for t in all_tools if t.safety in allowed])
