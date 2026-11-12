"""Tool packages, assembled per phase (DESIGN §7.1).

`build_registry()` returns the tools available for a phase. Phases exist because
a small model does one job well: SURVEY can only read, PLAN cannot write at all,
ACT gets one write or exec, VERIFY may only run checks.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..safety.denylist import DEFAULT_DENIED
from ..safety.policy import Policy, SafetyClass
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
from .shell import ExecConfig, RunPythonTool, RunShellTool

__all__ = [
    "ArgSpec",
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
    "act": frozenset({SafetyClass.READ, SafetyClass.WRITE, SafetyClass.EXEC, SafetyClass.VCS}),
    "verify": frozenset({SafetyClass.READ, SafetyClass.EXEC}),
    "report": frozenset({SafetyClass.READ}),
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
        hide_gpu=True,  # PRD D4: verification is CPU-only by default
        engine_dir=Path(repo_root) / ".coagent" / "tmp",
    )


def build_registry(
    repo_root: Path,
    cfg: Config,
    *,
    phase: str = Phase.ACT,
    policy: Policy | None = None,
) -> ToolRegistry:
    """Every tool this phase may use, already filtered by safety class."""
    root = Path(repo_root)
    policy = policy or build_policy(root, cfg)
    session = FileSession(repo_root=root, policy=policy)
    exec_cfg = build_exec_config(root, cfg)

    all_tools: list[BaseTool] = [
        ListDirTool(session),
        ReadFileTool(session),
        GrepTool(session),
        EditFileTool(session),
        WriteFileTool(session),
        RunShellTool(exec_cfg, policy),
        RunPythonTool(exec_cfg, policy),
    ]
    allowed = PHASE_SAFETY.get(phase)
    if allowed is None:
        raise ValueError(f"unknown phase {phase!r}; expected one of {list(PHASE_SAFETY)}")
    return ToolRegistry([t for t in all_tools if t.safety in allowed])
