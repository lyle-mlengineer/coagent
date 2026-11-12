"""Execution tools: shell and Python, isolated at the process level (FR-TOL-3).

There is no container on Colab, so isolation is process-level and the docs say
so. What this module guarantees:

* a fresh process group per call, killed on timeout (`killpg`, not `kill`);
* a scrubbed environment — GitHub/HF/cloud tokens never reach a child process;
* `CUDA_VISIBLE_DEVICES=""` by default, so verification cannot fight the model
  for VRAM (PRD D4);
* a hard output cap with head/tail truncation, so a 500-line traceback cannot
  evict the context the model needs to fix it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..safety.policy import Policy, Verdict
from .base import ArgSpec, BaseTool, SafetyClass, ToolError, ToolResult

__all__ = ["RunShellTool", "RunPythonTool", "ExecConfig", "scrubbed_env", "run_subprocess"]

# Never propagate these to a child process.
_SECRET_ENV_HINTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "APIKEY",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "GITHUB_PAT",
    "GH_TOKEN",
    "HF_TOKEN",
)


@dataclass
class ExecConfig:
    repo_root: Path
    max_output_bytes: int = 65536
    timeout_s: int = 120
    #: Notebooks run longer than a shell command (PRD FR-TOL-6: 300 s).
    notebook_timeout_s: int = 300
    hide_gpu: bool = True
    engine_dir: Path | None = None  # .coagent/tmp — the scratch jail

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        if self.engine_dir is not None:
            self.engine_dir = Path(self.engine_dir).resolve()


def scrubbed_env(cfg: ExecConfig, extra: dict[str, str] | None = None) -> dict[str, str]:
    """A minimal environment with every secret-shaped variable removed."""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(hint in upper for hint in _SECRET_ENV_HINTS):
            continue
        env[key] = value
    env.pop("GIT_ASKPASS", None)
    env.pop("SSH_AUTH_SOCK", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["COAGENT_CHILD"] = "1"
    if cfg.hide_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if extra:
        env.update(extra)
    return env


@dataclass(slots=True)
class ExecOutcome:
    returncode: int
    output: str
    truncated: bool
    timed_out: bool
    duration_s: float


def run_subprocess(
    argv: list[str],
    *,
    cfg: ExecConfig,
    cwd: Path | None = None,
    timeout_s: int | None = None,
    extra_env: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> ExecOutcome:
    """Run `argv` in its own process group with a scrubbed env and an output cap."""
    import time

    workdir = Path(cwd) if cwd else cfg.repo_root
    limit = timeout_s or cfg.timeout_s
    started = time.monotonic()
    # start_new_session puts the child in its own process group on POSIX so a
    # timeout can kill the whole tree. It is not available on Windows.
    popen_kwargs: dict[str, Any] = {
        "cwd": str(workdir),
        "env": scrubbed_env(cfg, extra_env),
        "stdin": subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "errors": "replace",
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:  # pragma: no cover - Colab is POSIX
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(argv, **popen_kwargs)
    timed_out = False
    try:
        raw, _ = proc.communicate(input=stdin_text, timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        try:
            raw, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL should win
            raw = ""
    duration = time.monotonic() - started
    output = raw or ""
    truncated = False
    if len(output.encode("utf-8", errors="replace")) > cfg.max_output_bytes:
        output, truncated = _cap_output(output, cfg.max_output_bytes)
    return ExecOutcome(
        returncode=proc.returncode if proc.returncode is not None else -1,
        output=output,
        truncated=truncated,
        timed_out=timed_out,
        duration_s=duration,
    )


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Kill the whole process group, falling back to the direct child.

    Colab is Linux, so `killpg` is the real path. `os.getpgid` does not exist on
    Windows; there we fall back to plain terminate/kill so the tool is still
    usable on a dev laptop (without claiming real group isolation).
    """
    pgid: int | None = None
    if hasattr(os, "getpgid") and hasattr(os, "killpg"):
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None

    for sig in (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.kill()
        except OSError:
            return
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            continue


def _cap_output(text: str, limit: int) -> tuple[str, bool]:
    """Head 60% / tail 40%, with an explicit omission marker (DESIGN §6.2)."""
    encoded = text.encode("utf-8", errors="replace")
    head_bytes = int(limit * 0.6)
    tail_bytes = limit - head_bytes
    head = encoded[:head_bytes].decode("utf-8", errors="replace")
    tail = encoded[-tail_bytes:].decode("utf-8", errors="replace")
    omitted = len(encoded) - head_bytes - tail_bytes
    return f"{head}\n… [{omitted} bytes omitted] …\n{tail}", True


class _ExecTool(BaseTool):
    def __init__(self, cfg: ExecConfig, policy: Policy) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.policy = policy

    def _gate(self, cmd: str) -> None:
        decision = self.policy.check_shell(cmd)
        if decision.verdict is Verdict.DENY:
            raise ToolError(f"denied: {decision.reason}")
        if decision.verdict is Verdict.ASK:
            # The loop must collect approval before dispatch.
            raise ToolError(f"needs approval: {decision.reason}")

    def _result(self, outcome: ExecOutcome, label: str) -> ToolResult:
        status = "timeout" if outcome.timed_out else f"exit {outcome.returncode}"
        header = f"$ {label}\n[{status}, {outcome.duration_s:.1f}s]"
        body = f"{header}\n{outcome.output}"
        if outcome.timed_out:
            body += "\n[command exceeded its timeout and its process group was killed]"
        return ToolResult(
            ok=outcome.returncode == 0 and not outcome.timed_out,
            content=body,
            truncated=outcome.truncated,
            meta={"returncode": outcome.returncode, "timed_out": outcome.timed_out},
        )


class RunShellTool(_ExecTool):
    name = "run_shell"
    description = (
        "Run a shell command in the repo. Prefer argv mode (`argv` list) over a "
        "shell string. 120 s timeout. Needs approval for installs, network, and deletes."
    )
    safety = SafetyClass.EXEC
    args = (
        ArgSpec("argv", "array", "Command as an argument list, e.g. ['pytest','-q'].", required=False, default=None),
        ArgSpec("command", "string", "Shell string, used only when argv is absent.", required=False, default=""),
        ArgSpec("timeout_s", "integer", "Timeout override in seconds.", required=False, default=0),
    )

    def run(self, argv: Any = None, command: str = "", timeout_s: int = 0) -> ToolResult:
        if argv:
            if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
                raise ToolError("argv must be a list of strings")
            label = " ".join(argv)
            self._gate(label)
            exec_argv = list(argv)
        elif command:
            label = command
            self._gate(command)
            exec_argv = ["/bin/sh", "-c", command] if os.name != "nt" else ["cmd", "/c", command]
        else:
            raise ToolError("provide either `argv` (preferred) or `command`.")
        outcome = run_subprocess(
            exec_argv, cfg=self.cfg, timeout_s=timeout_s or None
        )
        return self._result(outcome, label)


class RunPythonTool(_ExecTool):
    name = "run_python"
    description = (
        "Run a Python snippet in a fresh subprocess (never the kernel), with the repo "
        "on PYTHONPATH. A stray sys.exit() or an infinite loop cannot kill the session."
    )
    safety = SafetyClass.EXEC
    args = (
        ArgSpec("code", "string", "Python source to execute."),
        ArgSpec("timeout_s", "integer", "Timeout override in seconds.", required=False, default=0),
    )

    def run(self, code: str, timeout_s: int = 0) -> ToolResult:
        self._gate("python -c <snippet>")
        src_dir = self.cfg.repo_root / "src"
        extra_env = {"PYTHONPATH": os.pathsep.join([str(self.cfg.repo_root), str(src_dir)])}
        outcome = run_subprocess(
            [sys.executable, "-c", code],
            cfg=self.cfg,
            timeout_s=timeout_s or None,
            extra_env=extra_env,
        )
        return self._result(outcome, "python -c <snippet>")
