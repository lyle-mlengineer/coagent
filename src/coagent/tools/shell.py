"""Execution tools: shell and Python, isolated at the process level (FR-TOL-3).

There is no container on Colab, so isolation is process-level and the docs say
so. What this module guarantees:

* a scrubbed environment — tokens and API keys never reach a child process
  (FR-TOL-5), and GPU visibility is hidden for verification runs (PRD D4);
* a timeout that kills the whole process group, not just the parent, so a
  grandchild cannot outlive the turn;
* an output cap that keeps the head and the tail and elides the middle, because
  the interesting parts of a build log are the invocation and the error;
* a policy gate, applied to the command string before anything is spawned.

`RunTestsTool` is a thin shell over `tools/pyt.py`: pytest output is parsed into
failing test names plus a bounded digest, because a 500-line pytest dump evicts
the context the model needs to fix the failure (DESIGN §7.5).
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..safety.policy import Policy, SafetyClass, Verdict
from .base import ArgSpec, BaseTool, ToolError, ToolResult
from .pyt import parse_pytest_output, summarise_pytest

__all__ = [
    "ExecConfig",
    "ExecOutcome",
    "RunPythonTool",
    "RunShellTool",
    "RunTestsTool",
    "run_subprocess",
    "scrubbed_env",
]

#: Environment names that must never reach a child process (FR-TOL-5).
#: Matched as a substring, case-insensitively, against the variable name.
_SECRET_MARKERS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
    "auth",
    "session_key",
)

#: Names that look like secrets but are safe and needed to run anything.
_SECRET_ALLOWLIST = frozenset(
    {
        "SSH_AUTH_SOCK",  # a socket path, not a secret
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    }
)

#: Separator inserted where output was elided.
_OMITTED = "\n... [{n} bytes omitted] ...\n"


@dataclass(slots=True)
class ExecConfig:
    """Everything the exec layer needs, fixed once per session."""

    repo_root: Path
    max_output_bytes: int = 65536
    timeout_s: int = 120
    notebook_timeout_s: int = 300
    #: PRD D4: verification is CPU-only, so CUDA is hidden by default.
    hide_gpu: bool = True
    #: Temp dir for generated scripts, outside the repo so it is never committed.
    engine_dir: Path | None = None

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        if self.engine_dir is None:
            self.engine_dir = self.repo_root / ".coagent" / "tmp"


@dataclass(slots=True)
class ExecOutcome:
    """What a finished (or killed) process produced."""

    returncode: int
    output: str
    truncated: bool = False
    timed_out: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def scrubbed_env(cfg: ExecConfig, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a child process sees: no secrets, bounded GPU visibility.

    Secrets are removed rather than blanked, so a child that checks for the
    variable's presence sees the truth: it is not available (FR-TOL-5).
    """
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in _SECRET_ALLOWLIST:
            env[name] = value
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in _SECRET_MARKERS):
            continue
        env[name] = value

    # Keep the repo importable without an install, and stop Python writing
    # .pyc files into a tree the agent is about to diff.
    existing = env.get("PYTHONPATH", "")
    root = str(cfg.repo_root)
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("NO_COLOR", "1")
    # A stray TERM/COLUMNS makes tools emit progress bars and ANSI codes.
    env.pop("COLUMNS", None)
    env.pop("LINES", None)

    if cfg.hide_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Only expose the first device: the T4 is a single-GPU box.
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    if extra:
        env.update(extra)
    return env


def run_subprocess(
    argv: list[str],
    *,
    cfg: ExecConfig,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: int | float | None = None,
    stdin_text: str | None = None,
) -> ExecOutcome:
    """Run `argv` in its own process group with a scrubbed env and an output cap.

    The process group matters: `pytest` spawns workers, and killing only the
    parent leaves orphans holding the pipe open, which turns a 1s timeout into
    a hang until the workers exit on their own.
    """
    cwd_path = Path(cwd) if cwd is not None else cfg.repo_root
    effective_timeout = float(timeout_s if timeout_s is not None else cfg.timeout_s)
    limit = cfg.max_output_bytes

    started = time.monotonic()
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd_path),
        "env": env if env is not None else scrubbed_env(cfg),
        "stdin": subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "errors": "replace",
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:  # pragma: no cover - exercised on Windows only
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        proc = subprocess.Popen(argv, **popen_kwargs)  # noqa: S603 - argv is policy-gated
    except FileNotFoundError as exc:
        return ExecOutcome(
            returncode=127,
            output=f"{argv[0]}: command not found ({exc.strerror or 'no such file'})",
            duration_s=time.monotonic() - started,
        )
    except OSError as exc:
        return ExecOutcome(
            returncode=126,
            output=f"failed to start {argv[0]}: {exc}",
            duration_s=time.monotonic() - started,
        )

    timed_out = False
    try:
        raw, _ = proc.communicate(input=stdin_text, timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        try:
            raw, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
            raw = ""
        raw = (raw or "") + f"\n[timeout after {effective_timeout:.0f}s; process group killed]"

    duration = time.monotonic() - started
    text, truncated = _cap_output(raw or "", limit)
    return ExecOutcome(
        returncode=proc.returncode if proc.returncode is not None else -1,
        output=text,
        truncated=truncated,
        timed_out=timed_out,
        duration_s=duration,
    )


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Kill the process and everything it spawned."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(OSError):
            proc.kill()


def _cap_output(text: str, limit: int) -> tuple[str, bool]:
    """Keep the head and the tail, elide the middle.

    The head says what ran; the tail says what went wrong. The middle of a build
    log is progress noise, so it is the cheapest thing to drop.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False
    head_len = limit // 3
    tail_len = limit - head_len
    head = text[:head_len]
    tail = text[-tail_len:]
    omitted = len(text) - head_len - tail_len
    return head + _OMITTED.format(n=omitted) + tail, True


class _ExecTool(BaseTool):
    """Shared gate and result shaping for the three exec tools."""

    #: Running anything is EXEC, never READ: SURVEY must not be able to spawn a
    #: process, so the class is declared once here rather than defaulted.
    safety = SafetyClass.EXEC

    def __init__(self, cfg: ExecConfig, policy: Policy) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.policy = policy

    def _gate(self, cmd: str) -> None:
        """Deny or ask before anything is spawned (FR-TOL-4)."""
        decision = self.policy.check_shell(cmd)
        if decision.verdict is Verdict.DENY:
            raise ToolError(f"denied: {decision.reason}")
        if decision.verdict is Verdict.ASK:
            # In non-interactive mode ASK resolves to deny, so the message is the
            # same shape either way: say what would unblock it.
            raise ToolError(f"needs approval: {decision.reason} ({cmd.split()[0]})")

    def _result(self, outcome: ExecOutcome, label: str) -> ToolResult:
        content = outcome.output
        if not content.strip():
            content = f"{label}: no output"
        if outcome.timed_out:
            return ToolResult.failure(
                f"timeout: {label} exceeded {self.cfg.timeout_s}s and was killed\n{content}".strip(),
                timed_out=True,
                returncode=outcome.returncode,
                duration_s=outcome.duration_s,
                truncated=outcome.truncated,
            )
        if outcome.returncode != 0:
            return ToolResult.failure(
                f"exit {outcome.returncode}: {label}\n{content}".strip(),
                returncode=outcome.returncode,
                timed_out=False,
                duration_s=outcome.duration_s,
                truncated=outcome.truncated,
            )
        return ToolResult.success(
            content,
            returncode=0,
            timed_out=False,
            duration_s=outcome.duration_s,
            truncated=outcome.truncated,
        )


class RunShellTool(_ExecTool):
    """Run one command in the repo.

    Both an argv list and a shell string are accepted. argv is preferred because
    it needs no quoting rules; the string form is what the model usually emits,
    so it is normalized back to argv rather than handed to a shell.
    """

    name = "run_shell"
    description = (
        "Run a shell command in the repository. Prefer argv (a list of words) over a "
        "command string: it avoids quoting bugs. Use for linters, formatters, git "
        "inspection and build steps. Not for installing packages without approval."
    )
    args = (
        ArgSpec(
            "argv",
            "array",
            "Command as a list of words, e.g. ['pytest', '-q']. Required unless command is given.",
            required=False,
        ),
        ArgSpec(
            "command",
            "string",
            "Command as one string, e.g. 'pytest -q'. Required unless argv is given.",
            required=False,
        ),
        ArgSpec(
            "timeout_s",
            "integer",
            "Seconds before the process group is killed. Defaults to the session timeout.",
            required=False,
        ),
    )

    def run(
        self,
        argv: list[str] | None = None,
        command: str | None = None,
        timeout_s: int | None = None,
    ) -> ToolResult:
        cmd_argv = _resolve_argv(argv, command)
        self._gate(" ".join(cmd_argv))
        outcome = run_subprocess(
            cmd_argv,
            cfg=self.cfg,
            timeout_s=timeout_s if timeout_s else None,
        )
        return self._result(outcome, " ".join(cmd_argv))


class RunTestsTool(_ExecTool):
    """Run the verify command and return failing test names, not a wall of text.

    The digest caps at `feedback_max_tokens` (approximated as 4 chars/token) so
    the model gets the verdict, the failing node ids, and the last evidence for
    each — without the passing-test noise (DESIGN §7.5).
    """

    name = "run_tests"
    description = (
        "Run the test suite (or one test file) and get the failures back as a short "
        "digest: pass/fail verdict, failing test names, and the error for each. Use "
        "this instead of run_shell when you want pytest output you can act on."
    )
    args = (
        ArgSpec(
            "command",
            "string",
            "Test command, e.g. 'pytest -q' or 'pytest tests/test_x.py::test_y'.",
            required=False,
            default="pytest -q",
        ),
        ArgSpec(
            "argv",
            "array",
            "Command as a list of words. Prefer this when the command is already tokenized, "
            "e.g. from [verify] commands in config.",
            required=False,
        ),
        ArgSpec(
            "max_tokens",
            "integer",
            "Digest budget in tokens (approximate). Defaults to the verify config.",
            required=False,
        ),
    )

    def __init__(
        self,
        cfg: ExecConfig,
        policy: Policy,
        *,
        verify_commands: list[str] | None = None,
        feedback_max_tokens: int = 1500,
    ) -> None:
        super().__init__(cfg, policy)
        self.verify_commands = list(verify_commands or ["pytest -q"])
        self.feedback_max_tokens = feedback_max_tokens

    def run(
        self,
        command: str = "pytest -q",
        argv: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> ToolResult:
        # argv wins when given: a tokenized command (as config carries it) must
        # not be re-split, or a quoted interpreter path loses its meaning.
        cmd_argv = _resolve_argv(argv, command)
        label = " ".join(cmd_argv)
        self._gate(label)

        budget = int(max_tokens) if max_tokens else self.feedback_max_tokens
        cap_chars = max(200, budget * 4)

        outcome = run_subprocess(cmd_argv, cfg=self.cfg)
        parsed = parse_pytest_output(outcome.output)
        digest = summarise_pytest(outcome.output, cap=cap_chars)

        meta: dict[str, Any] = {
            "returncode": outcome.returncode,
            "timed_out": outcome.timed_out,
            "duration_s": outcome.duration_s,
            "truncated": outcome.truncated,
            "failures": parsed.failed_names,
            "failed": parsed.failures,
            "passed": parsed.passed,
        }
        if outcome.returncode == 0 and not outcome.timed_out and not parsed.failures:
            # Keep the output when there is any: a passing lint or a script that
            # prints a count is information, and "tests passed" would discard it.
            return ToolResult.success(digest or "tests passed", **meta)
        if outcome.timed_out:
            return ToolResult.failure(
                f"timeout: tests exceeded {self.cfg.timeout_s}s and were killed\n{digest}".strip(),
                **meta,
            )
        return ToolResult.failure(digest or f"exit {outcome.returncode}", **meta)


class RunPythonTool(_ExecTool):
    """Run a Python snippet in a fresh interpreter inside the repo.

    A fresh process, not the agent's own: `sys.exit` in generated code would
    otherwise end the session, and module-level state would leak between turns.
    """

    name = "run_python"
    description = (
        "Run a short Python snippet in a fresh interpreter with the repo importable. "
        "Use for quick checks, printing values, or exercising a function. The process "
        "is killed on timeout; nothing it prints is lost."
    )
    safety = SafetyClass.EXEC
    args = (
        ArgSpec(
            "code",
            "string",
            "Python source to execute.",
            required=True,
        ),
        ArgSpec(
            "timeout_s",
            "integer",
            "Seconds before the process group is killed. Defaults to the session timeout.",
            required=False,
        ),
    )

    def run(self, code: str, timeout_s: int | None = None) -> ToolResult:
        if not code.strip():
            raise ToolError("run_python needs non-empty code")
        # Gate on the source too: generated code that shells out must be reviewed.
        self._gate(code)
        outcome = run_subprocess(
            [sys.executable, "-c", code],
            cfg=self.cfg,
            timeout_s=timeout_s if timeout_s else None,
        )
        return self._result(outcome, "python snippet")


def _resolve_argv(argv: list[str] | None, command: str | None) -> list[str]:
    """Normalize argv/command to one argv list, or explain what was missing."""
    if argv:
        words = [str(a) for a in argv if str(a) != ""]
        if words:
            return words
    if command and command.strip():
        return _split_command(command)
    raise ToolError("provide either argv or command")


def _split_command(command: str) -> list[str]:
    """Split a command string with shell-like quoting, but no shell features.

    Deliberately no pipes, redirects or `&&`: those need a shell, and handing a
    shell a string the model wrote is the one thing this layer exists to avoid.

    `posix=True` is the default because there is no shell here, so the only thing
    posix mode changes is quote handling: non-posix keeps the quotes and turns
    `python -c "print(7)"` into a program whose source is literally `"print(7)"`.
    On Windows, however, posix mode eats backslashes in `C:\\Python\\python.exe`,
    so the first token is re-checked against the raw string and the non-posix
    split wins when posix mangled an existing path.
    """
    try:
        import shlex

        words = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ToolError(f"could not parse command {command!r}: {exc}") from None

    if words and os.name == "nt":
        repaired = _repair_windows_exe(words[0], command)
        if repaired is not None:
            words[0] = repaired
    return [
        w[1:-1] if len(w) > 1 and w[0] == w[-1] == '"' else w
        for w in words
    ]


def _repair_windows_exe(token: str, command: str) -> str | None:
    """Recover an unquoted `C:\\Python\\python.exe` that posix splitting mangled.

    posix mode treats the backslashes as escapes, so a bare Windows path loses
    them. When the raw first word exists on disk and the split token does not,
    the raw word is the truth. This runs only on Windows and only for a token
    that looks like an executable, so it cannot silently rewrite an odd command.
    """
    raw_first = command.strip().split(" ", 1)[0].strip('"')
    if ".exe" not in raw_first.lower():
        return None
    if Path(token).exists() or not Path(raw_first).exists():
        return None
    return raw_first


#: Matches an ANSI escape sequence, so captured output stays readable in a diff.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:  # pragma: no cover - convenience helper
    return _ANSI.sub("", text)


def write_scratch(cfg: ExecConfig, suffix: str = ".py") -> Path:
    """A temp path under the scratch dir, never inside the tracked tree."""
    scratch = Path(cfg.engine_dir or (cfg.repo_root / ".coagent" / "tmp"))
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=suffix, dir=str(scratch))
    os.close(fd)
    return Path(name)
