"""The agent loop (DESIGN §7).

One explicit state machine: INTAKE → SURVEY → PLAN → ACT → VERIFY → CHECKPOINT →
(ACT | REPORT). No framework, no hidden prompt layer — with a 7B model, every
layer you cannot read is a debugging cost you cannot pay.

Deliberate properties, each traceable to a requirement:

* **PLAN cannot act.** It gets an empty tool registry (FR-AGT-2 / DESIGN §7.1).
* **One write or exec per turn.** A multi-write message is rejected with a
  corrective prompt (FR-AGT-3).
* **Read before write** is enforced by the tool, not by the prompt (FR-AGT-4).
* **Verify after write.** A write that cannot pass a syntax check never commits
  (FR-AGT-7).
* **Definition of done** is a code predicate: checks pass AND every plan item is
  ticked AND no new stubs (FR-AGT-8).
* **Never a silent stop.** Budget exhaustion checkpoints and reports (FR-AGT-5).
"""

from __future__ import annotations

import contextlib
import json
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, load_config
from ..gitx.commit import CommitMeta
from ..gitx.durable import SyncResult
from ..gitx.git import GitError
from ..gitx.log import LogRecord, SessionLog
from ..gitx.session import Session, open_session
from ..llm.base import BudgetError, GenerationConfig, LLMBackend, Message, TokenBudget
from ..llm.grammar import EnvelopeError, looks_like_final_answer, parse_envelope
from ..llm.registry import get_model_by_name
from ..safety.policy import RiskClass, SafetyClass, Verdict
from ..safety.redact import Redactor
from ..tools import Phase, build_policy, build_registry
from ..tools.base import ToolError, ToolRegistry, ToolResult, ToolSchemaError
from . import prompts
from .budget import Budget, BudgetExhausted, BudgetLimits, LoopDetector, LoopVerdict
from .state import PlanFile, SessionState, slugify

__all__ = ["AgentResult", "coAgent", "ApprovalRequest", "Approver"]


Approver = Callable[["ApprovalRequest"], bool]


@dataclass(slots=True)
class ApprovalRequest:
    """A risky action awaiting a human decision (DESIGN §9.1)."""

    kind: str  # a RiskClass value
    command: str
    reason: str

    def render(self) -> str:
        return f"[{self.kind}] {self.command}\n  why: {self.reason}"


@dataclass
class AgentResult:
    task: str
    done: bool
    branch: str
    session_id: str
    steps: int = 0
    commits: int = 0
    files_changed: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    blocked_reason: str = ""
    report: str = ""
    sync: SyncResult | None = None
    stopped_by: str = ""

    def render(self) -> str:
        verdict = "done" if self.done else "not done"
        lines = [
            f"coAgent · {verdict} · session {self.session_id}",
            f"branch: {self.branch}",
            f"steps: {self.steps} · commits: {self.commits}",
        ]
        if self.files_changed:
            lines.append(f"files: {', '.join(self.files_changed[:10])}")
        if self.artifacts:
            lines.append(f"artifacts: {', '.join(self.artifacts)}")
        if self.blocked_reason:
            lines.append(f"blocked: {self.blocked_reason}")
        if self.sync is not None:
            lines.append(f"durable: {self.sync.message}")
        if self.report:
            lines.append("")
            lines.append(self.report.strip())
        return "\n".join(lines)


_WRITE_TOOLS = {"edit_file", "write_file", "edit_notebook"}
_EXEC_TOOLS = {"run_shell", "run_python"}
#: Tools whose success means a file on disk changed and a verify+commit is due.
_COMMIT_TOOLS = _WRITE_TOOLS | {"exec_notebook"}
#: Tools that leave a path read (and, for exec_notebook, rewritten).
_READ_TOOLS = {"read_file", "read_notebook"}

# Files that must never be committed on their own (state, not code).
_PLAN_PREFIX = ".coagent/plans/"


class coAgent:
    """The agent. Construct once per session, then `chat()` or `run()`."""

    def __init__(
        self,
        repo: str | Path = ".",
        *,
        model: str | None = None,
        backend: LLMBackend | None = None,
        config: Config | None = None,
        interactive: bool = True,
        auto: bool = False,
        session_id: str | None = None,
        token: str | None = None,
        approver: Approver | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        force: bool = False,
    ) -> None:
        self.cfg = config or load_config(start=repo)
        self.repo_path = Path(repo).resolve()
        self.interactive = interactive
        self.auto = auto
        self.token = token
        self.approver = approver
        self.on_event = on_event or (lambda _kind, _data: None)
        self.redactor = Redactor([token] if token else [])

        if backend is None:
            from ..llm import make_backend

            spec = get_model_by_name(self.cfg.model.name)
            key = spec.key if spec else "qwen-coder-7b"
            backend = make_backend(
                "transformers", model_key=key, max_context=self.cfg.model.max_context
            )
        self.backend = backend
        self.model_id = model or getattr(backend, "model_id", "unknown")

        self.session: Session = open_session(
            self.repo_path,
            session_id=session_id,
            force=force,
            author_name=self.cfg.git.author_name,
            author_email=self.cfg.git.author_email,
            token=token,
        )
        self.policy = build_policy(
            self.repo_path, self.cfg, interactive=interactive and not auto
        )
        self.registry: ToolRegistry = build_registry(
            self.repo_path, self.cfg, phase=Phase.ACT, policy=self.policy
        )
        self.log = SessionLog(
            self.repo_path, self.session.session_id, redactor=self.redactor
        )
        self.budget = Budget(limits=BudgetLimits.from_config(self.cfg))
        self.loop_detector = LoopDetector(
            replan_after=self.cfg.agent.replan_after_repeats,
            stop_after=self.cfg.agent.stop_after_repeats,
        )
        self.state: SessionState | None = None
        self._cancelled = False
        self._pending_sync_tip: str | None = None
        # Per-instance, not class-level: two agents in one process must not
        # share a "last failure" or the loop would cross-contaminate.
        self._last_notes = ""
        self._last_failure = ""
        self._last_was_write = False

    # ------------------------------------------------------------- public API

    @property
    def branch(self) -> str:
        return self.session.branch

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def status_line(self) -> str:
        ctx = self.cfg.model.max_context
        return (
            f"coAgent · {self.model_id} · {ctx // 1000}k ctx · "
            f"session {self.session.session_id} · {self.budget.describe()}"
        )

    def cancel(self) -> None:
        """Stop the next generation or command without losing the checkpoint."""
        self._cancelled = True

    def chat(self, task: str | None = None) -> AgentResult:
        """Run one task. With no task, ask the approver/user for one."""
        if task is None:
            task = self._prompt_text("What should coAgent do?")
        return self.run(task)

    def run(self, task: str) -> AgentResult:
        """survey → plan → (approve) → act/verify loop → checkpoint → report."""
        self.state = SessionState(
            task=task, plan=PlanFile(self.repo_path, slugify(task))
        )
        self._emit("task_start", {"task": task, "branch": self.branch})
        result = AgentResult(task=task, done=False, branch=self.branch, session_id=self.session_id)

        try:
            self._phase_survey()
            plan_ok = self._phase_plan()
            if not plan_ok:
                result.blocked_reason = "the plan was rejected; nothing was written"
                result.stopped_by = "plan_rejected"
                result.report = "The plan was rejected, so no files were changed."
                return result

            self._act_loop(result)
        except BudgetExhausted as exc:
            result.stopped_by = exc.kind
            result.blocked_reason = exc.detail
        except KeyboardInterrupt:
            result.stopped_by = "interrupt"
            result.blocked_reason = "interrupted by the user"
        finally:
            self._final_checkpoint(result)
            result.steps = self.budget.steps
            result.commits = self.budget.commits
            result.files_changed = sorted(self.state.files_written) if self.state else []
            result.artifacts = list(self.state.artifacts) if self.state else []
            try:
                result.report = self._phase_report(result)
            except (BudgetError, BudgetExhausted) as exc:  # report must never fail the run
                result.report = f"(report skipped: {exc})"
            self._emit("task_end", {"done": result.done, "stopped_by": result.stopped_by})

        return result

    def resume(self) -> str:
        """Print the briefing for this session (FR-GIT-5)."""
        from ..gitx.resume import build_briefing

        self._emit("resume", {})
        return build_briefing(self.session.repo, self.session_id, self.branch).render()

    def push(self) -> SyncResult:
        from ..gitx.durable import sync_push

        return sync_push(self.session.repo, self.cfg.durability.remote, self.branch, policy=self.policy)

    def close(self) -> None:
        self.session.close()

    # ---------------------------------------------------------------- phases

    def _phase_survey(self) -> str:
        """Read-only exploration. Returns notes for the PLAN prompt."""
        assert self.state is not None
        registry = build_registry(self.repo_path, self.cfg, phase=Phase.SURVEY, policy=self.policy)
        repo_map = self._repo_map()
        prompt = prompts.survey_prompt(
            task=self.state.task,
            repo_map=repo_map,
            tool_list=", ".join(registry.names()),
            budget_note=f"Budget: {self.budget.describe()}.",
        )
        notes = self._generate_text(
            [Message("system", prompts.SYSTEM_CORE), Message("user", prompt)],
            temperature=self.cfg.model.temperature_edit,
            phase="survey",
        )
        self._emit("survey", {"notes": notes})
        return notes

    def _phase_plan(self) -> bool:
        """Write the plan file. PLAN has no tools, by construction (FR-AGT-2)."""
        assert self.state is not None
        prompt = prompts.plan_prompt(
            task=self.state.task,
            survey_notes=(self._last_notes or "(no survey notes)"),
            files_touched=", ".join(sorted(self.state.files_read)[:15]) or "(none yet)",
            require_tests=self.cfg.verify.require_tests,
        )
        text = self._generate_text(
            [Message("system", prompts.SYSTEM_CORE), Message("user", prompt)],
            temperature=self.cfg.model.temperature_plan,
            phase="plan",
            max_new_tokens=min(2048, self.cfg.model.max_new_tokens * 2),
        )
        steps = self._parse_plan(text)
        if not steps:
            # A plan with no checklist cannot be tracked, so synthesise one from
            # the survey rather than proceeding without memory.
            steps = self._fallback_plan()
        title, steps = self._split_title(text, steps)
        path = self.state.plan.write(title, steps)
        self._emit("plan", {"path": str(path), "title": title, "steps": [s.text for s in steps]})

        if self.auto:
            return True
        decision = self._approve(
            ApprovalRequest(
                kind="plan",
                command=str(path),
                reason="\n".join(step.checkbox() for step in steps),
            )
        )
        if not decision:
            self._emit("plan_rejected", {})
            return False
        return True

    def _act_loop(self, result: AgentResult) -> None:
        """ACT → VERIFY → CHECKPOINT until done or out of budget."""
        assert self.state is not None
        while True:
            if self._cancelled:
                result.stopped_by = "cancelled"
                result.blocked_reason = "stopped by the user"
                return
            self.budget.check()

            if self.state.plan.complete:
                if self._verify_and_commit(result):
                    result.done = True
                    result.stopped_by = "complete"
                    return
                # Checks failed with a complete plan: reopen the failure as work.
                self._emit("verify_failed_after_complete", {})

            action = self._next_action()
            if action is None:
                # The model answered in prose rather than calling a tool.
                result.stopped_by = "no_action"
                result.blocked_reason = "the model stopped proposing actions"
                return
            tool, args = action

            verdict = self.loop_detector.observe(tool, args)
            if verdict is LoopVerdict.STOP:
                result.stopped_by = "loop"
                result.blocked_reason = self.loop_detector.corrective(LoopVerdict.STOP)
                return
            if verdict is LoopVerdict.REPLAN:
                self._emit("replan", {"corrective": self.loop_detector.corrective(verdict)})
                self._force_replan()
                continue

            self.budget.record_step()
            executed = self._execute(tool, args)
            self.state.add_observation(tool, executed.ok, executed.content, step=self.budget.steps)
            if not executed.ok:
                try:
                    self.budget.record_failure()
                except BudgetExhausted:
                    result.stopped_by = "consecutive_failures"
                    result.blocked_reason = (
                        f"{self.budget.consecutive_failures} consecutive failures; "
                        "the last error is in the session log"
                    )
                    return
            else:
                self.budget.record_success()

            if tool in _COMMIT_TOOLS and executed.ok:
                self._verify_and_commit(result)

    def _phase_report(self, result: AgentResult) -> str:
        assert self.state is not None
        if not result.done and not result.blocked_reason:
            result.blocked_reason = "the task did not reach a verified state"
        prompt = prompts.report_prompt(
            task=self.state.task,
            plan=self.state.plan.render(),
            results=self.state.last_verification or "(no verification ran)",
            branch=self.branch,
            artifacts=result.artifacts,
        )
        try:
            text = self._generate_text(
                [Message("system", prompts.SYSTEM_CORE), Message("user", prompt)],
                temperature=0.5,
                phase="report",
                max_new_tokens=400,
            )
        except (BudgetError, BudgetExhausted):
            text = ""
        # REPORT has no tools. A model that answers with an envelope is not
        # producing a report, and pasting raw JSON to the user is worse than a
        # generated summary.
        if not text.strip() or self._looks_like_envelope(text):
            text = self._static_report(result)
        return text

    # ------------------------------------------------------------- act helper

    def _next_action(self) -> tuple[str, dict[str, Any]] | None:
        """Ask the model for exactly one action."""
        assert self.state is not None
        schemas = json.dumps(self.registry.schemas(), indent=None, separators=(",", ":"))
        prompt = prompts.act_prompt(
            task=self.state.task,
            plan=self.state.plan.render(),
            observations=prompts.format_observations(self.state.observation_pairs()),
            tool_schemas=schemas,
            next_step=self.budget.steps + 1,
            failures=self._last_failure,
            retrieved=self._retrieve(),
        )
        messages = [Message("system", prompts.SYSTEM_CORE), Message("user", prompt)]

        for attempt in (1, 2):
            text = self._generate_text(
                messages,
                temperature=self.cfg.model.temperature_edit,
                phase="act",
                max_new_tokens=self.cfg.model.max_new_tokens,
            )
            if looks_like_final_answer(text) and "{" not in text:
                return None
            try:
                env = parse_envelope(text, allowed_tools=set(self.registry.names()))
                return env.tool, env.args
            except EnvelopeError as exc:
                self._emit("envelope_error", {"corrective": exc.corrective})
                self._last_failure = exc.corrective
                if attempt == 2:
                    self.budget.record_failure()
                    return None
                messages.append(Message("assistant", text))
                messages.append(Message("user", exc.corrective))
        return None

    def _execute(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Enforce the one-write rule, approvals, then dispatch."""
        try:
            spec = self.registry.get(tool)
        except ToolSchemaError as exc:
            return ToolResult.failure(str(exc))

        if (spec.safety in {SafetyClass.WRITE, SafetyClass.EXEC}) and self._last_was_write:
            return ToolResult.failure(
                "You made a write or exec call in the previous message. Send only one "
                "write or exec action per message; batch reads instead."
            )

        # Approvals for risky classes, decided *before* anything runs.
        denied = self._gate(tool, args)
        if denied is not None:
            return denied

        try:
            result = self.registry.call(tool, args)
        except ToolSchemaError as exc:
            self._last_failure = str(exc)
            return ToolResult.failure(str(exc))
        except ToolError as exc:
            self._last_failure = str(exc)
            return ToolResult.failure(str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool crash must not kill the loop
            self._last_failure = f"{type(exc).__name__}: {exc}"
            return ToolResult.failure(f"tool crashed: {self._last_failure}")

        self._last_was_write = spec.safety in {SafetyClass.WRITE, SafetyClass.EXEC}
        if result.ok:
            self._last_failure = ""
        else:
            self._last_failure = result.content[:800]
            self.state and self.state.add_observation(tool, False, result.content)

        if tool in _WRITE_TOOLS and result.ok:
            path = str(args.get("path", ""))
            if path:
                self.state and self.state.note_written(path)
                # A step that names this file is satisfied by editing it.
                self.state and self.state.plan.tick_matching(path)
                self.state and self.state.plan.write(
                    self.state.plan.title, self.state.plan.steps
                )
        elif tool == "exec_notebook" and result.ok:
            # Execution rewrites the notebook in place with outputs, so it is a
            # write as well as a check: the file must be staged and committed.
            path = str(args.get("path", ""))
            if path:
                self.state and self.state.note_written(path)
                self.state and self.state.plan.tick_matching(path)
        elif tool in _READ_TOOLS and result.ok:
            path = str(args.get("path", ""))
            if path:
                self.state and self.state.note_read(path)
                # Ticking read steps only counts once something else is done too;
                # otherwise a plan of pure reads would report complete.
                if len(self.state.files_written) > 0:
                    self.state.plan.tick_matching(path)

        self.log.append(
            LogRecord(
                step=self.budget.steps,
                phase=Phase.ACT,
                tool=tool,
                ok=result.ok,
                tokens_in=0,
                tokens_out=0,
                error="" if result.ok else result.content[:400],
                note=str(args.get("path", ""))[:200],
            )
        )
        self._emit(
            "tool",
            {"tool": tool, "ok": result.ok, "content": self.redactor.redact(result.content)[:2000]},
        )
        return result

    def _gate(self, tool: str, args: dict[str, Any]) -> ToolResult | None:
        """Return a failure result when an approval was refused."""
        if tool in _EXEC_TOOLS:
            command = args.get("command") or " ".join(args.get("argv") or [])
            decision = self.policy.check_shell(str(command))
            if decision.verdict is Verdict.DENY:
                return ToolResult.failure(f"denied: {decision.reason}")
            if decision.verdict is Verdict.ASK:
                kind = self._risk_of(str(command))
                if not self._approve(ApprovalRequest(kind, str(command), decision.reason)):
                    self._emit("denied", {"command": str(command)})
                    return ToolResult.failure(
                        f"the user denied this command: {decision.reason}. Choose another approach."
                    )
        return None

    @staticmethod
    def _risk_of(command: str) -> str:
        low = command.lower()
        if "pip install" in low or "apt" in low or "conda install" in low:
            return RiskClass.INSTALL.value
        if "rm " in low or "mv " in low or "chmod" in low:
            return RiskClass.DELETE.value
        return RiskClass.NETWORK.value

    # --------------------------------------------------------------- verify

    def _verify_and_commit(self, result: AgentResult) -> bool:
        """Run configured checks; commit only if they pass (FR-AGT-7)."""
        assert self.state is not None
        commands = self.cfg.verify.commands
        if not commands:
            self.state.last_verification = "no verify.commands configured"
            return False

        outputs: list[str] = []
        all_ok = True
        for command in commands:
            argv = shlex.split(command) if isinstance(command, str) else [str(a) for a in command]
            label = command if isinstance(command, str) else " ".join(argv)
            try:
                tool_result = self.registry.call("run_shell", {"argv": argv})
            except (ToolError, ToolSchemaError) as exc:
                tool_result = ToolResult.failure(str(exc))
            outputs.append(f"$ {label}\n{tool_result.content[-1200:]}")
            if not tool_result.ok:
                all_ok = False
                break

        summary = "\n".join(outputs)
        self.state.last_verification = summary
        self._emit("verify", {"ok": all_ok, "summary": summary[-1500:]})

        if not all_ok:
            self._last_failure = summary[-1200:]
            try:
                self.budget.record_failure()
            except BudgetExhausted:
                raise
            return False

        self.budget.record_success()
        # Passing checks complete the plan item that named them, which is what
        # makes "done" a predicate rather than the model's opinion (FR-AGT-8).
        self._tick_verification_steps()
        self._commit_step(result, verify_summary=self._short_verify(commands))
        return True

    def _tick_verification_steps(self) -> None:
        assert self.state is not None
        plan = self.state.plan
        for keyword in ("test", "check", "verify", "run", "lint"):
            if plan.tick_matching(keyword):
                break
        plan.write(plan.title, plan.steps)

    def _commit_step(self, result: AgentResult, *, verify_summary: str) -> None:
        """Commit tracked changes with the full trailer set (FR-GIT-3)."""
        assert self.state is not None
        repo = self.session.repo
        if not repo.is_dirty(ignore=(_PLAN_PREFIX, ".coagent/lock.json")):
            # Nothing on disk changed, so there is no step to record.
            return

        # Stage the work before committing. `git commit` does not stage for you:
        # without this the commit fails with a bare exit 1 and empty stderr, and
        # the step is silently lost.
        self._stage_changes()
        if not repo.staged_paths():
            return

        step = self.state.bump_step()
        meta = CommitMeta(
            session=self.session_id,
            step=step,
            task=self.state.task,
            verify=verify_summary,
            model=self.model_id,
            artifacts=self.state.artifacts,
            subject=f"coagent: {self.state.plan.title}"[:72],
            body=self._commit_body(),
        )
        try:
            sha = repo.commit_with_trailers(meta.default_subject(), meta.body, meta.trailers())
        except (ValueError, GitError) as exc:
            self._emit("commit_failed", {"error": str(exc)})
            return

        self.budget.record_commit()
        self._emit("commit", {"sha": sha, "step": step})

        # A commit is not a checkpoint; the sync is (DESIGN §3.2).
        from ..gitx.durable import checkpoint

        sync = checkpoint(
            repo,
            branch=self.branch,
            session_id=self.session_id,
            mode=self.cfg.durability.mode,
            remote=self.cfg.durability.remote,
            bundle_dir=self._bundle_dir(),
            previous_tip=self._pending_sync_tip,
            policy=self.policy,
        )
        self._pending_sync_tip = sha
        result.sync = sync
        self._emit("sync", {"ok": sync.ok, "message": sync.message})

    def _stage_changes(self) -> None:
        """Stage tracked and new work, but never coAgent's own runtime state."""
        repo = self.session.repo
        repo.add_all_tracked()
        for path in repo.changed_paths():
            if path.startswith(".coagent/"):
                continue
            try:
                repo.add(path)
            except GitError:
                # A file deleted between listing and staging is not an error.
                continue

    def _commit_body(self) -> str:
        assert self.state is not None
        done = [s.text for s in self.state.plan.steps if s.done]
        return "\n".join(f"- {d}" for d in done[:8]) if done else ""

    def _short_verify(self, commands: list) -> str:
        joined = "; ".join(c if isinstance(c, str) else " ".join(map(str, c)) for c in commands)
        return f"{joined} (passed)"[:150]

    def _final_checkpoint(self, result: AgentResult) -> None:
        """Commit partial work so a stopped session is recoverable (FR-REL-1).

        This commits even when checks failed, which is the point: the work is
        saved rather than lost. The trailer records the real verification state
        so a failing commit is never labelled as passing.
        """
        assert self.state is not None
        try:
            if not result.done and self.state.files_written:
                verify = "failed" if self.state.last_verification and not result.done else "not run"
                self._commit_step(result, verify_summary=f"{verify} (work in progress)")
        except Exception as exc:  # noqa: BLE001 - a checkpoint must not mask the real error
            self._emit("final_checkpoint_failed", {"error": str(exc)})

    # ---------------------------------------------------------------- helpers

    def _generate_text(
        self,
        messages: list[Message],
        *,
        temperature: float,
        phase: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Budget-checked generation with token accounting."""
        limit = max_new_tokens or self.cfg.model.max_new_tokens
        budget = TokenBudget(
            max_context=self.cfg.model.max_context,
            count=self.backend.count_tokens,
        )
        used = budget.check(messages, max_new_tokens=limit)
        self.budget.record_tokens(used)

        started = time.monotonic()
        result = self.backend.generate(
            messages,
            GenerationConfig(
                max_new_tokens=limit,
                temperature=temperature,
                top_p=self.cfg.model.top_p,
                seed=self.cfg.model.seed,
            ),
            cancel=lambda: self._cancelled,
        )
        self.budget.record_gpu(time.monotonic() - started)
        self.budget.record_tokens(result.usage.completion_tokens)

        if phase == "survey":
            self._last_notes = result.text
        if result.truncated:
            self._emit("truncated", {"phase": phase})
        return result.text

    def _parse_plan(self, text: str) -> list:
        from .prompts import PlanStep

        steps: list[PlanStep] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- [ ]", "- [x]", "- [X]")):
                steps.append(PlanStep(stripped[5:].strip(), done=stripped[3].lower() == "x"))
            elif stripped.startswith(("- ", "* ")) and len(stripped) > 3:
                steps.append(PlanStep(stripped[2:].strip()))
        return steps

    def _split_title(self, text: str, steps: list) -> tuple[str, list]:
        """Use the plan's own H1; otherwise fall back to the task, not a step."""
        for line in text.splitlines():
            if line.strip().startswith("# "):
                return line.strip()[2:].strip(), steps
        assert self.state is not None
        return self.state.task.strip()[:80] or "task", steps

    def _fallback_plan(self) -> list:
        from .prompts import PlanStep

        assert self.state is not None
        steps = [PlanStep(f"read {p}") for p in sorted(self.state.files_read)[:3]]
        steps.append(PlanStep("make the change the task asks for"))
        steps.append(PlanStep("run " + " and ".join(self.cfg.verify.commands)))
        return steps

    def _force_replan(self) -> None:
        """Re-derive the plan from current state after a repeated action."""
        assert self.state is not None
        self.state.plan.read()
        if not self.state.plan.steps:
            self.state.plan.write(self.state.task, self._fallback_plan())

    def _repo_map(self) -> str:
        """Ranked, signature-only map of the repository (DESIGN §5.2).

        The task text and the files this session has already touched are fed in
        as ranking signals so the map leads with what matters. Budget comes from
        the repo map's §5.1 slot; a repo with no Python degrades to a note
        rather than an empty prompt.
        """
        mapper = self._mapper()
        if mapper is None:
            return "(repo map unavailable; explore with list_dir and grep)"
        try:
            return mapper.render()
        except Exception as exc:  # noqa: BLE001 - a map must never end a session
            self._emit("repo_map_degraded", {"error": str(exc)})
            return "(repo map unavailable; explore with list_dir and grep)"

    def _mapper(self):  # noqa: ANN202 - RepoMap, imported lazily
        """Build the repo map once per session and reuse it.

        Parsing is the only real cost in the context layer, and the ACT loop asks
        for it every turn. The map is invalidated when a file is written, since
        that is the only thing that can change a signature.
        """
        assert self.state is not None
        from ..context.repo_map import RepoMap, RepoMapConfig

        written = frozenset(self.state.files_written)
        cached = getattr(self, "_map_cache", None)
        if cached is not None and cached[0] == written:
            return cached[1]

        touched = set(self.state.files_written) | set(self.state.files_read)
        mapper = RepoMap(
            self.repo_path,
            task=self.state.task,
            recently_edited=touched,
            config=RepoMapConfig(
                max_tokens=self.cfg.context.repo_map_tokens,
                max_files=self.cfg.context.max_files,
            ),
        )
        self._map_cache = (written, mapper)
        return mapper

    def _retrieve(self) -> str:
        """BM25 excerpts for the current plan step (DESIGN §5.3).

        The query is the task plus the first unchecked plan item, because that is
        what the model is about to act on — retrieving against the whole task on
        every step would return the same five chunks for 40 turns. Files already
        read are excluded so the excerpt slot only ever shows new code.
        """
        assert self.state is not None
        mapper = self._mapper()
        if mapper is None:
            return ""
        try:
            from ..context.retrieval import RetrievalConfig, Retriever

            query = self.state.task
            pending = next((s for s in self.state.plan.steps if not s.done), None)
            if pending is not None:
                query = f"{query} {pending.text}"

            retriever = Retriever(
                mapper,
                config=RetrievalConfig(max_tokens=self.cfg.context.retrieved_tokens),
            )
            return retriever.render(query, in_context=set(self.state.files_read))
        except Exception as exc:  # noqa: BLE001 - retrieval is an optimisation
            self._emit("retrieval_degraded", {"error": str(exc)})
            return ""

    def _approve(self, request: ApprovalRequest) -> bool:
        if self.auto:
            self._emit("auto_approved", {"kind": request.kind})
            return True
        if self.approver is not None:
            return bool(self.approver(request))
        if not self.interactive:
            return False
        return self._prompt_yes_no(request.render())

    def _prompt_text(self, question: str) -> str:
        try:
            return input(f"{question}\n> ").strip()
        except EOFError:
            return ""

    def _prompt_yes_no(self, question: str) -> bool:
        try:
            answer = input(f"{question}\napprove? [y/N] ").strip().lower()
        except EOFError:
            return False
        return answer in {"y", "yes"}

    def _bundle_dir(self) -> Path | None:
        if self.cfg.durability.mode != "bundle":
            return self.repo_path / ".coagent" / "bundles"
        return self.repo_path / ".coagent" / "bundles"

    @staticmethod
    def _looks_like_envelope(text: str) -> bool:
        """True when the reply is a tool call rather than prose.

        Uses the real parser so a fenced block is recognised as an envelope
        instead of being pasted into the report.
        """
        try:
            parse_envelope(text)
        except EnvelopeError:
            return False
        return True

    def _static_report(self, result: AgentResult) -> str:
        assert self.state is not None
        verdict = "Done" if result.done else "Not done"
        blocked = result.blocked_reason or "nothing"
        return (
            f"1. Landed: {len(result.files_changed)} file(s) changed on {self.branch}.\n"
            f"2. Blocked: {blocked}\n"
            f"3. Verdict: {verdict}. Plan progress {self.state.plan.progress}.\n"
            "4. Next: review the branch diff, then run `coagent push`."
        )

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        # A UI callback must never break the loop.
        with contextlib.suppress(Exception):
            self.on_event(kind, data)
