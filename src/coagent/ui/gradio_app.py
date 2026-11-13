"""Optional Gradio surface for coAgent (DESIGN §8.3).

Scope, deliberately small: this is a *nicety*. The notebook UI (§8.1) must be
good enough that the product works without Gradio, so nothing here is required
by the loop and every import of `gradio` is guarded.

Served inside Colab with::

    from google.colab.output import serve_kernel_port_as_window
    app = GradioApp()
    app.launch()                              # starts on cfg.ui.gradio_port
    serve_kernel_port_as_window(app.port)     # opens it in the Colab tab

`share=True` and ngrok are out of scope by design: flaky, rate-limited, and a
security surface nobody asked for. A bare `server_name="127.0.0.1"` is the whole
of the exposure model.

The interesting logic lives in `ui/render.py` (`Transcript`, `Row`); this module
is the thin widget layer over it, which is why it is the only file that imports
Gradio and the only file without unit tests beyond the guarded-import path.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any

from .render import Transcript, render_status_line, summarise_verify

__all__ = ["GradioApp", "GradioUnavailable", "gradio_available"]


class GradioUnavailable(RuntimeError):
    """Gradio is not installed. The message says how to get it."""


def gradio_available() -> bool:
    """True when `gradio` can be imported. Cheap enough to call per launch."""
    try:
        import gradio  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return True


@dataclass
class _PendingApproval:
    """One approval request waiting on a human, plus the handles to resolve it."""

    request: Any
    decision: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    decided: bool = False


def _require_gradio():
    try:
        import gradio as gr
    except Exception as exc:  # noqa: BLE001
        raise GradioUnavailable(
            "the Gradio UI needs the optional dependency:\n"
            "    pip install 'coagent[ui]'      # or: pip install gradio\n"
            "The notebook surface (coagent.ui.ChatCell) works without it."
        ) from exc
    return gr


class GradioApp:
    """A chat pane, a transcript, and an approval queue, on one port.

    Usage::

        app = GradioApp(agent=agent)   # or: app = GradioApp(); app.attach(agent)
        app.launch()
        app.chat("add a chunked CSV loader")   # from a notebook cell, optionally

    The agent is optional at construction because the usual pattern in a Colab
    cell is to build the UI first and wire the agent in once the model is loaded.
    """

    def __init__(
        self,
        *,
        agent: Any = None,
        port: int | None = None,
        title: str = "coAgent",
        max_rows: int = 200,
        approval_timeout_s: float = 120.0,
    ) -> None:
        self.gr = _require_gradio()
        self.transcript = Transcript(max_rows=max_rows)
        self.agent = agent
        self.title = title
        self.approval_timeout_s = approval_timeout_s
        self._pending: _PendingApproval | None = None
        self._lock = threading.Lock()
        self._running = False

        # The port comes from config when not passed: ui.gradio_port in config.toml.
        self.port = int(port) if port else self._port_from_config(agent)

        self._ui: Any = None
        self.block = None

    # ------------------------------------------------------------- config

    @staticmethod
    def _port_from_config(agent: Any) -> int:
        cfg = getattr(agent, "cfg", None)
        ui = getattr(cfg, "ui", None) if cfg is not None else None
        return int(getattr(ui, "gradio_port", 7860) or 7860)

    # -------------------------------------------------------------- agent

    def attach(self, agent: Any) -> None:
        """Bind the agent and take over its event and approval hooks.

        The hooks are replaced rather than chained because two surfaces drawing
        the same run is a rare intent and always the wrong one: the transcript
        would render every event twice.
        """
        self.agent = agent
        agent.on_event = self.on_event
        agent.renderer = self.render_artifact
        agent.approver = self.approver
        if self.port == 7860:
            self.port = self._port_from_config(agent)

    # ------------------------------------------------------------- events

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        """`coAgent(on_event=app.on_event)`. Never raises into the loop."""
        try:
            self.transcript.handle(kind, data)
            if kind == "task_end":
                self._running = False
                self._clear_pending()
            self._refresh()
        except Exception as exc:  # noqa: BLE001 - a broken UI must not end the run
            print(f"[ui] could not update the Gradio transcript: {exc}")

    def render_artifact(self, kind: str, payload: str, mime: str) -> None:
        """`coAgent(renderer=app.render_artifact)`.

        Gradio can show an image or a file but has no inline "artifact" concept,
        so the transcript records the path and the notebook cell stays the place
        where artifacts are actually displayed. This keeps the two surfaces
        honest about their different capabilities instead of faking parity.
        """
        self.transcript.note_artifacts([payload] if payload else [])
        self._refresh()

    # ---------------------------------------------------------- approvals

    def approver(self, request: Any) -> bool:
        """Block the loop's thread until a button is pressed, or time out to deny.

        Timeout denies, matching the policy's non-interactive default: the safe
        answer to "nobody is looking" is no (DESIGN §9.1).
        """
        pending = _PendingApproval(request=request)
        with self._lock:
            self._pending = pending
        self._refresh()
        got = pending.decision.wait(timeout=self.approval_timeout_s)
        with self._lock:
            if self._pending is pending:
                self._pending = None
        self._refresh()
        return bool(got and pending.approved)

    def _resolve(self, approved: bool) -> None:
        with self._lock:
            pending = self._pending
            if pending is None:
                return
            pending.approved = approved
            pending.decided = True
            self._pending = None
        pending.decision.set()

    def _clear_pending(self) -> None:
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is not None:
            # Unblock the loop thread rather than leaving it waiting on a run
            # that has already ended.
            pending.approved = False
            pending.decision.set()

    # ---------------------------------------------------------------- chat

    def chat(self, task: str) -> Any:
        """Run a task on the attached agent, off the UI thread.

        The prompt is cleared as soon as it is submitted so a second click cannot
        submit the same text twice while the first run is still going.
        """
        if self.agent is None:
            raise RuntimeError("GradioApp has no agent: pass agent=... or call attach(agent).")
        self._running = True
        try:
            return self.agent.chat(task)
        finally:
            self._running = False
            self._refresh()

    def submit(self, task: str, _history: Any = None) -> tuple[str, str, str, str]:
        """The callback behind the Send button.

        Returns ``(transcript_markdown, status, prompt_value, approval_html)`` so
        Gradio can update four components from one event. Errors are reported in
        the transcript and not raised: a traceback in the UI tells the user
        nothing they can act on.
        """
        task = (task or "").strip()
        if not task:
            return self._view()
        if self.agent is None:
            self.transcript.handle("info", {"notes": "no agent attached yet"})
            return self._view()

        # Run synchronously on the Gradio worker: the loop emits approval
        # requests mid-run and the buttons must stay live, which they do because
        # `approver` blocks the loop's own thread, not the server's.
        try:
            self.chat(task)
        except Exception as exc:  # noqa: BLE001 - surface it, do not crash the server
            self.transcript.handle("info", {"notes": f"run failed: {exc}"})
        return self._view()

    def _view(self) -> tuple[str, str, str, str]:
        status = self._status_line()
        return (
            self.transcript.render(status_line=status),
            status,
            "",
            self._approval_html(),
        )

    def _status_line(self) -> str:
        agent = self.agent
        if agent is None:
            return "coAgent · no agent attached"
        budget = ""
        state = getattr(agent, "state", None)
        if state is not None:
            summary = getattr(state, "plan_summary", "") or ""
            if summary:
                budget = summary
        return render_status_line(
            model_id=getattr(agent, "model_id", "?"),
            ctx_tokens=int(getattr(agent, "ctx_tokens", 0) or 0),
            session_id=getattr(agent, "session_id", "?"),
            budget=budget,
        )

    def _approval_html(self) -> str:
        """The pending request, or empty. HTML because it must not render a plan."""
        with self._lock:
            pending = self._pending
        if pending is None:
            return ""
        req = pending.request
        command = str(getattr(req, "command", ""))
        reason = str(getattr(req, "reason", ""))
        kind = str(getattr(req, "kind", "action"))
        # Escape before interpolating: `command` is model output, and this is HTML.
        import html as _html

        return (
            f"<div style='border-left:3px solid #d97706;padding:8px 12px'>"
            f"<b>Approval needed</b> — <code>{_html.escape(kind)}</code><br>"
            f"<code>{_html.escape(command)}</code><br>"
            f"<small>{_html.escape(reason)}</small></div>"
        )

    def verify_line(self, summary: str) -> str:
        """One-line verify verdict, shared with the notebook surface."""
        return summarise_verify(summary)

    def _refresh(self) -> None:
        """Push the current view to the widgets, if the UI is built."""
        if self._ui is None:
            return
        try:
            body, status, prompt, approval = self._view()
            self._ui["transcript"].value = body
            self._ui["status"].value = status
            self._ui["prompt"].value = prompt
            self._ui["approval"].value = approval
            self._ui["buttons_row"].visible = bool(approval)
        except Exception:  # noqa: BLE001 - widget may not exist yet
            pass

    # --------------------------------------------------------------- build

    def build(self) -> Any:
        """Construct the Blocks UI. Idempotent: repeated calls return the same app."""
        if self.block is not None:
            return self.block
        gr = self.gr

        with gr.Blocks(title=self.title) as block:
            gr.Markdown(f"# {self.title}")
            self._ui = {}
            self._ui["status"] = gr.Markdown(self._status_line())
            self._ui["transcript"] = gr.Markdown("coAgent is idle.")
            self._ui["approval"] = gr.HTML("")
            with gr.Row(visible=False) as buttons_row:
                approve = gr.Button("Approve", variant="primary", size="sm")
                deny = gr.Button("Deny", variant="stop", size="sm")
            self._ui["buttons_row"] = buttons_row
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Describe the task, e.g. 'add a chunked CSV loader with tests'",
                    show_label=False,
                    scale=8,
                    lines=1,
                )
                send = gr.Button("Send", variant="primary", scale=1, size="sm")
            self._ui["prompt"] = prompt

            outputs = [self._ui["transcript"], self._ui["status"], prompt, self._ui["approval"]]
            send.click(self.submit, inputs=[prompt], outputs=outputs)
            prompt.submit(self.submit, inputs=[prompt], outputs=outputs)
            approve.click(lambda: (self._resolve(True), self._view())[1], outputs=outputs)
            deny.click(lambda: (self._resolve(False), self._view())[1], outputs=outputs)

        self.block = block
        return block

    def launch(self, *, open_browser: bool = False, **kwargs: Any) -> Any:
        """Start the server on 127.0.0.1 only.

        Loopback deliberately: in Colab the tab reaches it through the kernel's
        port proxy, so there is no reason to bind a public interface.
        """
        block = self.build()
        kwargs.setdefault("server_name", "127.0.0.1")
        kwargs.setdefault("server_port", self.port)
        kwargs.setdefault("share", False)  # §8.3: share=True is out of scope
        kwargs.setdefault("quiet", True)
        return block.launch(open_browser=open_browser, **kwargs)

    def close(self) -> None:
        self._clear_pending()
        if self.block is not None:
            with contextlib.suppress(Exception):
                self.block.close()
