"""The notebook chat cell — the guaranteed surface (DESIGN §8.1, PRD §7.8).

What this file is allowed to assume
-----------------------------------
`ipywidgets` and `IPython.display` may be absent (CLI, tests, a bare script), so
every import of them is inside a function. When they are missing, `ChatCell` still
constructs and still records the transcript; it just does not draw it. That is why
the interesting logic lives in `render.py` and this module is mostly plumbing.

The one genuinely hard part
---------------------------
Approvals are a *synchronous* callable in the loop (`Approver` returns a bool), but
a notebook cannot block its own cell waiting for a click without freezing the
front end. The mechanism here is a `threading.Event` that a button callback sets:

* the loop thread (which is the cell's thread) blocks on `event.wait()`;
* Colab delivers widget messages on a separate thread, so button callbacks still
  run and can set the event.

A timeout is mandatory. If the browser is gone, a disconnected runtime, or a
message that never arrives, `wait()` returns False and the action is **denied** —
the safe default, consistent with non-interactive mode (DESIGN §9.1).

FR-UX-7 (reconnect-safe)
------------------------
Widget state can be lost when a Colab tab reconnects. So the cell also accepts
plain text commands (`approve`, `deny`, `stop`, `resume`) through `command()`,
which sets the same pending decision. A stuck session is recoverable from the
keyboard.
"""

from __future__ import annotations

import contextlib
import html
import threading
from typing import Any

from .render import Transcript, render_status_line

__all__ = ["ChatCell", "ApprovalTimeout", "DEFAULT_APPROVAL_TIMEOUT_S"]

#: Ten minutes: long enough for a human to read a diff, short enough that an
#: abandoned tab does not hold a session lock until the heartbeat expires.
DEFAULT_APPROVAL_TIMEOUT_S = 600


class ApprovalTimeout(RuntimeError):
    """No decision arrived in time. Treated as a denial by the caller."""


class ChatCell:
    """Renders a coAgent run into a Colab output cell.

    Usage from a cell::

        cell = ChatCell()
        agent = coAgent(repo=".", on_event=cell.on_event, approver=cell.approver,
                        renderer=cell.render_artifact)
        cell.attach(agent)
        cell.chat("add a chunked CSV loader")
    """

    def __init__(
        self,
        *,
        title: str = "",
        approval_timeout_s: float = DEFAULT_APPROVAL_TIMEOUT_S,
        max_rows: int = 200,
        stream: bool = True,
    ) -> None:
        self.transcript = Transcript(max_rows=max_rows)
        self.approval_timeout_s = approval_timeout_s
        self.stream = stream
        self.agent: Any = None
        self.title = title
        #: Set by a button or by `command()`; the loop thread waits on it.
        self._decision: threading.Event = threading.Event()
        self._decision_value = False
        self._pending_request: Any = None
        self._lock = threading.Lock()

        self._handle: Any = None
        self._details_handle: Any = None
        self._decide_event: threading.Event | None = None
        self._buttons: dict[str, Any] = {}
        self._widgets_ok = False
        self._setup_surface()

    # ------------------------------------------------------------- surface

    def _setup_surface(self) -> None:
        """Create the output handles. Failures here are not fatal (headless)."""
        try:
            from IPython.display import display

            self._handle = display({"text/markdown": "coAgent is idle."}, raw=True, display_id=True)
            self._details_handle = display({"text/html": ""}, raw=True, display_id=True)
            self._widgets_ok = True
        except Exception:  # noqa: BLE001 - no IPython: stay silent, keep working
            self._handle = None
            self._details_handle = None
        self._paint()

    @property
    def live(self) -> bool:
        """True when there is something to draw on."""
        return self._handle is not None

    # -------------------------------------------------------------- events

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        """`coAgent(on_event=cell.on_event)`. Never raises into the loop."""
        try:
            self.transcript.handle(kind, data)
            if kind == "task_end":
                self._render_buttons(show=False)
            self._paint()
        except Exception as exc:  # noqa: BLE001 - the UI must never break a run
            # A broken renderer must not take the session with it; the log is the
            # durable record and the loop keeps going.
            self._safe_print(f"[ui] render failed: {exc}")

    def _paint(self) -> None:
        if self._handle is None:
            return
        try:
            status = self._status_text()
            body = self.transcript.render(status_line="")
            if status:
                body = f"```\n{status}\n```\n\n{body}"
            # `display_id` + update is what §8.1 asks for: `clear_output` flickers
            # and scrolls the pane, an in-place update does not.
            self._handle.update({"text/markdown": body}, raw=True)
            self._paint_buttons()
        except Exception:  # noqa: BLE001
            pass

    def _status_text(self) -> str:
        """FR-UX-3: model, context, GPU, session, and budget on every repaint."""
        agent = self.agent
        if agent is None:
            return self.transcript.status
        try:
            return render_status_line(
                model_id=getattr(agent, "model_id", "?"),
                ctx_tokens=agent.cfg.model.max_context,
                session_id=agent.session_id,
                budget=agent.budget.describe(),
                gpu=_gpu_label(),
            )
        except Exception:  # noqa: BLE001
            return self.transcript.status

    # ---------------------------------------------------------- approvals

    def approver(self, request: Any) -> bool:
        """The synchronous `Approver` the loop calls.

        Blocks on a button click or a text command. On timeout it denies: a
        missing human is not consent (DESIGN §9.1, non-interactive default).
        """
        with self._lock:
            self._pending_request = request
            self._decision.clear()
            self._decision_value = False

        self._render_buttons(show=True)
        self._paint()
        arrived = self._decision.wait(timeout=self.approval_timeout_s)

        with self._lock:
            decided = self._decision_value
            self._pending_request = None
        self._render_buttons(show=False)
        self._paint()

        if not arrived:
            self.transcript.handle(
                "info",
                {
                    "notes": (
                        f"No decision for the {request.kind} request within "
                        f"{self.approval_timeout_s:.0f}s, so it was denied. "
                        "Send `approve` to retry the action, or `stop` to end the session."
                    )
                },
            )
            self._paint()
            return False
        return decided

    def command(self, text: str) -> bool | None:
        """Accept a text decision when widgets are gone (FR-UX-7).

        Returns True/False when the command resolved a pending request, or None
        when the text was not a command.
        """
        word = text.strip().lower().split(maxsplit=1)[0] if text.strip() else ""
        if word in {"approve", "yes", "y", "a"}:
            return self._resolve(True)
        if word in {"deny", "no", "n", "reject", "d"}:
            return self._resolve(False)
        if word in {"stop", "cancel"}:
            if self.agent is not None:
                self.agent.cancel()
            return self._resolve(False)
        return None

    def _resolve(self, value: bool) -> bool | None:
        with self._lock:
            if self._pending_request is None:
                return None
            self._decision_value = value
        self._decision.set()
        return value

    # ------------------------------------------------------------ buttons

    def _render_buttons(self, *, show: bool) -> None:
        """The plan/approval controls (FR-UX-2).

        Widgets are created once and shown/hidden, because creating fresh widgets
        on every approval is what makes a Colab output cell grow without bound.
        """
        if not self._widgets_ok:
            return
        if show and not self._buttons:
            self._create_buttons()
        box = self._buttons.get("box")
        if box is None:
            return
        try:
            from IPython.display import display

            if show and getattr(box, "_coagent_shown", False) is False:
                display(box)
                box._coagent_shown = True  # noqa: SLF001 - display bookkeeping
            box.layout.display = "" if show else "none"
        except Exception:  # noqa: BLE001
            pass

    def _create_buttons(self) -> None:
        try:
            import ipywidgets as widgets
        except Exception:  # noqa: BLE001 - no widgets: text commands still work
            self._widgets_ok = False
            return

        approve = widgets.Button(description="Approve", button_style="success")
        deny = widgets.Button(description="Deny", button_style="danger")
        stop = widgets.Button(description="Stop")

        approve.on_click(lambda _b: self._resolve(True))
        deny.on_click(lambda _b: self._resolve(False))
        stop.on_click(self._on_stop)

        self._buttons = {
            "approve": approve,
            "deny": deny,
            "stop": stop,
            "box": widgets.HBox([approve, deny, stop]),
        }

    def _on_stop(self, _button: Any) -> None:
        if self.agent is not None:
            with contextlib.suppress(Exception):
                self.agent.cancel()
        self._resolve(False)

    # ---------------------------------------------------------- artifacts

    def render_artifact(self, kind: str, payload: str, mime: str) -> None:
        """`coAgent(renderer=cell.render_artifact)` — FR-UX-4.

        Renders below the transcript so a figure does not push the log off screen.
        `payload` is base64 for images and raw text otherwise, which is the
        contract `display_artifact` documents.
        """
        if self._details_handle is None:
            return
        try:
            if kind == "image":
                import base64

                from IPython.display import Image, display

                display(Image(data=base64.b64decode(payload)))
                return
            if kind == "html":
                from IPython.display import HTML, display

                display(HTML(payload))
                return
            escaped = html.escape(payload[:4000])
            self._details_handle.update(
                {"text/markdown": f"```\n{escaped}\n```"}, raw=True
            )
        except Exception as exc:  # noqa: BLE001
            self._safe_print(f"[ui] could not render artifact: {exc}")

    # --------------------------------------------------------------- chat

    def attach(self, agent: Any) -> None:
        """Bind the agent so the status line and Stop button have a target."""
        self.agent = agent

    def chat(self, task: str, agent: Any = None) -> Any:
        """Run a task with this cell as the surface."""
        target = agent or self.agent
        if target is None:
            raise RuntimeError(
                "ChatCell.chat needs an agent: pass one, or call attach(agent) first."
            )
        self.attach(target)
        return target.chat(task)

    def _safe_print(self, text: str) -> None:
        try:
            from IPython.display import display

            display({"text/plain": text}, raw=True)
        except Exception:  # noqa: BLE001
            print(text)


def _gpu_label() -> str:
    """`T4 16GB` when a GPU is visible, empty otherwise.

    Deliberately tolerant: this runs on every repaint, and a CUDA query that
    throws must not blank the status line.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        props = torch.cuda.get_device_properties(0)
        gib = props.total_memory / 1024**3
        return f"{props.name.split()[0]} {gib:.0f}GB"
    except Exception:  # noqa: BLE001
        return ""
