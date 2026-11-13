"""Gradio surface (DESIGN §8.3): the parts that are logic, not widgets.

Gradio is optional, so everything here either skips cleanly when it is absent or
exercises a method that does not touch a widget. The approval queue is the piece
worth testing: it decides whether a shell command runs, and a bug there is a
security bug, not a cosmetic one.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Gradio is optional and does not install on every supported Python. The logic
# under test is ours (the approval queue, event folding, HTML escaping), so when
# the real package is missing we substitute a stub with the same widget surface.
# The tests then exercise the real GradioApp methods rather than skipping.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from coagent.ui import GradioUnavailable, gradio_available  # noqa: E402

REAL_GRADIO = gradio_available()
if not REAL_GRADIO:
    import _fake_gradio

    sys.modules.setdefault("gradio", _fake_gradio)

pytestmark = pytest.mark.skipif(
    not gradio_available(), reason="no gradio and no usable stub on this interpreter"
)


@dataclass
class _Request:
    kind: str = "install"
    command: str = "pip install requests"
    reason: str = "installing packages needs approval"


def _app(**kwargs: Any):
    from coagent.ui import GradioApp

    return GradioApp(**kwargs)


# ------------------------------------------------------------ guarded import


def test_missing_gradio_is_a_clear_error() -> None:
    """The error must say how to fix it, not just that an import failed."""
    if REAL_GRADIO:
        pytest.skip("real gradio is installed; the guarded path cannot be exercised")
    from coagent.ui import GradioApp

    real = sys.modules.pop("gradio")
    try:
        with pytest.raises(GradioUnavailable) as exc:
            GradioApp()
    finally:
        sys.modules["gradio"] = real
    assert "pip install" in str(exc.value)


# ---------------------------------------------------------------- approvals


def test_approval_blocks_until_approved() -> None:
    app = _app()
    result: list[bool] = []

    def worker() -> None:
        result.append(app.approver(_Request()))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    # Give the thread a moment to block inside approver().
    for _ in range(100):
        if app._pending is not None:
            break
        time.sleep(0.01)
    assert result == [], "approver must not return before a decision"
    app._resolve(True)
    thread.join(timeout=5)
    assert result == [True]


def test_approval_denied_returns_false() -> None:
    app = _app()
    result: list[bool] = []

    def worker() -> None:
        result.append(app.approver(_Request()))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    for _ in range(100):
        if app._pending is not None:
            break
        time.sleep(0.01)
    app._resolve(False)
    thread.join(timeout=5)
    assert result == [False]


def test_approval_times_out_to_deny() -> None:
    """Nobody looking must mean no (DESIGN §9.1), not a hang."""
    app = _app(approval_timeout_s=0.05)
    assert app.approver(_Request()) is False


def test_approval_html_escapes_the_command() -> None:
    """`command` is model output and this is rendered as HTML."""
    from coagent.ui.gradio_app import _PendingApproval

    app = _app()
    app._pending = _PendingApproval(_Request(command="<script>alert(1)</script>"))
    html = app._approval_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_no_approval_html_when_idle() -> None:
    assert _app()._approval_html() == ""


def test_resolve_without_a_pending_request_is_a_noop() -> None:
    """A button clicked after a timeout must not raise or resurrect a decision."""
    app = _app()
    app._resolve(True)  # no pending
    assert app._pending is None


# -------------------------------------------------------------------- state


def test_chat_without_agent_raises_a_useful_error() -> None:
    app = _app()
    with pytest.raises(RuntimeError, match="no agent"):
        app.chat("do something")


def test_events_reach_the_transcript() -> None:
    app = _app()
    app.on_event("task_start", {"task": "add a loader"})
    app.on_event("commit", {"sha": "abcdef1234", "step": 1})
    body = app.transcript.render()
    assert "add a loader" in body
    assert "abcdef1" in body


def test_task_end_clears_a_pending_approval() -> None:
    """A run that ended must not leave the loop waiting on a dead request."""
    from coagent.ui.gradio_app import _PendingApproval

    app = _app()
    app._pending = _PendingApproval(_Request())
    app.on_event("task_end", {"done": True})
    assert app._pending is None


def test_broken_event_does_not_raise_into_the_loop() -> None:
    app = _app()
    app.transcript = _ExplodingTranscript()  # type: ignore[assignment]
    app.on_event("tool", {"tool": "read_file", "ok": True})  # must not raise


class _ExplodingTranscript:
    """Stands in for the transcript to prove the UI cannot break a run."""

    def handle(self, kind: str, data: dict[str, Any]) -> None:
        raise RuntimeError("boom")


# -------------------------------------------------------------------- build


def test_build_is_idempotent() -> None:
    app = _app()
    first = app.build()
    assert app.build() is first, "a second launch must reuse the same Blocks"


def test_launch_binds_loopback_only() -> None:
    """§8.3: reachable through the Colab port proxy, never on a public iface."""
    app = _app(port=7861)
    app.build()
    app.launch()
    kwargs = app.block.launched
    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["server_port"] == 7861
    assert kwargs["share"] is False


def test_port_defaults_to_the_config_value() -> None:
    @dataclass
    class _Ui:
        gradio_port: int = 7999

    @dataclass
    class _Cfg:
        ui: _Ui

    @dataclass
    class _Agent:
        cfg: _Cfg
        model_id: str = "qwen-coder-7b"
        ctx_tokens: int = 8000
        session_id: str = "s1"
        state: Any = None

    app = _app(agent=_Agent(cfg=_Cfg(ui=_Ui())))
    assert app.port == 7999


def test_attach_takes_over_the_hooks() -> None:
    @dataclass
    class _Agent:
        on_event: Any = None
        renderer: Any = None
        approver: Any = None
        model_id: str = "m"
        ctx_tokens: int = 1000
        session_id: str = "s"
        state: Any = None
        cfg: Any = None

    agent = _Agent()
    app = _app()
    app.attach(agent)
    assert agent.on_event == app.on_event
    assert agent.approver == app.approver
    assert agent.renderer == app.render_artifact


def test_submit_with_empty_text_is_a_noop() -> None:
    app = _app()
    body, _status, prompt, _approval = app.submit("   ")
    assert prompt == "", "the prompt must clear so a double click cannot resubmit"
    assert "no agent attached" in body


def test_submit_without_agent_reports_instead_of_raising() -> None:
    app = _app()
    body, *_ = app.submit("do it")
    assert "no agent attached" in body


def test_close_is_safe_twice() -> None:
    app = _app()
    app.build()
    app.close()
    app.close()
