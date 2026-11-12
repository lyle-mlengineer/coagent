"""Notebook UI tests (DESIGN §8.1, PRD FR-UX-1..5, FR-UX-7).

The render model carries the logic, so most of these test it directly: which
event becomes which row, what a collapsed tool call says, how a diff is counted,
how a failing verify reads. That is the part a user reads, and it is the part a
browser test could never pin down precisely.

The widget layer is tested for the two things that can go wrong without a browser:
approval must block and then resolve, and it must **deny on timeout** rather than
hanging a session forever.
"""

from __future__ import annotations

import threading
import time

import pytest

from coagent.ui.notebook import ChatCell
from coagent.ui.render import (
    MAX_ROW_OUTPUT,
    Row,
    Transcript,
    render_status_line,
    shorten_path,
    summarise_verify,
    tool_summary,
)

# --------------------------------------------------------------- status line


def test_status_line_shows_everything_fr_ux3_asks_for() -> None:
    line = render_status_line(
        model_id="Qwen2.5-Coder-7B",
        ctx_tokens=12288,
        session_id="2026-11-12-a1b2c3",
        budget="step 3/60 · 40 min left",
        gpu="T4 16GB",
    )
    assert "Qwen2.5-Coder-7B" in line
    assert "12k ctx" in line
    assert "T4 16GB" in line
    assert "2026-11-12-a1b2c3" in line
    assert "step 3/60" in line


def test_status_line_is_one_line_without_a_gpu() -> None:
    line = render_status_line(model_id="m", ctx_tokens=8192, session_id="s")
    assert "\n" not in line
    assert "T4" not in line
    assert "8k ctx" in line


# ---------------------------------------------------------------- transcript


def test_plan_event_renders_steps() -> None:
    t = Transcript()
    t.handle("plan", {"title": "Add loader", "steps": ["read a.py", "edit a.py"]})
    body = t.render()
    assert "plan" in body
    assert "[ ] read a.py" in body
    assert "[ ] edit a.py" in body


def test_task_start_shows_the_prompt() -> None:
    t = Transcript()
    t.handle("task_start", {"task": "add a chunked loader"})
    assert "**You:** add a chunked loader" in t.render()


def test_edit_row_counts_added_and_removed_lines() -> None:
    diff = "edited src/x.py at line 3\n--- a/src/x.py\n+++ b/src/x.py\n-old\n+new\n+extra"
    summary = tool_summary("edit_file", diff, ok=True, path="src/x.py")
    assert "+2" in summary
    assert "−1" in summary
    assert "src/x.py" in summary


def test_failed_tool_explains_itself_in_the_collapsed_row() -> None:
    summary = tool_summary("edit_file", "the search block does not appear in x.py", ok=False)
    assert "failed" in summary
    assert "search block" in summary


def test_read_row_names_the_path_and_size() -> None:
    summary = tool_summary("read_file", "1\ta\n2\tb", ok=True, path="src/deep/module.py")
    assert "src/deep/module.py" in summary
    assert "2 line(s)" in summary


def test_shell_row_reports_the_status_line() -> None:
    summary = tool_summary("run_shell", "[exit 1, 0.4s]\nfailed", ok=False)
    assert "run_shell" in summary


def test_verify_failure_is_a_bad_row() -> None:
    t = Transcript()
    t.handle("verify", {"ok": False, "summary": "$ pytest -q\n1 failed, 12 passed"})
    assert t.rows[-1].severity == "bad"
    assert "failed" in t.rows[-1].summary
    assert "1 failed, 12 passed" in t.rows[-1].summary


def test_verify_success_summary_uses_the_command() -> None:
    t = Transcript()
    t.handle("verify", {"ok": True, "summary": "$ pytest -q\n12 passed in 2.1s"})
    assert t.rows[-1].severity == "neutral"
    assert "passed" in t.rows[-1].summary


def test_commit_row_is_good_and_names_the_sha() -> None:
    t = Transcript()
    t.handle("commit", {"sha": "a1b2c3d4e5", "step": 2})
    row = t.rows[-1]
    assert row.severity == "good"
    assert "a1b2c3d4" in row.summary
    assert "step 2" in row.summary


def test_tool_rows_are_collapsible_and_truncated() -> None:
    t = Transcript()
    t.handle("tool", {"tool": "read_file", "ok": True, "content": "x" * 5000, "path": "a.py"})
    row = t.rows[-1]
    assert row.collapsible
    assert len(row.body) <= MAX_ROW_OUTPUT


def test_rendered_row_uses_details_and_escapes_html() -> None:
    row = Row("tool", "**read** <script>", body="body")
    html = row.render_markdown()
    assert html.startswith("<details>")
    assert "&lt;script&gt;" in html


def test_a_row_with_no_body_is_a_bare_line_without_nested_emphasis() -> None:
    row = Row("commit", "**committed** `abc`")
    rendered = row.render_markdown()
    assert "<details>" not in rendered
    assert rendered == "✓ **committed** `abc`"
    assert "****" not in rendered  # no doubled emphasis from re-wrapping


def test_unknown_events_are_kept_not_dropped() -> None:
    """A new loop event should appear the day it is added."""
    t = Transcript()
    t.handle("brand_new_event", {"detail": "something happened"})
    assert any("brand_new_event" in row.summary for row in t.rows)
    assert "something happened" in t.rows[-1].body


def test_diagnostic_events_carry_severity() -> None:
    t = Transcript()
    t.handle("denied", {"command": "pip install x"})
    assert t.rows[-1].severity == "bad"
    t.handle("repo_map_degraded", {"error": "no parser"})
    assert t.rows[-1].severity == "warn"


def test_transcript_is_bounded_but_keeps_the_newest() -> None:
    t = Transcript(max_rows=5)
    for i in range(20):
        t.handle("commit", {"sha": f"{i:08d}", "step": i})
    assert len(t.rows) == 5
    assert "00000019" in t.rows[-1].summary


def test_artifacts_are_listed_once() -> None:
    t = Transcript()
    t.note_artifacts(["artifacts/run/a.png"])
    t.note_artifacts(["artifacts/run/a.png"])
    assert t.artifacts == ["artifacts/run/a.png"]
    assert sum(1 for r in t.rows if r.kind == "artifact") == 2


def test_empty_transcript_says_idle() -> None:
    assert Transcript().render() == "coAgent is idle."


def test_render_is_stable_between_identical_calls() -> None:
    t = Transcript()
    t.handle("task_start", {"task": "x"})
    assert t.render(status_line="s") == t.render(status_line="s")


# ------------------------------------------------------------------ helpers


def test_shorten_path_keeps_both_ends() -> None:
    long = "a/" * 40 + "module.py"
    out = shorten_path(long, limit=30)
    assert len(out) <= 31
    assert out.startswith("a/")
    assert out.endswith("module.py")


def test_summarise_verify_prefers_the_failure_line() -> None:
    assert "AssertionError" in summarise_verify("$ pytest\nE AssertionError: nope\n1 failed")


def test_summarise_verify_falls_back_to_the_command() -> None:
    assert summarise_verify("$ ruff check .\nAll checks passed") == "$ ruff check ."


def test_summarise_verify_with_no_output() -> None:
    assert summarise_verify("") == "no output"


# -------------------------------------------------------------- the widget layer


class _FakeAgent:
    def __init__(self) -> None:
        self.cancelled = False
        self.session_id = "s1"
        self.model_id = "fake"

        class _Cfg:
            class model:  # noqa: N801 - matches the real nesting
                max_context = 12288

        self.cfg = _Cfg()

        class _Budget:
            def describe(self) -> str:
                return "step 1/60"

        self.budget = _Budget()

    def cancel(self) -> None:
        self.cancelled = True

    def chat(self, task: str) -> str:
        self.task = task
        return "ran"


def test_cell_is_constructible_without_ipython(monkeypatch) -> None:
    """No display available must not be fatal: the transcript still records."""
    import coagent.ui.notebook as module

    monkeypatch.setattr(module.ChatCell, "_setup_surface", lambda self: None)
    cell = ChatCell()
    assert cell.live is False
    cell.on_event("task_start", {"task": "hello"})
    assert "hello" in cell.transcript.render()


def test_approver_denies_on_timeout() -> None:
    """A missing human is not consent (DESIGN §9.1)."""
    cell = ChatCell(approval_timeout_s=0.05)

    class _Request:
        kind = "install"
        command = "pip install x"
        reason = "installing packages needs approval"

    start = time.monotonic()
    assert cell.approver(_Request()) is False
    assert time.monotonic() - start < 5


def test_approved_by_a_button_sets_the_decision() -> None:
    cell = ChatCell(approval_timeout_s=5)

    class _Request:
        kind = "install"
        command = "pip install x"
        reason = "r"

    result: dict[str, bool] = {}

    def approve_later() -> None:
        time.sleep(0.05)
        cell._resolve(True)

    thread = threading.Thread(target=approve_later)
    thread.start()
    result["value"] = cell.approver(_Request())
    thread.join()
    assert result["value"] is True


def test_text_command_resolves_a_pending_approval() -> None:
    """FR-UX-7: a lost widget must not strand the session."""
    cell = ChatCell(approval_timeout_s=5)

    class _Request:
        kind = "network"
        command = "curl x"
        reason = "r"

    outcome: dict[str, bool] = {}

    def command_later() -> None:
        time.sleep(0.05)
        # The decision is pending, so `approve` must apply to it.
        while cell._pending_request is None:
            time.sleep(0.01)
        outcome["resolved"] = cell.command("approve") is True

    thread = threading.Thread(target=command_later)
    thread.start()
    allowed = cell.approver(_Request())
    thread.join()
    assert allowed is True
    assert outcome.get("resolved") is True


def test_deny_command_blocks_the_action() -> None:
    cell = ChatCell(approval_timeout_s=5)

    class _Request:
        kind = "install"
        command = "pip install x"
        reason = "r"

    def deny_later() -> None:
        time.sleep(0.05)
        while cell._pending_request is None:
            time.sleep(0.01)
        cell.command("deny")

    thread = threading.Thread(target=deny_later)
    thread.start()
    assert cell.approver(_Request()) is False
    thread.join()


def test_stop_command_cancels_the_agent() -> None:
    cell = ChatCell(approval_timeout_s=5)
    agent = _FakeAgent()
    cell.attach(agent)
    assert cell.command("stop") is None  # nothing pending: not an error
    assert agent.cancelled is True


def test_non_command_text_is_not_a_decision() -> None:
    cell = ChatCell()
    assert cell.command("what does this do?") is None


def test_chat_requires_an_agent() -> None:
    with pytest.raises(RuntimeError) as err:
        ChatCell().chat("do something")
    assert "attach" in str(err.value)


def test_chat_runs_through_the_attached_agent() -> None:
    cell = ChatCell()
    agent = _FakeAgent()
    cell.attach(agent)
    assert cell.chat("go") == "ran"
    assert agent.task == "go"


def test_a_broken_event_handler_does_not_escape() -> None:
    """The UI must never take the session with it."""

    class _Exploding(Transcript):
        def handle(self, kind, data):
            raise RuntimeError("render bug")

    cell = ChatCell()
    cell.transcript = _Exploding()
    cell.on_event("commit", {"sha": "abc"})  # must not raise
