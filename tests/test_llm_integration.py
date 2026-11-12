"""Integration: model output -> envelope -> tool registry -> real file change.

This is the seam the loop sits on. If this test passes, the loop only has to
add prompting, phases, and budgets — it does not have to make these pieces fit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coagent.config import Config
from coagent.llm import (
    FakeBackend,
    GenerationConfig,
    Message,
    TokenBudget,
    looks_like_final_answer,
    parse_envelope,
)
from coagent.llm.grammar import EnvelopeError
from coagent.tools import Phase, build_registry
from coagent.tools.base import ToolError


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "loader.py").write_text('def load(path):\n    return "todo"\n', encoding="utf-8")
    return root


@pytest.fixture()
def act(repo: Path):
    return build_registry(repo, Config(), phase=Phase.ACT)


def _turn(backend: FakeBackend, reg, *, tools: set[str], task: str = "fix the loader"):
    messages = [Message("system", "you are coagent"), Message("user", task)]
    budget = TokenBudget(max_context=12288, count=backend.count_tokens)
    budget.check(messages, max_new_tokens=1024)
    result = backend.generate(messages, GenerationConfig(max_new_tokens=1024))
    envelope = parse_envelope(result.text, allowed_tools=tools)
    return envelope, reg.call(envelope.tool, envelope.args), result


def test_read_then_edit_round_trip(repo: Path, act) -> None:
    backend = FakeBackend(
        [
            'Looking first.\n```json\n{"tool": "read_file", "args": {"path": "src/loader.py"}}\n```',
            "```json\n"
            '{"tool": "edit_file", "args": {"path": "src/loader.py", '
            '"search": "    return \\"todo\\"", '
            '"replace": "    return Frame.from_rows(read(path))"}}\n'
            "```",
        ]
    )
    tools = set(act.names())

    env1, out1, _ = _turn(backend, act, tools=tools)
    assert env1.tool == "read_file"
    assert out1.ok

    env2, out2, _ = _turn(backend, act, tools=tools)
    assert env2.tool == "edit_file"
    assert out2.ok
    assert "Frame.from_rows" in (repo / "src" / "loader.py").read_text()


def test_edit_without_read_is_refused(act) -> None:
    """The model can only edit after reading; enforced by the tool, not the prompt."""
    backend = FakeBackend(
        [
            '```json\n{"tool": "edit_file", "args": {"path": "src/loader.py", '
            '"search": "todo", "replace": "done"}}\n```',
        ]
    )
    result = backend.generate([Message("user", "go")])
    env = parse_envelope(result.text, allowed_tools=set(act.names()))
    with pytest.raises(ToolError, match="has not been read"):
        act.call(env.tool, env.args)


def test_unknown_tool_from_model_becomes_a_corrective_message(act) -> None:
    backend = FakeBackend(['```json\n{"tool": "rm_rf", "args": {}}\n```'])
    messages = [Message("user", "go")]
    result = backend.generate(messages)
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope(result.text, allowed_tools=set(act.names()))
    assert "rm_rf" in exc.value.corrective
    assert "read_file" in exc.value.corrective


def test_malformed_then_recovered(act) -> None:
    """The ladder: a bad first reply, a good second one — one re-prompt."""
    backend = FakeBackend(
        [
            "I will read the file now.",  # prose only -> corrective
            '```json\n{"tool": "read_file", "args": {"path": "src/loader.py"}}\n```',
        ]
    )
    messages = [Message("user", "go")]
    first = backend.generate(messages)
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope(first.text, allowed_tools=set(act.names()))

    # The loop appends the corrective text verbatim and retries once.
    messages.append(Message("user", exc.value.corrective))
    second = backend.generate(messages)
    env = parse_envelope(second.text, allowed_tools=set(act.names()))
    assert env.tool == "read_file"
    assert backend.calls[1]["messages"][-1]["content"] == exc.value.corrective


def test_final_prose_answer_is_recognised(act) -> None:
    backend = FakeBackend(["All done: 12 tests pass on branch coagent/2026-11-12-a1b2c3."])
    result = backend.generate([Message("user", "go")])
    assert looks_like_final_answer(result.text)


def test_budget_refuses_before_generation() -> None:
    from coagent.llm.base import BudgetError

    backend = FakeBackend()
    budget = TokenBudget(max_context=50, count=lambda t: len(t), reserve_output=40)
    messages = [Message("user", "x" * 500)]
    with pytest.raises(BudgetError, match="was not sent"):
        budget.check(messages, max_new_tokens=30)
    assert backend.calls == []  # nothing was generated
