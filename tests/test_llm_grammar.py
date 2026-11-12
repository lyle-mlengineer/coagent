"""Envelope parsing tests — the reliability core (DESIGN §4.3)."""

from __future__ import annotations

import json

import pytest

from coagent.llm.grammar import (
    EnvelopeError,
    extract_json_candidates,
    looks_like_final_answer,
    parse_envelope,
    repair_json,
    strip_fences,
)

TOOLS = {"read_file", "edit_file", "grep", "run_shell"}


# -------------------------------------------------------------- happy paths


def test_clean_fenced_json() -> None:
    text = 'I will read it.\n```json\n{"tool": "read_file", "args": {"path": "a.py"}}\n```'
    env = parse_envelope(text, allowed_tools=TOOLS)
    assert env.tool == "read_file"
    assert env.args == {"path": "a.py"}
    assert not env.repaired


def test_unlabelled_fence() -> None:
    text = '```\n{"tool": "grep", "args": {"pattern": "def"}}\n```'
    assert parse_envelope(text, allowed_tools=TOOLS).tool == "grep"


def test_bare_json_object() -> None:
    text = 'Here is the call: {"tool": "read_file", "args": {"path": "x.py"}} and that is it.'
    env = parse_envelope(text, allowed_tools=TOOLS)
    assert env.tool == "read_file"


def test_json_with_prose_after_braces() -> None:
    text = '{"tool": "read_file", "args": {"path": "a{b}.py"}} trailing }'
    env = parse_envelope(text, allowed_tools=TOOLS)
    assert env.args["path"] == "a{b}.py"


def test_nested_args_are_preserved() -> None:
    text = '```json\n{"tool": "run_shell", "args": {"argv": ["pytest", "-q"], "timeout_s": 60}}\n```'
    env = parse_envelope(text, allowed_tools=TOOLS)
    assert env.args["argv"] == ["pytest", "-q"]
    assert env.args["timeout_s"] == 60


def test_braces_inside_strings_do_not_break_extraction() -> None:
    code = 'print("}")'
    text = '```json\n' + json.dumps({"tool": "run_shell", "args": {"command": code}}) + "\n```"
    env = parse_envelope(text, allowed_tools=TOOLS)
    assert env.args["command"] == code


def test_missing_args_defaults_to_empty() -> None:
    env = parse_envelope('```json\n{"tool": "grep"}\n```', allowed_tools=TOOLS)
    assert env.args == {}


def test_args_null_becomes_empty() -> None:
    env = parse_envelope('```json\n{"tool": "grep", "args": null}\n```', allowed_tools=TOOLS)
    assert env.args == {}


# ------------------------------------------------------------------ repairs


def test_trailing_comma_repaired() -> None:
    text = '```json\n{"tool": "read_file", "args": {"path": "a.py",},}\n```'
    env = parse_envelope(text, allowed_tools=TOOLS)
    assert env.repaired
    assert env.tool == "read_file"


def test_single_quotes_repaired() -> None:
    text = "```json\n{'tool': 'read_file', 'args': {'path': 'a.py'}}\n```"
    env = parse_envelope(text, allowed_tools=TOOLS)
    assert env.repaired
    assert env.args == {"path": "a.py"}


def test_unquoted_keys_repaired() -> None:
    text = '```json\n{tool: "read_file", args: {path: "a.py"}}\n```'
    env = parse_envelope(text, allowed_tools=TOOLS)
    assert env.repaired
    assert env.tool == "read_file"


def test_repair_json_leaves_valid_json_alone() -> None:
    good = '{"tool": "read_file", "args": {"path": "a.py"}}'
    assert repair_json(good) == good


# ----------------------------------------------------------------- failures


def test_empty_response() -> None:
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope("   ")
    assert "empty" in str(exc.value)
    assert "```json" in exc.value.corrective


def test_prose_only_is_corrected() -> None:
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope("I think we should read the file first.")
    assert "could not find a JSON tool call" in exc.value.corrective
    assert "Do not describe" in exc.value.corrective


def test_json_without_tool_key() -> None:
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope('```json\n{"action": "read"}\n```')
    assert "missing the `tool` key" in exc.value.corrective


def test_unknown_tool_is_corrected_with_the_list() -> None:
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope('```json\n{"tool": "delete_repo", "args": {}}\n```', allowed_tools=TOOLS)
    assert "delete_repo" in exc.value.corrective
    assert "edit_file" in exc.value.corrective


def test_args_not_an_object() -> None:
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope('```json\n{"tool": "grep", "args": "pattern"}\n```', allowed_tools=TOOLS)
    assert "must be an object" in exc.value.corrective


def test_tool_not_a_string() -> None:
    with pytest.raises(EnvelopeError, match="not a string"):
        parse_envelope('```json\n{"tool": 42}\n```', allowed_tools=TOOLS)


def test_truncated_json_fails_cleanly() -> None:
    """A model cut off mid-object must not raise JSONDecodeError at the caller."""
    with pytest.raises(EnvelopeError):
        parse_envelope('```json\n{"tool": "edit_file", "args": {"path": "a.py"', allowed_tools=TOOLS)


# ------------------------------------------------------------------ helpers


def test_strip_fences_includes_whole_text() -> None:
    blocks = strip_fences("```json\n{}\n```")
    assert blocks[0] == "{}"
    assert blocks[-1] == "```json\n{}\n```"


def test_extract_candidates_is_ordered_and_unique() -> None:
    text = '```json\n{"tool":"a"}\n```\n{"tool":"b"}'
    candidates = extract_json_candidates(text)
    assert candidates[0] == '{"tool":"a"}'
    assert len(candidates) == len(set(candidates))


def test_final_answer_detection() -> None:
    assert looks_like_final_answer("Task complete. The branch is ready for review.")
    assert looks_like_final_answer("final_answer: 12 tests pass")
    assert not looks_like_final_answer('```json\n{"tool":"read_file","args":{}}\n```')


def test_no_allowed_tools_accepts_any_name() -> None:
    env = parse_envelope('```json\n{"tool": "anything", "args": {}}\n```')
    assert env.tool == "anything"


# --------------------------------------------------- repair / diagnostics


def test_raw_newline_in_string_is_repaired() -> None:
    """A model writing code emits a literal newline; that must still parse."""
    raw = '```json\n{"tool": "edit_file", "args": {"path": "g.py", "replace": "a\nb"}}\n```'
    env = parse_envelope(raw, allowed_tools={"edit_file"})
    assert env.args["replace"] == "a\nb"
    assert env.repaired


def test_raw_tab_in_string_is_repaired() -> None:
    raw = '```json\n{"tool": "edit_file", "args": {"path": "g.py", "replace": "a\tb"}}\n```'
    assert parse_envelope(raw, allowed_tools={"edit_file"}).args["replace"] == "a\tb"


def test_truncated_envelope_says_it_was_cut_off() -> None:
    raw = '```json\n{"tool": "edit_file", "args": {"path": "g.py", "replace": "def hello('
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope(raw, allowed_tools={"edit_file"})
    assert "cut off" in str(exc.value)
    assert "shorter" in exc.value.corrective


def test_malformed_json_is_diagnosed_separately() -> None:
    raw = '```json\n{"tool": "edit_file", "args": {"path": "g.py" "x"}}\n```'
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope(raw, allowed_tools={"edit_file"})
    assert "malformed" in str(exc.value)


def test_prose_is_not_reported_as_malformed() -> None:
    """Prose must not be called malformed: the model never tried to call a tool."""
    with pytest.raises(EnvelopeError) as exc:
        parse_envelope("I will now edit the file.")
    assert "no JSON object" in str(exc.value)
