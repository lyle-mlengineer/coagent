"""Tolerant extraction of the tool-call envelope (DESIGN §4.3).

A 7B model emits malformed JSON perhaps 3–8% of the time, and a malformed
envelope costs a full turn. This module is the reliability core: it never
raises on bad model output, always returns a diagnosis the loop can feed back.

Escalation ladder, cheapest first:

1. Exactly one ```` ```json ```` fence — the contract we ask for.
2. Any fence containing `{...}` — models sometimes label it ```` ``` ````.
3. A bare top-level `{...}` object (no fence at all).
4. JSON5-ish repair: single quotes, trailing commas, unquoted keys.
5. Failure with a **corrective message** naming what was wrong, so the loop's
   single re-prompt is targeted rather than "try again".

Only `{"tool": ..., "args": {...}}` is accepted. An invented tool name is a
schema error the registry catches, not a parse error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Envelope",
    "EnvelopeError",
    "parse_envelope",
    "extract_json_candidates",
    "repair_json",
    "strip_fences",
    "looks_like_final_answer",
]

_FENCE_RE = re.compile(r"```(?:json|json5|JSON)?\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)")
_SINGLE_QUOTE_RE = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")


class EnvelopeError(ValueError):
    """No usable envelope. `corrective` is fed back to the model."""

    def __init__(self, message: str, *, corrective: str, raw: str = "") -> None:
        super().__init__(message)
        self.corrective = corrective
        self.raw = raw


@dataclass(slots=True)
class Envelope:
    tool: str
    args: dict[str, Any]
    raw: str = ""
    repaired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args}


def strip_fences(text: str) -> list[str]:
    """Every fenced block, plus the whole text as a last candidate."""
    blocks = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    blocks.append(text.strip())
    return blocks


def _balanced_objects(text: str) -> list[str]:
    """Extract balanced `{...}` runs, ignoring braces inside strings.

    Needed because a model often writes prose around a bare object:
    `Here is the call: {"tool": ...} and I will now...`
    """
    found: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                found.append(text[start : i + 1])
                start = -1
    return found


def extract_json_candidates(text: str) -> list[str]:
    """Ordered candidates, most likely first, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for block in strip_fences(text):
        for candidate in (block, *_balanced_objects(block)):
            candidate = candidate.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def repair_json(text: str) -> str:
    """Best-effort JSON5-ish repair for the two most common model slips."""
    fixed = text
    # Single-quoted strings → double-quoted, without touching apostrophes in prose.
    fixed = _SINGLE_QUOTE_RE.sub(lambda m: '"' + m.group(1).replace('"', '\\"') + '"', fixed)
    # Unquoted keys: {tool: "read_file"} → {"tool": "read_file"}
    fixed = _UNQUOTED_KEY_RE.sub(lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', fixed)
    # Trailing commas before a closing brace/bracket.
    fixed = _TRAILING_COMMA_RE.sub(r"\1", fixed)
    return fixed


def _try_load(candidate: str) -> tuple[dict[str, Any] | None, bool]:
    """Return (object, repaired?) for one candidate."""
    for attempt, payload in ((False, candidate), (True, repair_json(candidate))):
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value, attempt
    return None, False


def parse_envelope(text: str, *, allowed_tools: set[str] | None = None) -> Envelope:
    """Extract `{"tool": ..., "args": {...}}` from model output.

    Raises `EnvelopeError` with a `corrective` message — never a bare
    `JSONDecodeError` — because the loop feeds that text straight back.
    """
    if not text or not text.strip():
        raise EnvelopeError(
            "the response was empty",
            corrective=(
                "Your last message was empty. Reply with exactly one fenced JSON block:\n"
                '```json\n{"tool": "<tool name>", "args": {...}}\n```'
            ),
        )

    saw_json = False
    for candidate in extract_json_candidates(text):
        value, repaired = _try_load(candidate)
        if value is None:
            continue
        saw_json = True
        if "tool" not in value:
            continue
        tool = value.get("tool")
        if not isinstance(tool, str):
            raise EnvelopeError(
                "`tool` is not a string",
                corrective=f'`tool` must be a string tool name, got {type(tool).__name__}.',
                raw=text,
            )
        if allowed_tools is not None and tool not in allowed_tools:
            raise EnvelopeError(
                f"unknown tool {tool!r}",
                corrective=(
                    f"There is no tool named {tool!r}. Choose one of: "
                    f"{', '.join(sorted(allowed_tools))}."
                ),
                raw=text,
            )
        args = value.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise EnvelopeError(
                "`args` is not an object",
                corrective=(
                    f'`args` must be an object such as {{"path": "src/a.py"}}, got '
                    f"{type(args).__name__}."
                ),
                raw=text,
            )
        return Envelope(tool=tool, args=args, raw=text, repaired=repaired)

    if saw_json:
        raise EnvelopeError(
            "JSON found but no `tool` key",
            corrective=(
                "Your JSON block is missing the `tool` key. Reply with exactly:\n"
                '```json\n{"tool": "<tool name>", "args": {...}}\n```'
            ),
            raw=text,
        )
    raise EnvelopeError(
        "no JSON object in the response",
        corrective=(
            "I could not find a JSON tool call in your last message. Do not describe "
            "the action in prose; reply with exactly one fenced JSON block:\n"
            '```json\n{"tool": "<tool name>", "args": {...}}\n```'
        ),
        raw=text,
    )


_FINAL_MARKERS = ("final_answer", "task_complete", "done")


def looks_like_final_answer(text: str) -> bool:
    """True when the model is answering in prose rather than calling a tool.

    Used by the loop to decide between "prose: treat as the report" and "re-prompt
    for an envelope" — never to guess at an action.
    """
    lowered = text.lower()
    if any(marker in lowered for marker in _FINAL_MARKERS):
        return True
    return not any("{" in candidate for candidate in extract_json_candidates(text))
