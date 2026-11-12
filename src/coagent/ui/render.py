"""Transcript rendering: the text the chat cell shows, built as data (§8.1).

Why this is separate from the widgets
-------------------------------------
`ipywidgets` cannot be exercised in a test, and the interesting decisions are not
widget decisions: which events become rows, how a tool call reads when collapsed,
how a diff is summarised, when the plan is worth re-rendering. All of that is a
pure function of the event stream, so it lives here and is tested directly. The
widget layer in `notebook.py` is then thin enough to read in one sitting.

The rendering targets §8.1's layout::

    coAgent · Qwen2.5-Coder-7B · 12k ctx · session a1b2c3
    ────────────────────────────────────────────────────────
    You: add a chunked CSV loader and test dtype preservation

    ▸ plan  4 steps, est. 6 min                          [approve] [edit]
    ▸ step 1  read src/loader.py, src/frame.py                 1.2s
    ▸ step 2  edit src/loader.py  +18 −3                       0.4s
    ▸ step 3  pytest -q  → 1 failed                            2.1s
    ✓ committed a1b2c3d · artifacts none

Tool rows are *collapsed* `<details>` elements: one line when closed, args and
truncated output when opened. That is what keeps a 40-step run readable on a
phone-width Colab pane without hiding the evidence.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Row",
    "Transcript",
    "MODE_GLYPH",
    "render_status_line",
    "tool_summary",
    "shorten_path",
    "summarise_verify",
]

#: One glyph per event kind, so a run scans vertically.
MODE_GLYPH: dict[str, str] = {
    "plan": "▸",
    "plan_rejected": "✗",
    "tool": "▸",
    "verify": "▸",
    "commit": "✓",
    "sync": "⇡",
    "artifact": "◈",
    "error": "✗",
    "info": "·",
}

#: An event kind's severity, for colouring and for "is this a stop?".
_SEVERITY = {
    "commit": "good",
    "sync": "good",
    "artifact": "good",
    "verify": "neutral",
    "tool": "neutral",
    "error": "bad",
    "plan_rejected": "bad",
    "denied": "bad",
    "commit_failed": "bad",
    "envelope_error": "warn",
    "replan": "warn",
    "truncated": "warn",
    "repo_map_degraded": "warn",
    "retrieval_degraded": "warn",
    "artifact_manifest_degraded": "warn",
    "final_checkpoint_failed": "warn",
    "verify_failed_after_complete": "warn",
}

#: Long output kept in a collapsed block. The full text is in the session log;
#: this is the evidence a human needs, not the whole stream (§5.1).
MAX_ROW_OUTPUT = 1600


@dataclass
class Row:
    """One line of the transcript, plus the details hidden behind it."""

    kind: str
    summary: str
    body: str = ""
    severity: str = "neutral"
    #: Seconds, when the event knows how long it took.
    duration_s: float | None = None
    #: Extra metadata for a renderer that wants to be smarter than text.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def glyph(self) -> str:
        return MODE_GLYPH.get(self.kind, "·")

    @property
    def collapsible(self) -> bool:
        return bool(self.body.strip())

    def headline(self) -> str:
        """The one line shown when the row is collapsed."""
        parts = [f"{self.glyph} {self.summary}"]
        if self.duration_s is not None:
            parts.append(f"{self.duration_s:.1f}s")
        return "  ".join(parts)

    def render_markdown(self) -> str:
        line = self.headline()
        if not self.collapsible:
            # No nested emphasis: the summary already carries its own markers, and
            # wrapping it again renders as literal `**` in a notebook.
            return line if "**" in line else f"`{line}`"
        closed = "<code>" + html.escape(line) + "</code>"
        return f"<details><summary>{closed}</summary>\n\n```\n{self.body}\n```\n\n</details>"


class Transcript:
    """The chat cell's whole visible state, as data.

    Events arrive from `coAgent.on_event`; each one either appends a row or
    updates the plan/status. `render()` is called far more often than events
    arrive (a widget update per streamed chunk), so it must stay cheap and stable
    — no timestamps, no counters that drift between identical renders.
    """

    def __init__(self, *, max_rows: int = 200) -> None:
        self.rows: list[Row] = []
        self.max_rows = max_rows
        self.task = ""
        self.plan_title = ""
        self.plan_steps: list[str] = []
        self.plan_decided = False
        self.artifacts: list[str] = []
        self.final = ""
        self.status = ""

    # ------------------------------------------------------------- events

    def handle(self, kind: str, data: dict[str, Any]) -> None:
        """Fold one loop event into the transcript."""
        if kind == "task_start":
            self.task = str(data.get("task", ""))
            self.status = "running"
            return
        if kind == "task_end":
            self.status = "done" if data.get("done") else "stopped"
            return
        if kind == "plan":
            self.plan_title = str(data.get("title", ""))
            self.plan_steps = [str(s) for s in data.get("steps", [])]
            steps = ", ".join(f"`{s}`" for s in self.plan_steps[:4])
            self._add(
                Row(
                    "plan",
                    f"**plan** {len(self.plan_steps)} steps — {self.plan_title}",
                    body="\n".join(f"[ ] {s}" for s in self.plan_steps) or steps,
                )
            )
            return
        if kind == "plan_rejected":
            self.plan_decided = True
            self._add(Row("plan_rejected", "plan rejected — nothing was written", severity="bad"))
            return
        if kind == "tool":
            self._add(self._tool_row(data))
            return
        if kind == "verify":
            ok = bool(data.get("ok"))
            summary = summarise_verify(str(data.get("summary", "")))
            self._add(
                Row(
                    "verify",
                    f"**verify** {'passed' if ok else 'failed'} — {summary}",
                    body=str(data.get("summary", ""))[:MAX_ROW_OUTPUT],
                    severity="neutral" if ok else "bad",
                )
            )
            return
        if kind == "commit":
            self._add(
                Row(
                    "commit",
                    f"**committed** `{str(data.get('sha', ''))[:8]}` · step {data.get('step')}",
                    severity="good",
                )
            )
            return
        if kind == "sync":
            ok = bool(data.get("ok"))
            self._add(
                Row(
                    "sync",
                    f"{'**checkpoint** ' if ok else '**checkpoint warning** '}"
                    f"{data.get('message', '')}",
                    severity="good" if ok else "warn",
                )
            )
            return
        if kind in {
            "denied",
            "commit_failed",
            "plan_rejected",
            "envelope_error",
            "replan",
            "truncated",
            "repo_map_degraded",
            "retrieval_degraded",
            "artifact_manifest_degraded",
            "final_checkpoint_failed",
            "verify_failed_after_complete",
        }:
            self._add(self._diagnostic_row(kind, data))
            return
        if kind in {"survey", "resume"}:
            notes = str(data.get("notes", "")).strip()
            self._add(Row("info", f"**{kind}**", body=notes[:MAX_ROW_OUTPUT]))
            return
        # Unknown events are kept, not dropped: a new loop event should show up
        # in the transcript the day it is added, not the day this file learns of it.
        self._add(Row("info", f"**{kind}**", body=_format_data(data)))

    def _tool_row(self, data: dict[str, Any]) -> Row:
        tool = str(data.get("tool", "?"))
        ok = bool(data.get("ok"))
        body = str(data.get("content", ""))
        summary = tool_summary(tool, body, ok=ok, path=str(data.get("path", "")))
        return Row(
            "tool" if ok else "error",
            summary,
            body=body[:MAX_ROW_OUTPUT],
            severity="neutral" if ok else "bad",
            meta={"tool": tool, "ok": ok},
        )

    def _diagnostic_row(self, kind: str, data: dict[str, Any]) -> Row:
        label = kind.replace("_", " ")
        detail = str(data.get("error") or data.get("corrective") or data.get("command") or "")
        severity = _SEVERITY.get(kind, "warn")
        summary = f"**{label}**" + (f" — {detail[:120]}" if detail else "")
        return Row(kind, summary, body=_format_data(data), severity=severity)

    def _add(self, row: Row) -> None:
        self.rows.append(row)
        if len(self.rows) > self.max_rows:
            # Old rows are dropped from the *view*, never from the session log.
            self.rows = self.rows[-self.max_rows :]

    # ------------------------------------------------------------ artifacts

    def note_artifacts(self, paths: list[str]) -> None:
        for path in paths:
            if path not in self.artifacts:
                self.artifacts.append(path)
        if paths:
            self._add(
                Row("artifact", "**artifacts** " + ", ".join(f"`{p}`" for p in paths), severity="good")
            )

    # -------------------------------------------------------------- render

    def render(self, *, status_line: str | None = None, max_rows: int | None = None) -> str:
        """The full markdown body for the chat cell."""
        head = status_line if status_line is not None else self.status
        blocks: list[str] = []
        if head:
            blocks.append(f"```\n{head}\n```")
        if self.task:
            blocks.append(f"**You:** {self.task}")
        rows = self.rows[-max_rows:] if max_rows else self.rows
        blocks.extend(row.render_markdown() for row in rows)
        if self.final:
            blocks.append(self.final)
        return "\n\n".join(blocks) if blocks else "coAgent is idle."

    def render_plan(self) -> str:
        """The plan block alone, for a widget that updates in place."""
        if not self.plan_steps:
            return ""
        lines = [f"**{self.plan_title}**"] if self.plan_title else []
        lines.extend(f"- [ ] {step}" for step in self.plan_steps)
        return "\n".join(lines)


# ------------------------------------------------------------------ helpers


def render_status_line(
    *,
    model_id: str,
    ctx_tokens: int,
    session_id: str,
    budget: str = "",
    gpu: str = "",
) -> str:
    """FR-UX-3: model, context, GPU, session, and budget, always visible.

    Deliberately one line: it sits at the top of every update, so a second line
    would cost vertical space in a Colab pane on every repaint.
    """
    parts = [f"coAgent · {model_id}", f"{ctx_tokens // 1000}k ctx"]
    if gpu:
        parts.append(gpu)
    parts.append(f"session {session_id}")
    if budget:
        parts.append(budget)
    return " · ".join(parts)


_DIFF_STAT = re.compile(r"^\+([0-9]+) [−-]([0-9]+)$")


def tool_summary(tool: str, content: str, *, ok: bool, path: str = "") -> str:
    """One line describing a tool call, the way §8.1 shows it.

    The point of collapsing is that the headline alone is informative: which file
    was edited and by how much, which command ran and whether it passed.
    """
    if not ok:
        first = _first_meaningful_line(content)
        return f"**{tool}** failed — {first}" if first else f"**{tool}** failed"

    if tool in {"edit_file", "edit_notebook", "write_file"}:
        added = content.count("\n+")
        removed = content.count("\n-")
        # `--- a/...` and `+++ b/...` are headers, not content.
        added = max(0, added - content.count("\n+++"))
        removed = max(0, removed - content.count("\n---"))
        label = _short_target(content) or path or tool
        return f"**{tool}** {label}  +{added} −{removed}" if added or removed else f"**{tool}** {label}"

    if tool in {"read_file", "read_notebook", "list_dir"}:
        # The path is the useful fact; the line count tells you how much came back.
        lines = sum(1 for line in content.splitlines() if line.strip())
        if path:
            return f"**{tool}** {shorten_path(path)} · {lines} line(s)"
        return f"**{tool}** {lines} line(s)"

    if tool in {"run_shell", "run_python", "exec_notebook"}:
        status = _first_match(content, r"^\[(.*)\]$") or ""
        verdict = "ok" if ok else "failed"
        detail = shorten_path(status) if status else verdict
        return f"**{tool}** {detail}" if detail else f"**{tool}** {verdict}"

    if tool in {"save_artifact", "display_artifact"}:
        line = _first_meaningful_line(content)
        return f"**{tool}** {line}" if line else f"**{tool}**"

    if tool in {"grep"}:
        count = _first_match(content, r"^no matches")
        return f"**{tool}** {count}" if count else f"**{tool}** {_count_hits(content)} match(es)"

    first = _first_meaningful_line(content)
    return f"**{tool}** {first[:80]}" if first else f"**{tool}**"


def summarise_verify(summary: str) -> str:
    """`pytest -q → 12 passed`, or the first failure line.

    A verify summary is a blob of `$ cmd` + output. The transcript wants the
    verdict, not the blob; the blob is in the collapsed body.
    """
    for line in summary.splitlines():
        lowered = line.lower()
        if any(word in lowered for word in ("failed", "error", "traceback")):
            return line.strip()[:100]
    for line in summary.splitlines():
        if line.strip().startswith("$"):
            return line.strip()[:100]
    for line in reversed(summary.splitlines()):
        if line.strip():
            return line.strip()[:100]
    return "no output"


def shorten_path(text: str, limit: int = 60) -> str:
    """Middle-ellipsis a path so both the root and the filename survive."""
    text = text.strip()
    if len(text) <= limit:
        return text
    keep = max(10, (limit - 1) // 2)
    return f"{text[:keep]}…{text[-keep:]}"


def _first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("[", "$", "```")):
            return stripped[:120]
    return ""


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(0).strip() if match else ""


def _short_target(content: str) -> str:
    match = re.search(r"^--- [ab]/(.+)$", content, re.MULTILINE)
    return shorten_path(match.group(1)) if match else ""


def _count_hits(content: str) -> int:
    if "matches=0" in content:
        return 0
    return sum(1 for line in content.splitlines() if re.match(r"^.+:\d+: ", line))


def _format_data(data: dict[str, Any]) -> str:
    if not data:
        return ""
    lines = []
    for key, value in data.items():
        text = str(value)
        if len(text) > 400:
            text = text[:397] + "…"
        lines.append(f"{key}: {text}")
    return "\n".join(lines)
