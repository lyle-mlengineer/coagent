"""Parse pytest output into what the model actually needs (DESIGN §7.5).

"Only the failing parts are fed back — test names plus the last 40 lines of each
traceback, capped at ~1500 tokens." A 500-line pytest dump is worse than no
output: it evicts the context the model needs to fix the failure.

The parser is the unit under test here: pure text in, structure out, no process
and no git. `run_tests` in `tools/shell.py` is a thin shell over it.

What pytest actually emits (sections, in order)
-----------------------------------------------
````
=========================== FAILURES ============================
____________________________ test_deep _____________________________

    def test_deep():
>       helper(3)

test_sample.py:5:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
x = 3

    def helper(x):
>       raise KeyError("missing")
E       KeyError: 'missing'

test_sample.py:2: KeyError
=========================== short test summary info ===========================
FAILED test_sample.py::test_deep - KeyError: 'missing'
1 failed, 1 passed in 0.08s
````

Two details worth stating because they are easy to get wrong, and I got them
wrong first: a failing *top-level test function* has no `file.py:N: in name`
header at all — the frame footer is `file.py:N: ExceptionType` with an empty
function name — and the actionable part is the `>  ` source pointer plus the
`E  ` error block, not the `def` body around it. Selecting the `E` lines and the
pointer, rather than a fixed number of trailing lines, is what makes the digest
survive a traceback that is mostly library frames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "ParseResult",
    "is_failure_output",
    "parse_pytest_output",
    "summarise_pytest",
]

#: e.g. `FAILED tests/test_x.py::test_it - AssertionError: nope`
_FAILED_LINE = re.compile(r"^(?P<status>FAILED|ERROR)\s+(?P<nodeid>\S+?)(?:\s+-\s+(?P<reason>.*))?$")
#: Section banner naming one failure: `_____ test_name _____`
_FAILURE_HEADER = re.compile(r"^_{3,}\s*(?P<name>.+?)\s*_{3,}$")
#: The `>` source pointer pytest marks the failing line with.
_POINTER = re.compile(r"^>\s*(?P<code>\S.*)$")
#: Our own excepted lines: `E   KeyError: 'x'`
_ERROR_LINE = re.compile(r"^E\s{1,4}(?P<text>.*)$")
#: A frame footer: `test_sample.py:5: ` or `test_sample.py:2: KeyError`
_FRAME_FOOTER = re.compile(r"^(?P<path>[\w./\\-]+\.py):(?P<line>\d+):\s*(?P<exc>.*)$")
#: Separator between traceback frames.
_FRAME_RULE = re.compile(r"^_[\s_]+$")


@dataclass(slots=True)
class ParseResult:
    """Structured pytest output."""

    failed_names: list[str] = field(default_factory=list)
    #: One entry per failing test: pointer lines, error lines, and the final
    #: frame footer (`file.py:12: TypeError`).
    traces: list[str] = field(default_factory=list)
    summary_lines: list[str] = field(default_factory=list)
    pytest_version: str = ""
    duration_s: float = 0.0

    @property
    def failures(self) -> int:
        return len(self.failed_names)

    @property
    def passed(self) -> int:
        for line in self.summary_lines:
            m = re.search(r"(\d+)\s+passed", line)
            if m:
                return int(m.group(1))
        return 0

    def render(self, *, cap: int = 1500) -> str:
        """Assemble the model-facing digest within the §7.5 budget.

        Order matters and is deliberate: the verdict first (it is the answer to
        "did it pass"), then the failing names (the map), then the evidence.
        """
        blocks: list[str] = []
        summary = " ".join(s.strip() for s in self.summary_lines if "short test summary" not in s)
        summary = re.sub(r"[=\s]{3,}", " ", summary).strip()
        if summary:
            blocks.append(summary)
        if self.failed_names:
            blocks.append("failing: " + ", ".join(self.failed_names[:20]))
        return _fit(blocks, self.traces, cap)


def is_failure_output(text: str) -> bool:
    """True when `text` looks like a pytest failure report."""
    return bool(_FAILED_LINE.search(text)) or "short test summary info" in text


def parse_pytest_output(text: str) -> ParseResult:
    """Split pytest output into failed names, per-failure traces, and a summary."""
    lines = text.splitlines()
    result = ParseResult()
    in_failures = False
    current: list[str] | None = None

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if "short test summary info" in stripped:
            in_failures = False
            _flush(current, result)
            current = None
            result.summary_lines.append(stripped)
            continue

        if not in_failures and stripped.startswith("===") and "FAILURES" in stripped:
            in_failures = True
            continue

        if in_failures:
            header = _FAILURE_HEADER.match(stripped)
            if header and "FAILURES" not in stripped:
                _flush(current, result)
                current = [f"--- {header.group('name')}"]
                continue
            if current is not None and not _FRAME_RULE.match(stripped):
                _keep(current, stripped)
            continue

        failed = _FAILED_LINE.match(stripped)
        if failed:
            result.failed_names.append(failed.group("nodeid"))
            continue

        if result.summary_lines or _FAILED_LINE.search(stripped):
            result.summary_lines.append(stripped)

    _flush(current, result)

    version_m = re.search(r"pytest\s+version\s+([\d.]+)", text)
    if version_m:
        result.pytest_version = version_m.group(1)
    duration_m = re.search(r"in\s+([\d.]+)s", text)
    if duration_m:
        result.duration_s = float(duration_m.group(1))
    return result


def _keep(block: list[str], line: str) -> None:
    """Keep the lines that explain a failure, drop the ones that pad it.

    Source context (`def`, blank lines, local-variable echoes) is mostly noise
    at this budget; the pointer and the `E` block are the failure.
    """
    if not line:
        return
    if _POINTER.match(line) or _ERROR_LINE.match(line) or _FRAME_FOOTER.match(line):
        block.append(line)


def _flush(block: list[str] | None, result: ParseResult) -> None:
    if not block:
        return
    header, rest = block[0], block[1:]
    if not rest:
        return
    # Keep at most the last 40 lines per failure (§7.5), and never the middle.
    if len(rest) > 40:
        rest = rest[-40:]
    result.traces.append("\n".join([header, *rest]))


def summarise_pytest(text: str, *, cap: int = 1500) -> str:
    """The one-call entry point: the digest, or the original when it is smaller."""
    if not is_failure_output(text):
        return _fit_text(text, cap)
    parsed = parse_pytest_output(text)
    digest = parsed.render(cap=cap)
    return digest if 0 < len(digest) <= len(text) else _fit_text(text, cap)


def _fit(head: list[str], traces: list[str], cap: int) -> str:
    """Fit the invariant header plus as many whole traces as the cap allows.

    Never truncate a trace mid-line: half a traceback is worse than a whole one
    omitted, because it looks like evidence and is not. Traces are added from the
    *last* backwards, so a tight budget keeps the most recent failure in full.
    """
    header = "\n".join(head)
    if len(header) >= cap:
        return header[:cap]
    room = cap - len(header) - 1
    # Reserve room for the omission note before spending on traces, or the note
    # itself pushes the digest over the cap it is supposed to respect.
    kept: list[str] = []
    for index in range(len(traces), 0, -1):
        trace = traces[index - 1]
        # How many traces would be dropped if we keep this one, including every
        # one already skipped between the end and here.
        omitted_if_kept = index - 1 - len(kept)
        note = (
            f"\n[{omitted_if_kept} more failing trace(s) omitted]" if omitted_if_kept else ""
        )
        cost = len(trace) + 2 + len(note)
        if cost > room:
            break
        kept.append(trace)
        room -= len(trace) + 2
    kept.reverse()
    if not kept and traces:
        # Nothing fit. Saying so is the difference between "no tracebacks" and
        # "tracebacks withheld for budget" — the model must not read the first.
        return header + f"\n[{len(traces)} failing trace(s) omitted for budget]"
    out = header + ("\n" + "\n\n".join(kept) if kept else "")
    omitted = len(traces) - len(kept)
    if omitted:
        out += f"\n[{omitted} more failing trace(s) omitted]"
    return out


def _fit_text(text: str, cap: int) -> str:
    """Fallback for non-pytest output: keep the tail, where the error is."""
    if len(text) <= cap:
        return text
    return "…" + text[-cap:]
