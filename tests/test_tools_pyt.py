"""pytest output parsing (DESIGN §7.5): failing names and a bounded digest.

The parser is pure text in, structure out. These tests use real pytest output
captured from pytest itself rather than hand-written fixtures, because the
failure mode that matters — a top-level test function having no
`file.py:N: in name` header — is exactly the detail a hand-written fixture gets
wrong.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from coagent.tools.pyt import (
    is_failure_output,
    parse_pytest_output,
    summarise_pytest,
)

FIXTURE = textwrap.dedent(
    '''
    def helper(x):
        raise KeyError(f"missing {x}")

    def test_deep():
        helper(3)

    def test_shallow():
        assert 1 == 2

    def test_ok():
        assert True
    '''
)


@pytest.fixture(scope="module")
def pytest_output(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Run pytest for real and capture its output."""
    import subprocess

    root: Path = tmp_path_factory.mktemp("pyt")
    (root / "test_sample.py").write_text(FIXTURE, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_sample.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0, "the fixture must contain failing tests"
    return proc.stdout + proc.stderr


# ------------------------------------------------------------------ detection


def test_detects_failure_output(pytest_output: str) -> None:
    assert is_failure_output(pytest_output)


def test_clean_output_is_not_a_failure() -> None:
    assert not is_failure_output("5 passed in 0.12s")
    assert not is_failure_output("")


# ------------------------------------------------------------------- parsing


def test_finds_every_failing_test_name(pytest_output: str) -> None:
    parsed = parse_pytest_output(pytest_output)
    assert "test_sample.py::test_deep" in parsed.failed_names
    assert "test_sample.py::test_shallow" in parsed.failed_names


def test_does_not_report_the_passing_test(pytest_output: str) -> None:
    parsed = parse_pytest_output(pytest_output)
    assert "test_sample.py::test_ok" not in parsed.failed_names
    assert parsed.failures == 2
    assert parsed.passed == 1


def test_top_level_test_captured_without_a_frame_header(pytest_output: str) -> None:
    """A top-level test has no `in name` header; the footer is the anchor."""
    parsed = parse_pytest_output(pytest_output)
    trace = next(t for t in parsed.traces if "test_deep" in t)
    assert "helper(3)" in trace
    assert "KeyError" in trace


def test_error_line_is_kept(pytest_output: str) -> None:
    parsed = parse_pytest_output(pytest_output)
    trace = next(t for t in parsed.traces if "test_deep" in t)
    assert any(line.startswith("E") and "KeyError" in line for line in trace.splitlines())


def test_summary_lines_capture_the_verdict(pytest_output: str) -> None:
    parsed = parse_pytest_output(pytest_output)
    joined = " ".join(parsed.summary_lines)
    assert "failed" in joined


def test_parses_version_and_duration() -> None:
    text = "pytest version 8.3.2\n1 failed in 0.31s\n"
    parsed = parse_pytest_output(text)
    assert parsed.pytest_version == "8.3.2"
    assert parsed.duration_s == pytest.approx(0.31)


def test_empty_output_is_empty_result() -> None:
    parsed = parse_pytest_output("")
    assert parsed.failed_names == []
    assert parsed.traces == []
    assert parsed.failures == 0


# -------------------------------------------------------------------- digest


def test_digest_names_the_failures(pytest_output: str) -> None:
    digest = summarise_pytest(pytest_output)
    assert "test_deep" in digest
    assert "test_shallow" in digest
    assert len(digest) < len(pytest_output), "the digest must be smaller than the raw dump"


def test_digest_respects_the_cap(pytest_output: str) -> None:
    for cap in (300, 1500):
        assert len(summarise_pytest(pytest_output, cap=cap)) <= cap


def test_digest_never_cuts_a_trace_in_half(pytest_output: str) -> None:
    """Every kept trace ends on a frame footer, never mid-line."""
    digest = summarise_pytest(pytest_output, cap=400)
    for block in digest.split("\n\n")[1:]:
        if block.startswith("---"):
            last = block.strip().splitlines()[-1]
            assert ":" in last or "omitted" in last


def test_omission_is_announced() -> None:
    """When whole traces do not fit, the digest says so rather than truncating one."""
    blocks = []
    for i in range(30):
        blocks.append(f"_____ test_{i} _____\n>   assert {i}\nE   AssertionError: nope\nt.py:{i}: AssertionError")
    text = (
        "=========================== FAILURES ============================\n"
        + "\n".join(blocks)
        + "\n=========================== short test summary info ===========================\n"
        + "\n".join(f"FAILED t.py::test_{i} - AssertionError: nope" for i in range(30))
        + "\n30 failed in 0.1s\n"
    )
    digest = summarise_pytest(text, cap=900)
    assert "more failing trace(s) omitted" in digest
    assert len(digest) <= 900, "the omission note must fit inside the cap, not push past it"


def test_failed_lines_without_traces_keep_the_tail() -> None:
    """`FAILED` lines with no FAILURES section have no traces to keep."""
    many = "\n".join(f"FAILED t.py::test_{i} - AssertionError: nope" for i in range(30))
    digest = summarise_pytest(many, cap=250)
    assert "test_29" in digest, "the last failure is the most recent and must survive"


def test_non_pytest_output_passes_through() -> None:
    text = "ruff check\nAll checks passed!"
    assert summarise_pytest(text) == text


def test_small_non_pytest_output_is_unchanged() -> None:
    assert summarise_pytest("ok") == "ok"


def test_huge_non_pytest_output_keeps_the_tail() -> None:
    """The tail is where the error is, so the head is what gets dropped."""
    text = "x" * 5000 + "THE_ERROR"
    digest = summarise_pytest(text, cap=200)
    assert "THE_ERROR" in digest
    assert len(digest) <= 201  # the leading ellipsis is one character


def test_digest_is_stable_between_calls(pytest_output: str) -> None:
    """Two renders of the same output must not drift: this is fed to a model."""
    assert summarise_pytest(pytest_output) == summarise_pytest(pytest_output)


def test_summarising_a_digest_is_idempotent(pytest_output: str) -> None:
    """The loop re-summarises `run_tests` output; that must not shrink it again."""
    once = summarise_pytest(pytest_output)
    assert summarise_pytest(once, cap=6000) == once
