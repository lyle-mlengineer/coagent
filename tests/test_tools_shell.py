"""Execution tool tests: scrubbed env, timeout kill, output cap (PRD FR-TOL-3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from coagent.config import Config
from coagent.tools import Phase, build_registry
from coagent.tools.base import ToolError
from coagent.tools.shell import ExecConfig, run_subprocess, scrubbed_env


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mod.py").write_text("VALUE = 42\n", encoding="utf-8")
    return root


@pytest.fixture()
def act(repo: Path):
    return build_registry(repo, Config(), phase=Phase.ACT)


# ---------------------------------------------------------------- env scrubbing


def test_scrubbed_env_removes_secrets(monkeypatch, repo: Path) -> None:
    monkeypatch.setenv("GH_TOKEN", "ghp_1234567890ABCDEFGHIJ")
    monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnop")
    monkeypatch.setenv("MY_API_KEY", "sk-abcdefghijklmnopqrstuvwx")
    monkeypatch.setenv("SAFE_VAR", "visible")
    env = scrubbed_env(ExecConfig(repo_root=repo))
    assert "GH_TOKEN" not in env
    assert "HF_TOKEN" not in env
    assert "MY_API_KEY" not in env
    assert env["SAFE_VAR"] == "visible"


def test_scrubbed_env_can_show_gpu_when_asked(repo: Path) -> None:
    env = scrubbed_env(ExecConfig(repo_root=repo, hide_gpu=False))
    assert env.get("CUDA_VISIBLE_DEVICES") != ""


def test_secret_absent_from_child_output(monkeypatch, repo: Path) -> None:
    monkeypatch.setenv("GH_TOKEN", "ghp_1234567890ABCDEFGHIJ")
    out = run_subprocess(
        [sys.executable, "-c", "import os; print(os.environ.get('GH_TOKEN', 'ABSENT'))"],
        cfg=ExecConfig(repo_root=repo),
    )
    assert "ABSENT" in out.output
    assert "ghp_" not in out.output


# -------------------------------------------------------------------- timeouts


def test_timeout_kills_process_group(repo: Path) -> None:
    out = run_subprocess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cfg=ExecConfig(repo_root=repo, timeout_s=1),
    )
    assert out.timed_out
    assert out.duration_s < 15


def test_timeout_through_tool(repo: Path, act) -> None:
    res = act.call("run_python", {"code": "import time; time.sleep(30)", "timeout_s": 1})
    assert not res.ok
    assert "timeout" in res.content
    assert res.meta["timed_out"] is True


# ------------------------------------------------------------------ output cap


def test_output_is_capped_head_and_tail(repo: Path) -> None:
    code = "print('A'*200000); print('ENDMARKER')"
    out = run_subprocess([sys.executable, "-c", code], cfg=ExecConfig(repo_root=repo, max_output_bytes=2000))
    assert out.truncated
    assert "bytes omitted" in out.output
    assert "ENDMARKER" in out.output


# -------------------------------------------------------------------- running


def test_run_python_sees_repo_on_path(repo: Path, act) -> None:
    res = act.call("run_python", {"code": "import mod; print(mod.VALUE)"})
    assert res.ok
    assert "42" in res.content


def test_run_python_exit_does_not_kill_session(repo: Path, act) -> None:
    res = act.call("run_python", {"code": "import sys; sys.exit(3)"})
    assert not res.ok
    assert res.meta["returncode"] == 3


def test_run_python_syntax_error_reported(repo: Path, act) -> None:
    res = act.call("run_python", {"code": "def f(:"})
    assert not res.ok
    assert res.meta["returncode"] != 0


def test_run_shell_argv_mode(repo: Path, act) -> None:
    res = act.call("run_shell", {"argv": [sys.executable, "-c", "print('hello')"]})
    assert res.ok
    assert "hello" in res.content


def test_run_shell_denied_command(repo: Path, act) -> None:
    with pytest.raises(ToolError, match="denied"):
        act.call("run_shell", {"argv": ["git", "push", "--force", "origin", "main"]})


def test_run_shell_needs_approval_for_install(repo: Path, act) -> None:
    with pytest.raises(ToolError, match="needs approval"):
        act.call("run_shell", {"argv": ["pip", "install", "requests"]})


def test_run_shell_network_needs_approval(repo: Path, act) -> None:
    with pytest.raises(ToolError, match="needs approval"):
        act.call("run_shell", {"argv": ["curl", "https://example.com"]})


def test_requires_argv_or_command(act) -> None:
    with pytest.raises(ToolError, match="provide either"):
        act.call("run_shell", {})


def test_verify_phase_has_no_write_tools(repo: Path) -> None:
    reg = build_registry(repo, Config(), phase=Phase.VERIFY)
    assert "edit_file" not in reg
    assert "run_shell" in reg


# ------------------------------------------------------------------ run_tests


def _pytest_repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "test_sample.py").write_text(
        "def helper(x):\n"
        "    raise KeyError(f'missing {x}')\n"
        "\n"
        "def test_deep():\n"
        "    helper(3)\n"
        "\n"
        "def test_ok():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    return root


def test_run_tests_reports_failing_names(tmp_path: Path) -> None:
    """The whole point: the model sees names and errors, not a 500-line dump."""
    repo = _pytest_repo(tmp_path)
    reg = build_registry(repo, Config(), phase=Phase.ACT)
    res = reg.call("run_tests", {"argv": [sys.executable, "-m", "pytest", "-q", "test_sample.py"]})

    assert not res.ok
    assert res.meta["failures"] == ["test_sample.py::test_deep"]
    assert res.meta["passed"] == 1
    assert "test_sample.py::test_deep" in res.content
    assert "KeyError" in res.content
    assert len(res.content) < 1000, "the digest must stay small"


def test_run_tests_passes_when_green(tmp_path: Path) -> None:
    repo = tmp_path / "ok"
    repo.mkdir()
    (repo / "test_ok.py").write_text("def test_a():\n    assert 1 == 1\n", encoding="utf-8")
    reg = build_registry(repo, Config(), phase=Phase.VERIFY)
    res = reg.call("run_tests", {"argv": [sys.executable, "-m", "pytest", "-q"]})
    assert res.ok
    assert res.meta["failures"] == []
    assert "1 passed" in res.content


def test_run_tests_argv_is_not_re_split(tmp_path: Path) -> None:
    """A tokenized command from config must survive quoting intact."""
    repo = tmp_path / "q"
    repo.mkdir()
    reg = build_registry(repo, Config(), phase=Phase.ACT)
    res = reg.call("run_tests", {"argv": [sys.executable, "-c", "print('hi')"]})
    assert res.ok
    assert "hi" in res.content


def test_run_tests_accepts_a_command_string(tmp_path: Path) -> None:
    repo = tmp_path / "s"
    repo.mkdir()
    reg = build_registry(repo, Config(), phase=Phase.ACT)
    res = reg.call("run_tests", {"command": f"{sys.executable} -c \"print(7)\""})
    assert res.ok
    assert "7" in res.content


def test_run_tests_is_exec_safety(tmp_path: Path) -> None:
    """It must not be reachable from the read-only SURVEY phase."""
    repo = tmp_path / "safety"
    repo.mkdir()
    reg = build_registry(repo, Config(), phase=Phase.SURVEY)
    assert "run_tests" not in reg
