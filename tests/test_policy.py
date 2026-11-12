"""Approval matrix and push allowlist tests (PRD FR-TOL-4, FR-GIT-8/9)."""

from __future__ import annotations

from pathlib import Path

from coagent.safety.policy import Policy, RiskClass, Verdict


def _policy(root: Path, *, interactive: bool = True, **kw) -> Policy:
    return Policy(repo_root=root, interactive=interactive, **kw)


# ------------------------------------------------------------------- reads


def test_read_inside_repo_allowed(tmp_path: Path) -> None:
    p = _policy(tmp_path)
    assert p.check_read(tmp_path / "src" / "a.py").verdict is Verdict.ALLOW


def test_read_outside_repo_denied(tmp_path: Path) -> None:
    p = _policy(tmp_path / "repo")
    assert p.check_read(tmp_path / "other" / "x").verdict is Verdict.DENY


def test_read_dot_git_denied(tmp_path: Path) -> None:
    p = _policy(tmp_path)
    assert p.check_read(tmp_path / ".git" / "config").verdict is Verdict.DENY


def test_read_allowlist_entry(tmp_path: Path) -> None:
    p = _policy(tmp_path / "repo", allow_read=[tmp_path / "data"])
    assert p.check_read(tmp_path / "data" / "x.csv").verdict is Verdict.ALLOW


def test_read_scratch_dir(tmp_path: Path) -> None:
    p = _policy(tmp_path / "repo", scratch=tmp_path / "scratch")
    assert p.check_read(tmp_path / "scratch" / "x").verdict is Verdict.ALLOW


# ------------------------------------------------------------------ writes


def test_write_inside_repo_allowed(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_write(tmp_path / "a.py").verdict is Verdict.ALLOW


def test_write_outside_asks_interactively(tmp_path: Path) -> None:
    d = _policy(tmp_path / "repo").check_write(tmp_path / "elsewhere" / "a.py")
    assert d.verdict is Verdict.ASK
    assert d.needs_approval


def test_write_outside_denied_non_interactive(tmp_path: Path) -> None:
    d = _policy(tmp_path / "repo", interactive=False).check_write(tmp_path / "elsewhere" / "a.py")
    assert d.verdict is Verdict.DENY


def test_write_outside_allowed_when_class_not_gated(tmp_path: Path) -> None:
    p = _policy(tmp_path / "repo", require_approval=())
    assert p.check_write(tmp_path / "elsewhere" / "a.py").verdict is Verdict.ALLOW


# ------------------------------------------------------------------- shell


def test_denylisted_command_denied_even_if_risk_not_gated(tmp_path: Path) -> None:
    p = _policy(tmp_path, require_approval=(), interactive=True)
    assert p.check_shell("git push --force origin coagent/x").verdict is Verdict.DENY


def test_reset_hard_denied(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_shell("git reset --hard HEAD~1").verdict is Verdict.DENY


def test_plain_tests_allowed(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_shell("pytest -q").verdict is Verdict.ALLOW


def test_install_asks_interactively(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_shell("pip install requests").verdict is Verdict.ASK


def test_install_denied_non_interactive(tmp_path: Path) -> None:
    assert _policy(tmp_path, interactive=False).check_shell("pip install requests").verdict is Verdict.DENY


def test_network_asks(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_shell("curl https://example.com").verdict is Verdict.ASK


def test_delete_asks(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_shell("rm -rf build").verdict is Verdict.ASK
    assert _policy(tmp_path).check_shell("chmod +x run.sh").verdict is Verdict.ASK


def test_non_interactive_flips_to_deny(tmp_path: Path) -> None:
    p = _policy(tmp_path, interactive=False)
    for cmd in ("pip install x", "curl https://x", "rm -rf build", "mv a b"):
        assert p.check_shell(cmd).verdict is Verdict.DENY, cmd


# -------------------------------------------------------------------- push


def test_push_session_branch_allowed(tmp_path: Path) -> None:
    p = _policy(tmp_path)
    assert p.check_push("origin", "coagent/2026-11-12-a1b2c3").verdict is Verdict.ALLOW


def test_push_main_denied(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_push("origin", "main").verdict is Verdict.DENY


def test_push_other_branch_denied(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_push("origin", "feature/x").verdict is Verdict.DENY


def test_push_unconfigured_remote_denied(tmp_path: Path) -> None:
    assert _policy(tmp_path).check_push("evil", "coagent/x").verdict is Verdict.DENY


def test_push_refs_heads_form(tmp_path: Path) -> None:
    p = _policy(tmp_path)
    assert p.check_push("origin", "refs/heads/coagent/x").verdict is Verdict.ALLOW


# --------------------------------------------------------------- risk classes


def test_risk_class_values_match_config_keys() -> None:
    assert {r.value for r in RiskClass} == {
        "network",
        "install",
        "delete",
        "write_outside_repo",
    }
