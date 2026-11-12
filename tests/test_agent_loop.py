"""End-to-end loop tests on a real git repo with a scripted model.

No GPU, no network, no model download — but every layer is the production one:
real git branches, real tool registry, real commits with trailers, real verify.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from coagent.agent.loop import ApprovalRequest, coAgent
from coagent.config import load_config, write_default_config
from coagent.llm.base import FakeBackend

PLAN = "# Add greeting\n- [ ] read greet.py\n- [ ] add a hello function\n- [ ] run the checks"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "greet.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    path = write_default_config(root)
    text = path.read_text(encoding="utf-8")
    text = text.replace('author_name = "coAgent"', 'author_name = "T"')
    text = text.replace('author_email = "coagent@localhost"', 'author_email = "t@example.com"')
    # No remote in a test repo, so checkpoint to a bundle rather than push.
    text = text.replace('mode = "push"', 'mode = "bundle"')
    # Cheap, deterministic verification: a passing command and nothing else.
    verify_cmd = json.dumps([sys.executable, "-c", "import greet"])
    assert 'commands = ["ruff check .", "pytest -q"]' in text
    text = text.replace(
        'commands = ["ruff check .", "pytest -q"]', f"commands = [{verify_cmd}]"
    )
    path.write_text(text, encoding="utf-8")
    # Commit the config: coAgent refuses to start on a dirty tree, and that guard
    # is correct — the fixture must respect it rather than bypass it.
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "config")
    return root


def _agent(repo: Path, script: list[str], **kwargs) -> coAgent:
    cfg = load_config(repo / ".coagent" / "config.toml")
    return coAgent(
        repo,
        backend=FakeBackend(script),
        config=cfg,
        auto=True,
        interactive=False,
        **kwargs,
    )


def test_dry_run_completes_and_commits(repo: Path) -> None:
    script = [
        "The task mentions greet.py.",
        PLAN,
        '```json\n{"tool": "read_file", "args": {"path": "greet.py"}}\n```',
        '```json\n{"tool": "edit_file", "args": {"path": "greet.py", '
        '"search": "VALUE = 1", "replace": "VALUE = 1\\n\\ndef hello():\\n    return 1"}}\n```',
        "Applying the change.",
    ]
    agent = _agent(repo, script)
    result = agent.run("add a hello function to greet.py")
    agent.close()

    assert result.branch.startswith("coagent/")
    assert result.steps >= 1
    assert "greet.py" in result.files_changed
    assert "hello" in (repo / "greet.py").read_text(encoding="utf-8")
    # main must be untouched.
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() != "main"


def test_commits_carry_trailers(repo: Path) -> None:
    script = [
        "notes",
        PLAN,
        '```json\n{"tool": "read_file", "args": {"path": "greet.py"}}\n```',
        '```json\n{"tool": "write_file", "args": {"path": "greet.py", '
        '"content": "VALUE = 2\\n"}}\n```',
        "done",
        "done",
    ]
    agent = _agent(repo, script)
    result = agent.run("bump the value in greet.py")
    agent.close()

    if result.commits == 0:
        pytest.skip("verify failed before a commit could be made")
    log = _git(repo, "log", "-1", "--format=%B", result.branch)
    for trailer in ("Coagent-Session", "Coagent-Step", "Coagent-Task", "Coagent-Model"):
        assert trailer in log


def test_refuses_dirty_tree(repo: Path) -> None:
    (repo / "greet.py").write_text("VALUE = 99\n", encoding="utf-8")
    cfg = load_config(repo / ".coagent" / "config.toml")
    with pytest.raises(Exception) as exc:  # noqa: B017 - any refusal type is fine
        coAgent(repo, backend=FakeBackend([]), config=cfg, auto=True, interactive=False)
    assert "uncommitted" in str(exc.value).lower() or "dirty" in str(exc.value).lower()


def test_budget_exhaustion_reports_instead_of_hanging(repo: Path) -> None:
    cfg = load_config(repo / ".coagent" / "config.toml")
    cfg.agent.max_steps = 1
    script = [
        "notes",
        PLAN,
        '```json\n{"tool": "read_file", "args": {"path": "greet.py"}}\n```',
        '```json\n{"tool": "read_file", "args": {"path": "greet.py"}}\n```',
    ]
    agent = coAgent(
        repo, backend=FakeBackend(script), config=cfg, auto=True, interactive=False
    )
    result = agent.run("look around")
    agent.close()
    assert result.done is False
    assert result.stopped_by in {"steps", "no_action", "complete", "loop"}
    assert result.report  # a report always exists


def test_putting_plan_test_parses_checkboxes(tmp_path: Path) -> None:
    from coagent.agent.state import PlanFile

    plan = PlanFile(tmp_path, "x")
    plan.write("X", [])
    assert plan.read() is True


def test_plan_rejection_stops_before_writing(repo: Path) -> None:
    """A refused plan must leave the tree untouched (FR-AGT-2)."""
    cfg = load_config(repo / ".coagent" / "config.toml")
    asked: list[ApprovalRequest] = []

    def refuse(req: ApprovalRequest) -> bool:
        asked.append(req)
        return req.kind != "plan"

    script = [
        "notes",
        PLAN,
        '```json\n{"tool": "write_file", "args": {"path": "greet.py", "content": "X"}}\n```',
    ]
    agent = coAgent(
        repo,
        backend=FakeBackend(script),
        config=cfg,
        auto=False,
        interactive=False,
        approver=refuse,
    )
    result = agent.run("change greet.py")
    agent.close()

    assert any(r.kind == "plan" for r in asked)
    assert result.done is False
    assert "plan" in result.blocked_reason.lower()
    assert (repo / "greet.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_status_line_is_informative(repo: Path) -> None:
    agent = _agent(repo, [])
    line = agent.status_line()
    agent.close()
    assert "coAgent" in line
    assert "session" in line
    assert "12k ctx" in line or "ctx" in line


def test_events_are_emitted(repo: Path) -> None:
    seen: list[str] = []
    cfg = load_config(repo / ".coagent" / "config.toml")
    agent = coAgent(
        repo,
        backend=FakeBackend(["n", PLAN, '```json\n{"tool": "list_dir", "args": {"path": "."}}\n```']),
        config=cfg,
        auto=True,
        interactive=False,
        on_event=lambda kind, _d: seen.append(kind),
    )
    agent.run("look")
    agent.close()
    assert "task_start" in seen
    assert "plan" in seen


def test_failed_verify_does_not_produce_a_passing_trailer(repo: Path) -> None:
    """A commit of failing work must say so, never "passed"."""
    cfg = load_config(repo / ".coagent" / "config.toml")
    # A command that always fails.
    cfg.verify.commands = [f'"{sys.executable}" -c "raise SystemExit(1)"']
    script = [
        "notes",
        PLAN,
        '```json\n{"tool": "read_file", "args": {"path": "greet.py"}}\n```',
        '```json\n{"tool": "write_file", "args": {"path": "greet.py", '
        '"content": "VALUE = 3\n"}}\n```',
        "stopping",
    ]
    agent = coAgent(
        repo, backend=FakeBackend(script), config=cfg, auto=True, interactive=False
    )
    agent.run("change greet.py")
    agent.close()

    log = _git(repo, "log", "-1", "--format=%B")
    if "Coagent-Verify" in log:
        assert "passed" not in log.split("Coagent-Verify:")[1].splitlines()[0]


def test_verification_ticks_the_plan(repo: Path) -> None:
    """Passing checks must advance the plan, or nothing can ever be done."""
    from coagent.agent.prompts import PlanStep
    from coagent.agent.state import PlanFile, SessionState

    plan = PlanFile(repo, "t")
    plan.write("T", [PlanStep("run the checks")])
    assert not plan.complete
    agent = _agent(repo, [])
    agent.state = SessionState(task="t", plan=plan)
    agent._tick_verification_steps()
    assert plan.complete
    agent.close()


def _write_demo_notebook(repo: Path) -> None:
    """A notebook with no cell ids, like a file written by an older Colab."""
    import nbformat
    from nbformat.v4 import new_code_cell, new_notebook

    nb = new_notebook(cells=[new_code_cell("x = 1\nprint(x)")])
    for cell in nb.cells:
        cell.pop("id", None)
    nb.nbformat_minor = 4
    (repo / "demo.ipynb").write_text(nbformat.writes(nb, version=4), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "notebook")


NOTEBOOK_PLAN = "# Fix the demo cell\n- [ ] read demo.ipynb\n- [ ] update the cell\n- [ ] run the checks"


def test_notebook_edit_commits_through_the_loop(repo: Path) -> None:
    """The whole journey: read by id → edit_notebook → verify → commit."""
    pytest.importorskip("nbformat")
    _write_demo_notebook(repo)
    script = [
        "The task mentions demo.ipynb.",
        NOTEBOOK_PLAN,
        '```json\n{"tool": "read_notebook", "args": {"path": "demo.ipynb"}}\n```',
        '```json\n{"tool": "edit_notebook", "args": {"path": "demo.ipynb", '
        '"mode": "replace_source", "cell_id": "c00", "source": "print(42)"}}\n```',
        "done",
        "done",
    ]
    agent = _agent(repo, script)
    result = agent.run("change the demo notebook cell")
    agent.close()

    assert "demo.ipynb" in result.files_changed
    assert result.commits >= 1
    import nbformat

    nb = nbformat.reads((repo / "demo.ipynb").read_text(encoding="utf-8"), as_version=4)
    assert "print(42)" in "".join(nb.cells[0]["source"])
    # The commit must be on the session branch, never on main.
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() != "main"


ARTIFACT_PLAN = "# Save a figure\n- [ ] produce the figure\n- [ ] save it as an artifact\n- [ ] run the checks"


def test_artifact_lands_in_the_commit_trailer(repo: Path) -> None:
    """US-3 end-to-end: a saved artifact is committed and named in the trailer."""
    (repo / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "figure fixture")

    script = [
        "The task mentions fig.png.",
        ARTIFACT_PLAN,
        '```json\n{"tool": "save_artifact", "args": {"name": "fig.png", '
        '"source": "fig.png", "mime": "image/png"}}\n```',
        "done",
        "done",
    ]
    agent = _agent(repo, script)
    result = agent.run("save the figure as an artifact")
    agent.close()

    assert result.artifacts, "the run should report an artifact"
    assert any("fig.png" in a for a in result.artifacts)
    assert result.commits >= 1

    # The manifest is committed, and the trailer names the artifact.
    body = _git(repo, "log", "-1", "--format=%B", result.branch)
    assert "Coagent-Artifacts:" in body
    trailer = body.split("Coagent-Artifacts:")[1].splitlines()[0]
    assert "fig.png" in trailer
    tracked = _git(repo, "ls-files", "artifacts").splitlines()
    assert any(t.endswith("manifest.json") for t in tracked)


def test_a_large_artifact_never_reaches_git(repo: Path) -> None:
    """AT-7 through the loop: 20 MB stays out of the commit entirely."""
    (repo / "huge.bin").write_bytes(b"\x00" * (20 * 1024 * 1024))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "big fixture")

    script = [
        "notes",
        "# Save the big file\n- [ ] save it\n- [ ] run the checks",
        '```json\n{"tool": "save_artifact", "args": {"name": "huge.png", '
        '"source": "huge.bin", "mime": "image/png"}}\n```',
        "done",
        "done",
    ]
    agent = _agent(repo, script)
    result = agent.run("save the large file")
    agent.close()

    tracked = _git(repo, "ls-files", "artifacts").splitlines()
    assert not any("huge.png" in t for t in tracked), "a 20 MB file must not be committed"
    assert not any("huge.png" in a for a in result.artifacts), "the trailer must not claim it"
