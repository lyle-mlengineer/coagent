"""Bootstrap contract and the bootstrap notebook (FR-BOOT-1, FR-BOOT-2, FR-DIS-2).

Two things are pinned here. First, the guarantees the one-cell story depends on:
the agent refuses unsuitable hardware *before* loading weights, and a repo URL is
cloned rather than mistaken for a path.

Second, the notebook itself. It is the landing page, and a landing page that
silently rots is worse than none — so its structure is asserted here rather than
noticed by a stranger on Colab.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from coagent.agent.loop import HardwareError, _is_url, coAgent
from coagent.config import load_config, write_default_config
from coagent.gitx.session import SessionError
from coagent.llm.base import FakeBackend

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "bootstrap.ipynb"


def _init_repo(root: Path) -> Path:
    """A real repo with a config that needs no network and no linter."""
    root.mkdir(parents=True, exist_ok=True)
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "T"),
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    (root / ".gitignore").write_text(".coagent/\n__pycache__/\n", encoding="utf-8")
    (root / "f.py").write_text("X = 1\n", encoding="utf-8")
    path = write_default_config(root)
    text = path.read_text(encoding="utf-8")
    text = text.replace('author_name = "coAgent"', 'author_name = "T"')
    text = text.replace('author_email = "coagent@localhost"', 'author_email = "t@example.com"')
    text = text.replace('mode = "push"', 'mode = "bundle"')
    text = text.replace('commands = ["ruff check .", "pytest -q"]', 'commands = ["true"]')
    path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "init"], cwd=root, check=True, capture_output=True
    )
    return root


# ------------------------------------------------------------------- preflight


def test_a_cpu_runtime_is_refused_before_weights_load(monkeypatch, tmp_path: Path) -> None:
    """FR-BOOT-2: stop early, with a reason, not a CUDA stack trace."""
    import coagent.llm as llm

    monkeypatch.setattr(
        llm,
        "preflight_gpu",
        lambda **_kw: llm.PreflightResult(ok=False, reason="no CUDA device"),
    )
    monkeypatch.setattr(
        llm,
        "make_backend",
        lambda *_a, **_k: pytest.fail("weights must not be loaded after a refusal"),
    )
    repo = _init_repo(tmp_path / "proj")

    with pytest.raises(HardwareError) as err:
        coAgent(repo, backend=None, config=load_config(repo / ".coagent" / "config.toml"))
    message = str(err.value)
    assert "no CUDA device" in message
    # The message has to say how to get unstuck, or it is just a wall.
    assert "allow_unsupported" in message


def test_allow_unsupported_gets_past_the_refusal(monkeypatch, tmp_path: Path) -> None:
    import coagent.llm as llm

    monkeypatch.setattr(
        llm,
        "preflight_gpu",
        lambda **_kw: llm.PreflightResult(ok=False, reason="not a T4"),
    )
    built: list[str] = []

    class _Backend:
        model_id = "fake"
        count_tokens = staticmethod(lambda text: len(str(text).split()))

        def generate(self, *_a, **_k):
            raise AssertionError("not reached")

    def fake_backend(*_a, **_k):
        built.append("yes")
        return _Backend()

    monkeypatch.setattr(llm, "make_backend", fake_backend)
    repo = _init_repo(tmp_path / "proj")
    agent = coAgent(repo, config=load_config(repo / ".coagent" / "config.toml"), allow_unsupported=True)
    assert agent.model_id == "fake"
    assert built == ["yes"]


def test_an_injected_backend_skips_preflight(monkeypatch, tmp_path: Path) -> None:
    """A caller passing a backend has already decided about hardware."""
    import coagent.llm as llm

    monkeypatch.setattr(
        llm,
        "preflight_gpu",
        lambda **_kw: pytest.fail("an injected backend must skip the GPU probe"),
    )
    repo = _init_repo(tmp_path / "proj")
    agent = coAgent(repo, backend=FakeBackend(["done"]), config=load_config(repo / ".coagent" / "config.toml"))
    assert agent.model_id


# ------------------------------------------------------------------- repo URLs


def test_urls_are_recognised_and_paths_are_not() -> None:
    assert _is_url("https://github.com/me/proj.git")
    assert _is_url("http://example.com/x")
    assert _is_url("git@github.com:me/proj.git")
    assert not _is_url("/tmp/proj")
    assert not _is_url("proj")  # a local directory named like a remote
    assert not _is_url("./some.git")  # a local path that ends in .git is a path


def test_a_url_clone_is_not_repeated(monkeypatch, tmp_path: Path) -> None:
    """FR-BOOT-1: `coAgent(repo="https://…")` clones once, then reuses."""
    from coagent.gitx import session as session_mod

    monkeypatch.setattr(
        session_mod, "clone_repo", lambda *_a, **_k: pytest.fail("must not re-clone")
    )
    repo = _init_repo(tmp_path / "proj")
    # `/content` is absent on the dev box, so the workspace lands in cwd.
    monkeypatch.setattr("coagent.agent.loop.Path.cwd", staticmethod(lambda: tmp_path))
    workspace = tmp_path / "coagent-workspace" / "proj"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    repo.rename(workspace)

    agent = coAgent(
        "https://github.com/me/proj.git",
        backend=FakeBackend(["done"]),
        config=load_config(workspace / ".coagent" / "config.toml"),
    )
    assert agent.repo_path == workspace.resolve()


def test_clone_failure_is_a_clear_error(monkeypatch, tmp_path: Path) -> None:
    from coagent.gitx import session as session_mod

    def boom(*_a, **_k):
        raise SessionError("repository not found")

    monkeypatch.setattr(session_mod, "clone_repo", boom)
    monkeypatch.setattr("coagent.agent.loop.Path.cwd", staticmethod(lambda: tmp_path))
    with pytest.raises(SessionError):
        coAgent("https://github.com/me/absent.git", backend=FakeBackend(["done"]))


# ------------------------------------------------------------------- notebook


def test_the_bootstrap_notebook_is_valid() -> None:
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert nb.metadata.get("accelerator") == "GPU", "must ask for the T4 by default"


def test_the_chat_cell_is_at_most_three_lines() -> None:
    """FR-BOOT-1 is a hard limit, and it is easy to drift past it."""
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    chat = [c for c in nb.cells if c.cell_type == "code" and "coAgent(" in c.source]
    assert len(chat) == 1
    lines = [line for line in chat[0].source.splitlines() if line.strip()]
    assert len(lines) <= 3, f"the chat cell is {len(lines)} lines: {lines}"


def test_the_notebook_checks_hardware_before_building_the_agent() -> None:
    """FR-BOOT-2: preflight comes first, in its own cell."""
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    preflight_at = next(i for i, s in enumerate(code) if "preflight_gpu" in s)
    agent_at = next(i for i, s in enumerate(code) if "coAgent(" in s)
    assert preflight_at < agent_at


def test_the_notebook_installs_and_documents_the_text_fallback() -> None:
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    everything = "\n".join(c.source for c in nb.cells)
    assert "pip" in everything and "git+https" in everything
    # FR-UX-7 has to be discoverable from the landing page, not just the code.
    for word in ("approve", "deny", "stop"):
        assert f'command("{word}")' in everything


def test_notebook_code_cells_compile() -> None:
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = "\n".join(
            line for line in cell.source.splitlines() if not line.startswith(("!", "%"))
        )
        compile(source, "<cell>", "exec")


def test_notebook_json_is_parseable() -> None:
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
