"""Notebook tool tests (PRD FR-TOL-6, FR-CTX-4; DESIGN §6, §6.2).

The properties that matter, and why each test exists:

* **Cells are addressed by id.** Indices shift under insert/delete, so an
  index-addressed edit is a silent wrong edit. An index is rejected outright.
* **A notebook without ids gets them on read**, before any edit, so an old
  nbformat-4.4 file is still safe to change.
* **Outputs never enter context.** A `display(df)` must not be able to evict the
  model's window; the test asserts a 500-line output arrives as a 40-line tail.
* **A bad edit never lands.** A syntax error is refused before the file is
  touched, and an invalid notebook is refused at write time.
* **Execution is real and isolated.** `exec_notebook` runs a kernel in a
  subprocess, survives a failing cell, and reports the traceback without
  dumping the whole output into the prompt.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from coagent.config import Config
from coagent.safety.policy import Policy, Verdict
from coagent.tools import Phase, build_registry
from coagent.tools.base import ToolError
from coagent.tools.notebooks import (
    MAX_TAIL_LINES,
    NotebookDoc,
    assign_cell_ids,
    load_notebook,
    render_cells,
    summarise_execution,
)

nbformat = pytest.importorskip("nbformat")


def _vnb(*cells):
    """Build a v4 notebook, like `new_notebook` but writable from source strings."""
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    built = []
    for kind, source in cells:
        built.append(new_markdown_cell(source) if kind == "md" else new_code_cell(source))
    return new_notebook(cells=built)


def _write_notebook(root: Path, rel: str, *cells, strip_ids: bool = True) -> Path:
    """Write a notebook, by default as nbformat 4.4 — i.e. with no cell ids."""
    nb = _vnb(*cells)
    if strip_ids:
        for cell in nb.cells:
            cell.pop("id", None)
        nb.nbformat_minor = 4
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbformat.writes(nb, version=4), encoding="utf-8")
    return path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _write_notebook(root, "analysis.ipynb", ("md", "# Demo"), ("code", "x = 1\nprint(x)"))
    return root


@pytest.fixture()
def act(repo: Path):
    return build_registry(repo, Config(), phase=Phase.ACT)


def _doc(path: Path, rel: str) -> NotebookDoc:
    return load_notebook(path, rel)


# ------------------------------------------------------------------ registration


def test_notebook_tools_are_registered_in_act(repo: Path) -> None:
    names = build_registry(repo, Config(), phase=Phase.ACT).names()
    assert {"read_notebook", "edit_notebook", "exec_notebook"} <= set(names)


def test_survey_phase_has_read_notebook_but_cannot_edit(repo: Path) -> None:
    names = build_registry(repo, Config(), phase=Phase.SURVEY).names()
    assert "read_notebook" in names
    assert "edit_notebook" not in names
    assert "exec_notebook" not in names


def test_plan_phase_has_no_tools(repo: Path) -> None:
    assert build_registry(repo, Config(), phase=Phase.PLAN).names() == []


def test_verify_can_execute_not_run_edits(repo: Path) -> None:
    names = build_registry(repo, Config(), phase=Phase.VERIFY).names()
    assert "exec_notebook" in names
    assert "edit_notebook" not in names


# ------------------------------------------------------------------ ids on read


def test_read_assigns_ids_to_an_old_notebook(repo: Path, act) -> None:
    out = act.call("read_notebook", {"path": "analysis.ipynb"})
    assert out.ok
    assert out.meta["ids_assigned"] == 2
    assert "cell c00 [markdown" in out.content
    assert "cell c01 [code" in out.content


def test_existing_ids_are_preserved(tmp_path: Path) -> None:
    from nbformat.v4 import new_code_cell, new_notebook

    nb = new_notebook(cells=[new_code_cell("a = 1")])
    nb.cells[0]["id"] = "keep-me"
    path = tmp_path / "n.ipynb"
    path.write_text(nbformat.writes(nb, version=4), encoding="utf-8")
    doc = _doc(path, "n.ipynb")
    assert doc.cells[0]["id"] == "keep-me"
    assert doc.assigned_ids == 0


def test_duplicate_ids_are_repaired_not_kept() -> None:
    from nbformat.v4 import new_code_cell, new_notebook

    nb = new_notebook(cells=[new_code_cell("a"), new_code_cell("b")])
    nb.cells[0]["id"] = "same"
    nb.cells[1]["id"] = "same"
    assigned = assign_cell_ids(nb)
    ids = [c["id"] for c in nb.cells]
    assert len(set(ids)) == 2
    assert assigned == 1


def test_read_does_not_show_cell_outputs(tmp_path: Path) -> None:
    from nbformat.v4 import new_code_cell, new_notebook, new_output

    cell = new_code_cell("df.head()")
    cell["outputs"] = [new_output("stream", name="stdout", text="A" * 5000)]
    nb = new_notebook(cells=[cell])
    path = tmp_path / "out.ipynb"
    path.write_text(nbformat.writes(nb, version=4), encoding="utf-8")
    body = render_cells(_doc(path, "out.ipynb"))
    assert "AAAA" not in body
    assert "1 output(s) hidden" in body


def test_read_file_redirects_ipynb_to_read_notebook(repo: Path, act) -> None:
    with pytest.raises(ToolError) as err:
        act.call("read_file", {"path": "analysis.ipynb"})
    assert "read_notebook" in str(err.value)


# -------------------------------------------------------------------- editing


def test_edit_by_id_replaces_source(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    out = act.call(
        "edit_notebook",
        {"path": "analysis.ipynb", "mode": "replace_source", "cell_id": "c01", "source": "print('new')"},
    )
    assert out.ok
    assert "-x = 1" in out.meta["diff"]
    assert "+print('new')" in out.meta["diff"]
    reread = act.call("read_notebook", {"path": "analysis.ipynb"})
    assert "print('new')" in reread.content


def test_edit_refuses_without_read(repo: Path) -> None:
    from coagent.tools.fs import FileSession

    policy = Policy(repo_root=repo)
    reg = build_registry(repo, Config(), phase=Phase.ACT, policy=policy)
    assert isinstance(reg.get("edit_notebook").session, FileSession)
    with pytest.raises(ToolError) as err:
        reg.call("edit_notebook", {"path": "analysis.ipynb", "mode": "delete", "cell_id": "c00"})
    assert "read_notebook" in str(err.value)


def test_index_is_rejected_with_valid_ids(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    with pytest.raises(ToolError) as err:
        act.call("edit_notebook", {"path": "analysis.ipynb", "mode": "delete", "cell_id": "1"})
    message = str(err.value)
    assert "by id, not index" in message
    assert "c00" in message and "c01" in message


def test_unknown_id_lists_the_real_ones(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    with pytest.raises(ToolError) as err:
        act.call(
            "edit_notebook",
            {"path": "analysis.ipynb", "mode": "replace_source", "cell_id": "nope", "source": "x=2"},
        )
    assert "Valid cell ids: c00, c01" in str(err.value)


def test_syntax_error_is_refused_and_file_untouched(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    before = (repo / "analysis.ipynb").read_text(encoding="utf-8")
    with pytest.raises(ToolError) as err:
        act.call(
            "edit_notebook",
            {"path": "analysis.ipynb", "mode": "replace_source", "cell_id": "c01", "source": "def (:"},
        )
    assert "would not parse" in str(err.value)
    assert (repo / "analysis.ipynb").read_text(encoding="utf-8") == before


def test_ipython_magics_are_not_syntax_errors(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    out = act.call(
        "edit_notebook",
        {
            "path": "analysis.ipynb",
            "mode": "replace_source",
            "cell_id": "c01",
            "source": "%matplotlib inline\n!ls\nprint('ok')",
        },
    )
    assert out.ok


def test_magic_cell_and_shell_escape_parse_gate(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    out = act.call(
        "edit_notebook",
        {"path": "analysis.ipynb", "mode": "replace_source", "cell_id": "c01", "source": "%%bash\necho hi"},
    )
    assert out.ok


def test_insert_at_top_and_after_a_cell(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    top = act.call(
        "edit_notebook",
        {"path": "analysis.ipynb", "mode": "insert", "cell_id": "", "cell_type": "markdown", "source": "# Title"},
    )
    assert top.ok
    assert "at the top" in top.content
    after = act.call(
        "edit_notebook",
        {"path": "analysis.ipynb", "mode": "insert", "cell_id": "c00", "source": "import os"},
    )
    assert after.ok
    body = act.call("read_notebook", {"path": "analysis.ipynb"}).content
    assert body.index("# Title") < body.index("# Demo") < body.index("import os")


def test_delete_requires_more_than_one_cell(tmp_path: Path) -> None:
    _write_notebook(tmp_path, "one.ipynb", ("code", "x = 1"))
    reg = build_registry(tmp_path, Config(), phase=Phase.ACT)
    reg.call("read_notebook", {"path": "one.ipynb"})
    with pytest.raises(ToolError) as err:
        reg.call("edit_notebook", {"path": "one.ipynb", "mode": "delete", "cell_id": "c00"})
    assert "only cell" in str(err.value)


def test_delete_one_of_several(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    out = act.call("edit_notebook", {"path": "analysis.ipynb", "mode": "delete", "cell_id": "c01"})
    assert out.ok
    body = act.call("read_notebook", {"path": "analysis.ipynb"}).content
    assert "1 cells" in body


def test_replace_metadata_sets_kernel(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    out = act.call(
        "edit_notebook", {"path": "analysis.ipynb", "mode": "replace_metadata", "kernel": "python3"}
    )
    assert out.ok
    doc = _doc(repo / "analysis.ipynb", "analysis.ipynb")
    assert doc.nb.metadata["kernelspec"]["name"] == "python3"


def test_edit_clears_stale_outputs(repo: Path, act) -> None:
    from nbformat.v4 import new_output

    doc = _doc(repo / "analysis.ipynb", "analysis.ipynb")
    doc.cells[1]["outputs"] = [new_output("stream", name="stdout", text="old")]
    doc.cells[1]["execution_count"] = 7
    (repo / "analysis.ipynb").write_text(nbformat.writes(doc.nb, version=4), encoding="utf-8")

    act.call("read_notebook", {"path": "analysis.ipynb"})
    act.call(
        "edit_notebook",
        {"path": "analysis.ipynb", "mode": "replace_source", "cell_id": "c01", "source": "print(2)"},
    )
    after = _doc(repo / "analysis.ipynb", "analysis.ipynb")
    assert after.cells[1]["outputs"] == []
    assert after.cells[1]["execution_count"] is None


def test_written_notebook_is_still_valid_json(repo: Path, act) -> None:
    act.call("read_notebook", {"path": "analysis.ipynb"})
    act.call("edit_notebook", {"path": "analysis.ipynb", "mode": "delete", "cell_id": "c01"})
    raw = (repo / "analysis.ipynb").read_text(encoding="utf-8")
    reparsed = nbformat.reads(raw, as_version=4)
    nbformat.validate(reparsed)
    assert len(reparsed.cells) == 1


def test_notebook_jail_applies(repo: Path, act, tmp_path: Path) -> None:
    outside = tmp_path / "outside.ipynb"
    shutil.copy(repo / "analysis.ipynb", outside)
    with pytest.raises(ToolError) as err:
        act.call("read_notebook", {"path": str(outside)})
    assert "jail" in str(err.value)


# ------------------------------------------------------------------- output summary


def _notebook_with_outputs(outputs):
    from nbformat.v4 import new_code_cell, new_notebook

    cell = new_code_cell("code")
    cell["outputs"] = outputs
    return new_notebook(cells=[cell])


def test_summary_keeps_only_the_tail_of_long_stdout() -> None:
    from nbformat.v4 import new_output

    lines = [f"line {i}\n" for i in range(500)]
    nb = _notebook_with_outputs([new_output("stream", name="stdout", text=lines)])
    doc = NotebookDoc(path=Path("x.ipynb"), rel="x.ipynb", nb=nb)
    summary, ok, meta = summarise_execution(doc)
    assert ok
    assert summary.count("\n") <= MAX_TAIL_LINES + 2
    assert "line 499" in summary
    assert "line 0" not in summary
    assert "omitted" in summary


def test_summary_reports_last_traceback() -> None:
    from nbformat.v4 import new_output

    error = new_output(
        "error",
        ename="ValueError",
        evalue="boom",
        traceback=["Traceback ...", "  cell line 3", "ValueError: boom"],
    )
    doc = NotebookDoc(path=Path("x.ipynb"), rel="x.ipynb", nb=_notebook_with_outputs([error]))
    summary, ok, meta = summarise_execution(doc)
    assert not ok
    assert meta["errors"] == 1
    assert "ValueError: boom" in summary


def test_summary_strips_ansi_escapes() -> None:
    from nbformat.v4 import new_output

    nb = _notebook_with_outputs([new_output("stream", name="stdout", text="\x1b[31mred\x1b[0m\n")])
    summary, _, _ = summarise_execution(NotebookDoc(path=Path("x.ipynb"), rel="x.ipynb", nb=nb))
    assert "red" in summary
    assert "\x1b" not in summary


def test_images_are_artifacts_not_context() -> None:
    from nbformat.v4 import new_output

    png = "iVBORw0KGgoAAAANSUhEUg==" * 200
    nb = _notebook_with_outputs([new_output("display_data", data={"image/png": png})])
    summary, ok, meta = summarise_execution(NotebookDoc(path=Path("x.ipynb"), rel="x.ipynb", nb=nb))
    assert ok
    assert "1 image(s)" in summary
    assert "iVBOR" not in summary  # the base64 never reaches the prompt
    assert meta["outputs"][0].mime == "image/png"


# ------------------------------------------------------------------- execution


def _has_kernel() -> bool:
    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError:  # pragma: no cover
        return False
    try:
        return bool(KernelSpecManager().find_kernel_specs())
    except Exception:  # noqa: BLE001
        return False


pytest_exec = pytest.mark.skipif(
    shutil.which("jupyter") is None or not _has_kernel(),
    reason="needs a Jupyter kernel (pip install ipykernel)",
)


@pytest_exec
def test_exec_runs_cells_and_reports_success(tmp_path: Path) -> None:
    _write_notebook(tmp_path, "ok.ipynb", ("code", "print(6 * 7)"))
    reg = build_registry(tmp_path, Config(), phase=Phase.ACT)
    out = reg.call("exec_notebook", {"path": "ok.ipynb", "timeout_s": 90})
    assert out.ok
    assert "42" in out.content
    assert "exit 0" in out.content


@pytest_exec
def test_exec_reports_a_failing_cell_without_dumping_output(tmp_path: Path) -> None:
    _write_notebook(
        tmp_path,
        "bad.ipynb",
        ("code", "print('noise' * 200)"),
        ("code", "raise ValueError('boom')"),
    )
    reg = build_registry(tmp_path, Config(), phase=Phase.ACT)
    out = reg.call("exec_notebook", {"path": "bad.ipynb", "timeout_s": 90})
    assert not out.ok
    assert "1 failing cell(s)" in out.content
    assert "ValueError: boom" in out.content
    # One 1000-char line must not slip past the 40-line tail (FR-CTX-4).
    assert len(out.content) < 1500
    assert "…[truncated]" in out.content


@pytest_exec
def test_exec_writes_outputs_into_the_notebook(tmp_path: Path) -> None:
    _write_notebook(tmp_path, "o.ipynb", ("code", "print('persisted')"))
    reg = build_registry(tmp_path, Config(), phase=Phase.ACT)
    reg.call("exec_notebook", {"path": "o.ipynb", "timeout_s": 90})
    doc = _doc(tmp_path / "o.ipynb", "o.ipynb")
    texts = [
        "".join(o.get("text", "")) for o in doc.cells[0].get("outputs", []) if o.get("output_type") == "stream"
    ]
    assert any("persisted" in t for t in texts)


@pytest_exec
def test_exec_timeout_kills_the_kernel(tmp_path: Path) -> None:
    _write_notebook(tmp_path, "slow.ipynb", ("code", "import time\ntime.sleep(60)"))
    reg = build_registry(tmp_path, Config(), phase=Phase.ACT)
    out = reg.call("exec_notebook", {"path": "slow.ipynb", "timeout_s": 5})
    assert not out.ok
    assert out.meta["timed_out"] is True
    assert "exceeded" in out.content


def test_exec_refuses_a_missing_notebook(repo: Path, act) -> None:
    with pytest.raises(ToolError) as err:
        act.call("exec_notebook", {"path": "nope.ipynb"})
    assert "does not exist" in str(err.value)


def test_shell_escape_in_a_cell_is_gated_by_policy(tmp_path: Path) -> None:
    """`!pip install` inside a cell must not bypass the approval matrix."""
    _write_notebook(tmp_path, "g.ipynb", ("code", "!pip install requests"))
    policy = Policy(repo_root=tmp_path, interactive=False)  # non-interactive ⇒ deny
    reg = build_registry(tmp_path, Config(), phase=Phase.ACT, policy=policy)
    with pytest.raises(ToolError) as err:
        reg.call("exec_notebook", {"path": "g.ipynb", "timeout_s": 5})
    assert "denied by policy" in str(err.value)


def test_policy_allows_a_plain_cell(tmp_path: Path) -> None:
    """A cell with no shell escape must not trip the shell gate."""
    policy = Policy(repo_root=tmp_path, interactive=False)
    _write_notebook(tmp_path, "p.ipynb", ("code", "x = 1"))
    assert policy.check_shell("python-notebook-cell").verdict is Verdict.ALLOW
