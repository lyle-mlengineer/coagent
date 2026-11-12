"""Filesystem tool tests: jail, read-before-write, SEARCH/REPLACE (DESIGN §6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from coagent.config import Config
from coagent.tools import Phase, build_registry
from coagent.tools.base import ToolError, ToolSchemaError
from coagent.tools.fs import apply_search_replace


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "loader.py").write_text(
        "def load(path):\n"
        "    rows = read(path)\n"
        "    return Frame(rows)\n"
        "\n"
        "def _infer_dtypes(rows):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# demo\n\nA small repo.\n", encoding="utf-8")
    return root


@pytest.fixture()
def act(repo: Path):
    return build_registry(repo, Config(), phase=Phase.ACT)


# --------------------------------------------------------------------- reads


def test_read_file_numbers_lines(repo: Path, act) -> None:
    out = act.call("read_file", {"path": "src/loader.py"})
    assert out.ok
    assert "1\tdef load(path):" in out.content
    assert "2\t    rows = read(path)" in out.content


def test_read_file_range(repo: Path, act) -> None:
    out = act.call("read_file", {"path": "src/loader.py", "start": 5, "end": 6})
    assert "5\tdef _infer_dtypes(rows):" in out.content
    assert "1\tdef load" not in out.content


def test_read_missing_file_is_actionable(act) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        act.call("read_file", {"path": "nope.py"})


def test_grep_finds_and_scopes(act) -> None:
    out = act.call("grep", {"pattern": r"def \w+", "path": "src"})
    assert out.ok
    assert "src/loader.py:1:" in out.content
    assert "src/loader.py:5:" in out.content


def test_grep_bad_regex(act) -> None:
    with pytest.raises(ToolError, match="invalid regex"):
        act.call("grep", {"pattern": "(["})


# ---------------------------------------------------------------- read-before-write


def test_edit_requires_prior_read(repo: Path, act) -> None:
    with pytest.raises(ToolError, match="has not been read this session"):
        act.call("edit_file", {"path": "src/loader.py", "search": "return {}", "replace": "return None"})


def test_edit_after_read_succeeds(repo: Path, act) -> None:
    act.call("read_file", {"path": "src/loader.py"})
    out = act.call(
        "edit_file",
        {
            "path": "src/loader.py",
            "search": "    return Frame(rows)",
            "replace": "    return Frame.from_rows(rows)",
        },
    )
    assert out.ok
    assert "Frame.from_rows(rows)" in (repo / "src" / "loader.py").read_text()
    assert "src/loader.py" in out.meta["diff"]


def test_edit_search_must_be_unique(repo: Path, act) -> None:
    (repo / "dup.py").write_text("x = 1\ny = 1\n", encoding="utf-8")
    act.call("read_file", {"path": "dup.py"})
    with pytest.raises(ToolError, match="matches 2 times"):
        act.call("edit_file", {"path": "dup.py", "search": "= 1", "replace": "= 2"})


def test_edit_no_match_returns_nearest_region(repo: Path, act) -> None:
    act.call("read_file", {"path": "src/loader.py"})
    with pytest.raises(ToolError) as exc:
        act.call(
            "edit_file",
            {
                "path": "src/loader.py",
                "search": "    return Frame(rows)\n    # trailing\n",
                "replace": "    pass\n",
            },
        )
    msg = str(exc.value)
    assert "similar" in msg or "does not appear" in msg


def test_edit_fuzzy_suggests_exact_text(repo: Path, act) -> None:
    act.call("read_file", {"path": "src/loader.py"})
    # A small typo: no exact match, so the fuzzy pass must return the real text.
    with pytest.raises(ToolError) as exc:
        act.call(
            "edit_file",
            {"path": "src/loader.py", "search": "    return Framerow(rows)", "replace": "    pass\n"},
        )
    assert "return Frame(rows)" in str(exc.value)


def test_edit_rejects_syntactically_invalid_result(repo: Path, act) -> None:
    act.call("read_file", {"path": "src/loader.py"})
    with pytest.raises(ToolError, match="unparseable"):
        act.call(
            "edit_file",
            {"path": "src/loader.py", "search": "    return Frame(rows)", "replace": "    return Frame("},
        )
    # The file must be untouched after a rejected edit.
    assert "return Frame(rows)" in (repo / "src" / "loader.py").read_text()


# ----------------------------------------------------------------------- writes


def test_write_new_file(act, repo: Path) -> None:
    out = act.call("write_file", {"path": "src/new.py", "content": "def f():\n    return 1\n"})
    assert out.ok
    assert (repo / "src" / "new.py").exists()


def test_write_rejects_large_rewrite(repo: Path, act) -> None:
    big = repo / "big.py"
    big.write_text("\n".join(f"v{i} = {i}" for i in range(300)), encoding="utf-8")
    act.call("read_file", {"path": "big.py"})
    with pytest.raises(ToolError, match="full rewrites are only allowed under 200"):
        act.call("write_file", {"path": "big.py", "content": "x = 1\n"})


def test_write_existing_requires_read(repo: Path, act) -> None:
    with pytest.raises(ToolError, match="already exists and has not been read"):
        act.call("write_file", {"path": "README.md", "content": "clobbered\n"})


def test_write_rejects_invalid_python(act) -> None:
    with pytest.raises(ToolError, match="would not parse"):
        act.call("write_file", {"path": "bad.py", "content": "def f(:\n"})


# ------------------------------------------------------------------- path jail


def test_read_outside_repo_is_denied(repo: Path, act) -> None:
    with pytest.raises(ToolError, match="outside the repo jail"):
        act.call("read_file", {"path": "../../etc/passwd"})


def test_absolute_path_outside_repo_denied(act, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    with pytest.raises(ToolError, match="outside the repo jail"):
        act.call("read_file", {"path": str(outside)})


def test_allow_read_extends_jail(repo: Path, tmp_path: Path) -> None:
    extra = tmp_path / "data"
    extra.mkdir()
    (extra / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    cfg = Config()
    cfg.tools.allow_read = [str(extra)]
    reg = build_registry(repo, cfg, phase=Phase.ACT)
    out = reg.call("read_file", {"path": str(extra / "x.csv")})
    assert out.ok and "a,b" in out.content


def test_git_dir_not_readable(repo: Path, act) -> None:
    (repo / ".git").mkdir(exist_ok=True)
    (repo / ".git" / "config").write_text("[remote]\n", encoding="utf-8")
    with pytest.raises(ToolError, match=r"\.git"):
        act.call("read_file", {"path": ".git/config"})


# --------------------------------------------------------------------- schemas


def test_unknown_tool_rejected(act) -> None:
    with pytest.raises(ToolSchemaError, match="unknown tool"):
        act.call("delete_everything", {})


def test_unknown_argument_rejected(act) -> None:
    with pytest.raises(ToolSchemaError, match="unknown argument"):
        act.call("read_file", {"path": "README.md", "verbose": True})


def test_missing_required_argument(act) -> None:
    with pytest.raises(ToolSchemaError, match="missing required argument 'path'"):
        act.call("read_file", {})


def test_integer_coerced_from_string(act) -> None:
    out = act.call("read_file", {"path": "README.md", "start": "1", "end": "1"})
    assert out.ok


# -------------------------------------------------------------- phase gating


def test_plan_phase_has_no_tools(repo: Path) -> None:
    reg = build_registry(repo, Config(), phase=Phase.PLAN)
    assert len(reg) == 0


def test_survey_phase_is_read_only(repo: Path) -> None:
    reg = build_registry(repo, Config(), phase=Phase.SURVEY)
    assert set(reg.names()) == {"grep", "list_dir", "read_file"}
    with pytest.raises(ToolSchemaError):
        reg.call("edit_file", {"path": "README.md", "search": "a", "replace": "b"})


def test_schemas_are_json_serialisable(repo: Path) -> None:
    import json

    reg = build_registry(repo, Config(), phase=Phase.ACT)
    schema = json.loads(reg.schemas_json())
    names = {s["name"] for s in schema}
    assert "edit_file" in names
    assert all("parameters" in s for s in schema)


# ------------------------------------------------------- search/replace unit


def test_apply_search_replace_exact() -> None:
    out, line = apply_search_replace("a\nb\nc\n", "b", "B")
    assert out == "a\nB\nc\n"
    assert line == 2


def test_apply_search_replace_empty_search() -> None:
    with pytest.raises(ToolError, match="search.*empty"):
        apply_search_replace("abc", "", "x")
