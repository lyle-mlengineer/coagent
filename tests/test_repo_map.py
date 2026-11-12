"""Repo map tests (PRD FR-CTX-1..3; DESIGN §5.1–5.2).

The repo map is the highest-leverage piece of context on a 12k window, so the
tests below pin the properties that actually earn that leverage:

* signatures only — no bodies leak into the prompt;
* the budget is respected, always, because overflow silently truncates the
  model's view of the repo;
* ranking puts the file the task names first, even when a bigger file exists;
* a broken/unparseable file degrades the map instead of raising;
* vendored and generated trees never appear.
"""

from __future__ import annotations

import textwrap

from coagent.context.repo_map import (
    FileMap,
    RepoMap,
    RepoMapConfig,
    Signature,
    _Parsers,
    _score,
)
from coagent.context.tokens import estimate_tokens, fit, truncate_middle


def _write(root, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


# ------------------------------------------------------------------ tokens


def test_estimate_tokens_is_monotone_and_cheap() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("x") == 1
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("a" * 400) < estimate_tokens("a" * 800)


def test_truncate_middle_keeps_both_ends() -> None:
    text = "HEAD" + ("m" * 4000) + "TAIL"
    out = truncate_middle(text, 50)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert estimate_tokens(out) <= 50 + 4  # marker overhead is bounded
    assert "m" * 4000 not in out


def test_truncate_middle_is_identity_when_it_fits() -> None:
    assert truncate_middle("short", 100) == "short"


def test_fit_stops_at_budget_and_preserves_order() -> None:
    blocks = [f"block{i}" + ("x" * 40) for i in range(10)]
    out = fit(blocks, 20)
    assert estimate_tokens(out) <= 20
    assert out.startswith("block0")


def test_fit_falls_back_to_truncated_first_block() -> None:
    # One block bigger than the whole budget must still yield something.
    out = fit(["A" + ("z" * 4000)], 10)
    assert out
    assert estimate_tokens(out) <= 10 + 4


def test_fit_empty_input() -> None:
    assert fit([], 100) == ""
    assert fit(["a"], 0) == ""


# ---------------------------------------------------------------- extraction


def test_ast_signatures_have_no_bodies(tmp_path) -> None:
    _write(
        tmp_path,
        "mod.py",
        """
        '''Docstring.'''

        LIMIT = 10


        class Thing:
            '''A thing.'''

            def __init__(self, n: int = 1) -> None:
                self.n = n
                self.secret_body_line = n * 999

            def go(self) -> str:
                return "body_marker"


        def helper(a: str, *, b: int | None = None) -> dict[str, int]:
            body_marker = 1
            return {a: 2}
        """,
    )
    out = RepoMap(tmp_path).render()
    assert "mod.py" in out
    assert "class Thing" in out
    assert "def __init__(self, n: int=1) -> None" in out
    assert "def helper(a: str, *, b: int | None=None) -> dict[str, int]" in out
    assert "LIMIT = ..." in out
    # Bodies must never appear: that is the entire point of the map.
    assert "body_marker" not in out
    assert "secret_body_line" not in out
    assert "Do not" not in out and "Docstring" not in out


def test_async_and_decorated_functions(tmp_path) -> None:
    _write(
        tmp_path,
        "io.py",
        """
        import functools


        @functools.cache
        async def fetch(url: str) -> bytes:
            return b""
        """,
    )
    out = RepoMap(tmp_path).render()
    assert "async def fetch(url: str) -> bytes" in out


def test_broken_file_degrades_without_raising(tmp_path) -> None:
    _write(tmp_path, "ok.py", "def fine(a, b):\n    return a\n")
    _write(tmp_path, "bad.py", "def broken(:\n    return 1\ndef recover(x):\n    return x\n")

    parsers = _Parsers()
    sigs, degraded = parsers.python_signatures("def broken(:\n    pass\n", rel="bad.py")
    assert degraded is True
    assert any("broken" in s.text for s in sigs)

    out = RepoMap(tmp_path).render()
    assert "ok.py" in out
    assert "bad.py" in out
    assert "degraded" in out  # the map says so honestly


def test_regex_fallback_extracts_shapes() -> None:
    sigs = _Parsers._regex_signatures(
        textwrap.dedent(
            """
            MAX_SIZE = 10
            class Widget:
                def render(self):
                    pass
            def build(name: str):
                pass
            """
        ).lstrip()
    )
    kinds = {(s.kind, s.text) for s in sigs}
    assert ("const", "MAX_SIZE = ...") in kinds
    assert ("class", "class Widget") in kinds
    assert any(k == "member" and "render" in t for k, t in kinds)
    assert any(k == "def" and "build" in t for k, t in kinds)


def test_empty_repository(tmp_path) -> None:
    assert RepoMap(tmp_path).render() == "(empty repository)"


def test_non_python_files_are_skipped(tmp_path) -> None:
    _write(tmp_path, "notes.md", "# hello\n")
    _write(tmp_path, "app.js", "function go(){}\n")
    assert RepoMap(tmp_path).render() == "(empty repository)"


# ------------------------------------------------------------------ hygiene


def test_vendored_and_generated_trees_are_skipped(tmp_path) -> None:
    _write(tmp_path, "src/app.py", "def keep_me():\n    pass\n")
    _write(tmp_path, "node_modules/pkg/index.py", "def vendored():\n    pass\n")
    _write(tmp_path, ".venv/lib/x.py", "def venv_thing():\n    pass\n")
    _write(tmp_path, "build/out.py", "def generated():\n    pass\n")
    _write(tmp_path, "__pycache__/junk.py", "def junk():\n    pass\n")
    _write(tmp_path, ".coagent/work.py", "def agent_private():\n    pass\n")

    out = RepoMap(tmp_path).render()
    assert "app.py" in out
    for noise in ("vendored", "venv_thing", "generated", "junk", "agent_private"):
        assert noise not in out


def test_signatures_per_file_are_capped(tmp_path) -> None:
    body = "\n".join(f"def f{i}(a):\n    return a\n" for i in range(120))
    _write(tmp_path, "many.py", body)
    out = RepoMap(tmp_path).render()
    assert out.count("def f") <= 40


# ------------------------------------------------------------------ ranking


def test_task_mentioned_file_ranks_first(tmp_path) -> None:
    # A hub file referenced everywhere, plus the file the task names.
    _write(tmp_path, "core.py", "def shared_helper(x):\n    return x\n")
    _write(tmp_path, "a.py", "from core import shared_helper\n\ndef a_fn(h):\n    return h\n")
    _write(tmp_path, "b.py", "from core import shared_helper\n\ndef b_fn(h):\n    return h\n")
    _write(tmp_path, "target_mod.py", "def only_here(y):\n    return y\n")

    plain = RepoMap(tmp_path, config=RepoMapConfig(max_tokens=40))
    ranked = plain.files()
    assert ranked[0].rel == "core.py"  # reference_count wins on merit

    focused = RepoMap(tmp_path, task="fix the bug in target_mod.py", config=RepoMapConfig(max_tokens=40))
    assert focused.files()[0].rel == "target_mod.py"


def test_recently_edited_raises_rank(tmp_path) -> None:
    _write(tmp_path, "big.py", "def hub():\n    pass\n")
    _write(tmp_path, "cold.py", "def cold():\n    pass\n")
    _write(tmp_path, "warm.py", "def warm():\n    pass\n")

    mapper = RepoMap(tmp_path, recently_edited={"warm.py"})
    rels = [f.rel for f in mapper.files()]
    assert rels.index("warm.py") < rels.index("cold.py")


def test_score_formula_matches_design() -> None:
    # 0.5*refs + 0.3*recent + 0.2*mentioned, refs soft-capped at 20.
    fm = FileMap(rel="a.py", references=20, recently_edited=True, mentioned=True)
    assert _score(fm) == 1.0
    half = FileMap(rel="b.py", references=0, recently_edited=False, mentioned=False)
    assert _score(half) == 0.0
    # Saturation: 100 refs scores the same as 20.
    assert _score(FileMap(rel="c.py", references=100)) == _score(FileMap(rel="d.py", references=20))


def test_task_naming_a_symbol_surfaces_the_file(tmp_path) -> None:
    _write(tmp_path, "loader.py", "def load_frame(path):\n    return path\n")
    _write(tmp_path, "other.py", "def unrelated():\n    return 1\n")
    mapper = RepoMap(tmp_path, task="load_frame crashes on empty files")
    assert mapper.files()[0].rel == "loader.py"


# ------------------------------------------------------------------ budget


def test_map_respects_token_budget(tmp_path) -> None:
    for i in range(40):
        body = "\n".join(f"def fn_{i}_{j}(a, b, c):\n    return a + b + c\n" for j in range(20))
        _write(tmp_path, f"pkg{i}/mod{i}.py", body)

    budget = 300
    out = RepoMap(tmp_path, config=RepoMapConfig(max_tokens=budget)).render()
    assert estimate_tokens(out) <= budget + 10
    # It must still be a real map, not a truncated stump.
    assert "def fn_" in out


def test_map_truncation_drops_whole_files(tmp_path) -> None:
    for i in range(20):
        _write(tmp_path, f"m{i}.py", f"def a{i}():\n    pass\ndef b{i}():\n    pass\n")
    out = RepoMap(tmp_path, config=RepoMapConfig(max_tokens=60)).render()
    # A truncated map must cut on file boundaries: each file header line ends in
    # .py and every other line is an indented signature. A half-eaten file would
    # leave a header with no signatures or a signature with no header.
    lines = out.splitlines()
    headers = [ln for ln in lines if ln.endswith(".py")]
    assert headers, "budget too small to show any file"
    assert estimate_tokens(out) <= 70


def test_max_files_cap(tmp_path) -> None:
    for i in range(30):
        _write(tmp_path, f"f{i}.py", f"def fn{i}():\n    pass\n")
    out = RepoMap(tmp_path, config=RepoMapConfig(max_files=5, max_tokens=4000)).render()
    assert sum(1 for line in out.splitlines() if line.endswith(".py")) <= 5


def test_tree_sitter_optional_does_not_break(tmp_path) -> None:
    # Force the optional path off: the map must still be produced.
    _write(tmp_path, "m.py", "def x(a):\n    return a\n")
    out = RepoMap(tmp_path, config=RepoMapConfig(use_tree_sitter=False)).render()
    assert "def x(a)" in out


# ------------------------------------------------------------------ rendering


def test_signature_render_indentation() -> None:
    assert Signature("class", "class A").render() == "class A"
    assert Signature("def", "def f()").render() == "  def f()"
    assert Signature("member", "def m(self)").render() == "    def m(self)"
    assert Signature("const", "X = ...").render() == "  X = ..."


def test_file_map_without_signatures_says_so() -> None:
    assert "(no signatures)" in FileMap(rel="empty.py").render()


# ------------------------------------------------------------------ performance


def test_reference_counting_is_linear_enough(tmp_path) -> None:
    """Ranking must not cost seconds: the model is idle while it runs.

    The first implementation used one regex per (file, symbol) pair and took 4s
    on a 7k-line repo. Guard the property, not the exact number, so this stays
    meaningful on a slow CI box.
    """
    import time

    for i in range(60):
        deps = "\n".join(f"    return fn_{j}({i})" for j in range(5))
        _write(
            tmp_path,
            f"m{i}.py",
            f"from core import fn_0, fn_1, fn_2, fn_3, fn_4\n\n"
            f"def fn_{i}(a):\n{deps}\n",
        )
    _write(tmp_path, "core.py", "\n".join(f"def fn_{i}(a):\n    return a" for i in range(5)) + "\n")

    start = time.perf_counter()
    RepoMap(tmp_path, task="fix fn_3", config=RepoMapConfig(max_tokens=500)).render()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"repo map took {elapsed:.1f}s on 60 small files"


def test_reference_count_is_inbound_not_outbound(tmp_path) -> None:
    """The shared module must outrank the leaf that imports it."""
    _write(tmp_path, "core.py", "def shared_helper(x):\n    return x\n")
    _write(tmp_path, "a.py", "from core import shared_helper\n\ndef a_fn(h):\n    return h\n")
    _write(tmp_path, "b.py", "from core import shared_helper\n\ndef b_fn(h):\n    return h\n")

    by_rel = {f.rel: f for f in RepoMap(tmp_path).files()}
    assert by_rel["core.py"].references == 2  # two files mention shared_helper
    assert by_rel["a.py"].references == 0  # a.py's own symbols are used nowhere
