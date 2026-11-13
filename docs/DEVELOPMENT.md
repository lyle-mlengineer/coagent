# Development

coAgent is a Python package (`src/` layout). It runs on a free Colab T4 in
production, but **everything except the model and the notebook UI must be
testable on a laptop CPU**. That rule is what keeps the loop debuggable.

## Install for development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # light: no torch, no transformers
```

The GPU stack is optional and only needed to actually generate code:

```bash
pip install -e ".[colab]"          # torch, transformers, bitsandbytes, ipywidgets…
```

`import coagent` and `coagent doctor` must never import torch at module scope,
so they work without the GPU extras.

## Common commands

| Command | Purpose |
| --- | --- |
| `pytest` | Unit tests (CPU-only, no GPU, no network) |
| `ruff check .` | Lint — also the first entry in `verify.commands` |
| `coagent init` | Write `.coagent/config.toml` into a repo |
| `coagent doctor` | Versions, config, GPU envelope verdict |
| `coagent config --effective` | Print the resolved config as TOML |
| `python -m build` | Build wheel + sdist into `dist/` |

## Layout

```
src/coagent/
  __init__.py        public surface: coAgent(), Config
  cli.py             init | doctor | config
  config.py          .coagent/config.toml load/validate/version-check
  templates/         config.toml shipped as package data
  agent/             loop.py, prompts.py, budget.py, state.py      (M1)
  llm/               base.py, registry.py, transformers_backend.py (M1)
  context/           repo_map.py, tokens.py, retrieval.py, …         (M2; repo map done)
  tools/             base.py, fs.py, shell.py, git_tools.py, …      (M1)
  gitx/              session.py, commit.py, durable.py, resume.py   (M2)
  ui/                notebook.py, render.py                         (M1)
  safety/            policy.py, denylist.py, redact.py              (M1)
  evals/             harness.py, tasks/
tests/               pytest, mirrors the package layout
notebooks/           bootstrap.ipynb — the only cell a user runs
```

## Packaging notes

- The config template ships as **package data** (`coagent/templates/config.toml`).
  It is read through `importlib.resources`, never via a relative filesystem path,
  so it works from a wheel, from an sdist, and from the source tree.
- Heavy dependencies live in extras. `dependencies` stays small so
  `pip install co-agent-coder` is fast and the test suite does not need CUDA.
- Bump `schema_version` in `config.py` only for incompatible config changes;
  older configs keep working, newer ones are refused with a migration hint.
