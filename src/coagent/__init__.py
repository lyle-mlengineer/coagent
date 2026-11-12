"""coAgent — a private AI coding agent for the free Google Colab T4.

Public surface (target v0.1)::

    from coagent import coAgent
    agent = coAgent(repo="https://github.com/you/project.git")
    agent.chat("add a chunked CSV loader and test dtype preservation")

Nothing here imports torch, transformers, or IPython at module scope: importing
`coagent` must stay cheap and CPU-only so `coagent doctor` and the test suite
run on a laptop.
"""

from __future__ import annotations

from .config import Config, ConfigError, load_config

__version__ = "0.1.0"

__all__ = ["Config", "ConfigError", "__version__", "coAgent", "load_config"]


def coAgent(*args, **kwargs):  # noqa: N802 - matches the documented notebook API
    """Construct the agent (v0.1 milestone M1).

    Imported lazily so that `import coagent` does not pull in the loop, the
    tools, or the model backend.
    """
    from .agent.loop import coAgent as _coAgent

    return _coAgent(*args, **kwargs)
