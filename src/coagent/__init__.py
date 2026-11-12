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

import sys

from .config import Config, ConfigError, load_config

__version__ = "0.1.0"

__all__ = ["Config", "ConfigError", "__version__", "coAgent", "load_config"]


def coAgent(*args, **kwargs):  # noqa: N802 - matches the documented notebook API
    """Construct the agent, with a notebook surface when one is useful.

    In a notebook (DESIGN §8.1) the chat cell is the product, so it is wired up
    for you: `on_event` paints the transcript, `approver` collects clicks, and
    `renderer` draws artifacts. Passing any of those explicitly wins, and
    `config [ui] mode = "headless"` turns it all off — which is what the CLI and
    the test suite rely on.

    Imported lazily so that `import coagent` does not pull in the loop, the
    tools, or the model backend.
    """
    from .agent.loop import coAgent as _coAgent

    cell = _maybe_chat_cell(args, kwargs)
    if cell is not None:
        kwargs.setdefault("on_event", cell.on_event)
        kwargs.setdefault("approver", cell.approver)
        kwargs.setdefault("renderer", cell.render_artifact)
    agent = _coAgent(*args, **kwargs)
    if cell is not None:
        cell.attach(agent)
        agent.cell = cell
    return agent


def _maybe_chat_cell(args: tuple, kwargs: dict):
    """A `ChatCell` if we are in a notebook and the user asked for one."""
    if "on_event" in kwargs or "approver" in kwargs or "renderer" in kwargs:
        return None  # the caller is wiring its own surface
    if not _in_notebook():
        return None
    config = kwargs.get("config")
    if config is not None and getattr(getattr(config, "ui", None), "mode", "show") != "show":
        return None
    if config is None:
        try:
            probe = load_config(start=kwargs.get("repo") or (args[0] if args else "."))
        except Exception:  # noqa: BLE001 - a bad config must not block construction
            probe = None
        if probe is not None and probe.ui.mode != "show":
            return None
    try:
        from .ui.notebook import ChatCell
    except Exception:  # noqa: BLE001 - no IPython/ipywidgets: stay headless
        return None
    return ChatCell()


def _in_notebook() -> bool:
    """True when running inside a live IPython kernel, not a plain script."""
    if "ipykernel" not in sys.modules:
        return False
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:  # noqa: BLE001
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
