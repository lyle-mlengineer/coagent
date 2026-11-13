"""The notebook surface (DESIGN §8.1).

`render.py` turns the loop's event stream into a transcript as plain data;
`notebook.py` draws it in a Colab cell and collects approvals. The split exists so
that the interesting logic can be tested without a browser, and so the CLI can
reuse the same transcript without widgets.
"""

from __future__ import annotations

from .render import (
    MODE_GLYPH,
    Row,
    Transcript,
    render_status_line,
    shorten_path,
    summarise_verify,
    tool_summary,
)

__all__ = [
    "ChatCell",
    "GradioApp",
    "GradioUnavailable",
    "MODE_GLYPH",
    "Row",
    "Transcript",
    "gradio_available",
    "render_status_line",
    "shorten_path",
    "summarise_verify",
    "tool_summary",
]


def __getattr__(name: str):
    """Import the widget surfaces lazily so `render` stays usable headless.

    Both surfaces pull in a different optional dependency (ipywidgets, gradio);
    neither should be needed to import the transcript that renders them.
    """
    if name == "ChatCell":
        from .notebook import ChatCell

        return ChatCell
    if name in {"GradioApp", "GradioUnavailable", "gradio_available"}:
        from . import gradio_app

        return getattr(gradio_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
