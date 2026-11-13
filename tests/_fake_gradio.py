"""A tiny stand-in for `gradio`, so GradioApp's own logic can be exercised.

Gradio does not install on Python 3.15 (its pinned deps need older Pythons), and
the parts of GradioApp worth testing — the approval queue, event folding, HTML
escaping — are ours, not Gradio's. This module implements just the widget surface
`GradioApp.build()` touches, so the real methods run against a real object graph.
"""

from __future__ import annotations


class _Component:
    def __init__(self, *args, **kwargs):
        self.value = kwargs.get("value", args[0] if args else "")
        self.visible = kwargs.get("visible", True)
        self.events: list[str] = []

    def click(self, fn, inputs=None, outputs=None, **kwargs):
        self.events.append(("click", fn))

    def submit(self, fn, inputs=None, outputs=None, **kwargs):
        self.events.append(("submit", fn))


class _Row(_Component):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Blocks(_Component):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.closed = False
        self.launched: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def launch(self, **kwargs):
        self.launched = kwargs
        return ("launched", kwargs)

    def close(self):
        self.closed = True


class Markdown(_Component):
    pass


class HTML(_Component):
    pass


class Textbox(_Component):
    pass


class Button(_Component):
    pass


Row = _Row
Blocks = _Blocks


def __getattr__(name):  # unknown widget -> a generic component
    return type(name, (_Component,), {})
