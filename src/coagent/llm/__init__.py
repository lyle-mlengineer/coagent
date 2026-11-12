"""The model layer: one interface, a local 4-bit backend, and a scripted fake.

`import coagent.llm` must stay torch-free. `make_backend()` is the only place
that reaches for `transformers`, and it does so lazily.
"""

from __future__ import annotations

from typing import Any

from .base import (
    BudgetError,
    FakeBackend,
    GenerationConfig,
    GenerationResult,
    LLMBackend,
    Message,
    StopReason,
    TokenBudget,
    Usage,
)
from .grammar import (
    Envelope,
    EnvelopeError,
    extract_json_candidates,
    looks_like_final_answer,
    parse_envelope,
    repair_json,
    strip_fences,
)
from .registry import (
    MODELS,
    T4_COMPUTE_CAPABILITY,
    T4_MIN_VRAM_GB,
    ModelSpec,
    PreflightResult,
    get_model,
    get_model_by_name,
    preflight_gpu,
)

__all__ = [
    "MODELS",
    "T4_COMPUTE_CAPABILITY",
    "T4_MIN_VRAM_GB",
    "BudgetError",
    "Envelope",
    "EnvelopeError",
    "FakeBackend",
    "GenerationConfig",
    "GenerationResult",
    "LLMBackend",
    "Message",
    "ModelSpec",
    "PreflightResult",
    "StopReason",
    "TokenBudget",
    "Usage",
    "extract_json_candidates",
    "get_model",
    "get_model_by_name",
    "looks_like_final_answer",
    "make_backend",
    "parse_envelope",
    "preflight_gpu",
    "repair_json",
    "strip_fences",
]


def make_backend(kind: str = "transformers", **kwargs: Any) -> LLMBackend:
    """Construct a backend by name.

    `kind="fake"` is the test/`--dry-run` path; `kind="transformers"` imports the
    GPU stack only when it is actually used.
    """
    if kind == "fake":
        return FakeBackend(**kwargs)
    if kind == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(**kwargs)
    raise ValueError(f"unknown backend {kind!r}; expected 'transformers' or 'fake'")
