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
    "backend_from_config",
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
    GPU stack only when it is actually used; `kind="openai"` talks to any
    OpenAI-compatible endpoint (`provider=` accepts a `ProviderConfig`).
    """
    if kind == "fake":
        return FakeBackend(**kwargs)
    if kind == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(**kwargs)
    if kind in {"openai", "openai-compatible"}:
        from .openai_backend import OpenAICompatibleBackend, ProviderConfig

        provider = kwargs.pop("provider", None)
        if not isinstance(provider, ProviderConfig):
            raise ValueError("kind='openai' needs provider=ProviderConfig(...)")
        return OpenAICompatibleBackend(provider)
    raise ValueError(
        f"unknown backend {kind!r}; expected 'transformers', 'openai', or 'fake'"
    )


def backend_from_config(cfg: Any, *, model_key: str = "qwen-coder-7b") -> LLMBackend:
    """Build the backend the config asks for.

    The provider is honoured **only** when `[model.provider] enabled = true`. With
    the shipped config this returns the local 4-bit backend, so the no-egress
    guarantee (PRD FR-MDL-7) holds by default rather than by convention.
    """
    provider_cfg = getattr(getattr(cfg, "model", None), "provider", None)
    if provider_cfg is not None and getattr(provider_cfg, "enabled", False):
        from .openai_backend import resolve_provider

        resolved = resolve_provider(
            provider=provider_cfg.name,
            base_url=provider_cfg.base_url,
            model=provider_cfg.model,
            api_key_env=provider_cfg.api_key_env,
            max_context=provider_cfg.max_context,
            timeout_s=provider_cfg.timeout_s,
        )
        return make_backend("openai", provider=resolved)
    return make_backend(
        "transformers", model_key=model_key, max_context=cfg.model.max_context
    )
