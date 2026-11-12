"""The local `transformers` backend (DESIGN §4.2, §4.5).

Pinned to what a Turing T4 can actually do:

* **NF4 4-bit weights, fp16 compute.** T4 (SM 7.5) has no bf16.
* **`attn_implementation="sdpa"`.** FlashAttention-2 needs Ampere+.
* **No `torch.compile`.** Compile time exceeds most sessions' savings.
* **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set before torch import.**

Everything torch-related is imported inside methods so `import coagent` stays
light and the test suite runs without a GPU.
"""

from __future__ import annotations

import contextlib
import gc
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import (
    GenerationConfig,
    GenerationResult,
    LLMBackend,
    Message,
    StopReason,
    Usage,
)
from .registry import MODELS, ModelSpec, get_model

__all__ = ["TransformersBackend", "OutOfMemoryLadder"]


def _torch():
    """Resolve torch at call time.

    Module-level accessors keep `import coagent` torch-free while still letting a
    test substitute a stub — patching a local `import` is impossible.
    """
    import torch

    return torch


def _transformers():
    """Resolve transformers at call time (same reason as `_torch`)."""
    import transformers

    return transformers


@dataclass(slots=True)
class OutOfMemoryLadder:
    """The OOM response order from DESIGN §4.5.

    The session must never be lost to an OOM, and the model must never grow.
    """

    step: int = 0
    context_reduction: float = 0.25
    max_new_tokens_reduction: float = 0.5

    def next_action(self) -> str:
        """`shrink_context`, `halve_output`, `fallback_model`, or `stop`."""
        return ("shrink_context", "halve_output", "fallback_model", "stop")[min(self.step, 3)]

    def advance(self) -> str:
        action = self.next_action()
        self.step += 1
        return action

    @property
    def exhausted(self) -> bool:
        """True once every recovery action has been tried (3 actions, then stop)."""
        return self.step >= 3


class TransformersBackend(LLMBackend):
    """A 4-bit causal LM on the local GPU."""

    name = "transformers"

    def __init__(
        self,
        model_key: str = "qwen-coder-7b",
        *,
        spec: ModelSpec | None = None,
        device_map: str | dict[str, Any] = "auto",
        trust_remote_code: bool = False,
        max_context: int | None = None,
    ) -> None:
        # Set the allocator flag before torch ever initialises CUDA.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.spec = spec or get_model(model_key)
        self._device_map = device_map
        self._trust = trust_remote_code
        self._max_context_override = max_context
        self._model: Any = None
        self._tokenizer: Any = None
        self._tokenizer_errors: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle

    @property
    def model_id(self) -> str:
        return self.spec.trailer_id

    @property
    def max_context(self) -> int:
        return self._max_context_override or self.spec.max_context

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        torch = _torch()
        transformers = _transformers()
        auto_model = transformers.AutoModelForCausalLM
        auto_tokenizer = transformers.AutoTokenizer

        quant = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # T4 has no bf16. fp16 compute is the only correct choice.
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self._tokenizer = auto_tokenizer.from_pretrained(
            self.spec.name, trust_remote_code=self._trust
        )
        self._model = auto_model.from_pretrained(
            self.spec.name,
            quantization_config=quant,
            device_map=self._device_map,
            attn_implementation="sdpa",  # flash-attn 2 needs Ampere+
            torch_dtype=torch.float16,
            trust_remote_code=self._trust,
        )
        self._model.eval()

    def unload(self) -> None:
        """Release VRAM. Required before `verify.gpu_commands` (DESIGN §4.6)."""
        if self._model is None:
            return
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            torch = _torch()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001 - unloading must never raise
            pass

    # ------------------------------------------------------------- tokenizer

    def count_tokens(self, text: str) -> int:
        """Tokenizer truth for the budget (FR-CTX-6).

        Falls back to a conservative character estimate only if the tokenizer
        cannot be loaded at all, and records that it did so.
        """
        if not text:
            return 0
        if self._tokenizer is None:
            try:
                self._tokenizer = _transformers().AutoTokenizer.from_pretrained(
                    self.spec.name, trust_remote_code=self._trust
                )
            except Exception:  # noqa: BLE001 - no network / no weights
                if "estimate" not in self._tokenizer_errors:
                    self._tokenizer_errors.add("estimate")
                # ~4 chars/token is the usual ratio; over-estimating is the safe bias.
                return max(1, len(text) // 3)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def using_estimated_tokens(self) -> bool:
        return "estimate" in self._tokenizer_errors

    # ------------------------------------------------------------ generation

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        *,
        on_token: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()
        self.load()
        torch = _torch()
        transformers = _transformers()

        assert self._tokenizer is not None
        prompt = self._tokenizer.apply_chat_template(
            [m.to_dict() for m in messages], tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        if cfg.seed:
            torch.manual_seed(cfg.seed)

        streamer = transformers.TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        stop_criteria = _make_stop_criteria(cancel) if cancel is not None else None
        stopping = transformers.StoppingCriteriaList([stop_criteria]) if stop_criteria else None

        gen_kwargs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.temperature > 0,
            "temperature": max(cfg.temperature, 1e-5),
            "top_p": cfg.top_p,
            "repetition_penalty": 1.05,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if stopping is not None:
            gen_kwargs["stopping_criteria"] = stopping

        error_holder: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(**gen_kwargs)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                error_holder.append(exc)
                with contextlib.suppress(Exception):
                    streamer.end()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        pieces: list[str] = []
        try:
            for chunk in streamer:
                if chunk:
                    pieces.append(chunk)
                    if on_token is not None:
                        on_token(chunk)
        finally:
            worker.join(timeout=1.0)

        if error_holder:
            exc = error_holder[0]
            if _is_cuda_oom(exc):
                self.unload()
                raise _oom_from(exc) from exc
            raise exc

        text = "".join(pieces)
        cancelled = cancel is not None and cancel()
        stop = StopReason.CANCELLED if cancelled else StopReason.STOP
        if not cancelled and len(pieces) and cfg.max_new_tokens and self.count_tokens(text) >= cfg.max_new_tokens:
            stop = StopReason.LENGTH

        return GenerationResult(
            text=text,
            stop_reason=stop,
            usage=Usage(
                prompt_tokens=int(inputs["input_ids"].shape[-1]),
                completion_tokens=self.count_tokens(text),
            ),
            model=self.model_id,
        )

    # ---------------------------------------------------------------- ladder

    def apply_oom_action(self, action: str, *, context: int, max_new_tokens: int) -> tuple[int, int, str]:
        """Return `(context, max_new_tokens, model_key)` for the next attempt."""
        ladder = OutOfMemoryLadder()
        if action == "shrink_context":
            return int(context * (1 - ladder.context_reduction)), max_new_tokens, self.spec.key
        if action == "halve_output":
            return context, max(64, max_new_tokens // 2), self.spec.key
        if action == "fallback_model":
            fallback = MODELS["qwen-coder-3b"]
            return fallback.max_context, max_new_tokens, fallback.key
        return context, max_new_tokens, self.spec.key


def _make_stop_criteria(cancel: Callable[[], bool]):
    """Build a real HF `StoppingCriteria` that polls the cancel flag (FR-UX-8).

    Imported lazily: this module must stay importable without transformers.
    """
    class _StopOnCancel(_transformers().StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001, ARG002
            return bool(cancel())

    return _StopOnCancel()


def _is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "outofmemory" in name
        or "out of memory" in text
        # A wrapped torch OOM often surfaces only as the class name in the message.
        or "outofmemoryerror" in text
        or "cuda error: out of memory" in text
    )


def _oom_from(exc: BaseException) -> RuntimeError:
    return RuntimeError(
        "CUDA is out of memory. coAgent unloaded the model and will retry with a "
        "smaller context; if that fails it halves the output length, then falls "
        "back to the 3B model. Your session and its checkpoints are intact."
    )
