"""The model interface (DESIGN §4).

`generate(messages) -> text` behind one interface, so the loop never knows which
backend it is talking to. That is what keeps the loop testable on a laptop with
a scripted backend while production runs a 4-bit Qwen on a T4.

Two rules this module enforces because a 7B model cannot be trusted with them:

* **No silent truncation.** A request over the token budget is rejected before
  generation (PRD D2), counted with the model's own tokenizer.
* **No inference egress.** Nothing in this layer opens a socket. The only
  outbound traffic in the product is the model download and git (FR-MDL-7).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BudgetError",
    "FakeBackend",
    "GenerationConfig",
    "GenerationResult",
    "LLMBackend",
    "Message",
    "StopReason",
    "TokenBudget",
    "Usage",
]


class BudgetError(RuntimeError):
    """The request exceeds the context budget. Raised before generation."""


class StopReason(str, enum.Enum):
    STOP = "stop"  # model emitted EOS
    LENGTH = "length"  # hit max_new_tokens (may be truncated — say so)
    CANCELLED = "cancelled"  # user pressed stop
    ERROR = "error"


@dataclass(slots=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 0
    stop: tuple[str, ...] = ()


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class GenerationResult:
    text: str
    stop_reason: StopReason = StopReason.STOP
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    @property
    def truncated(self) -> bool:
        return self.stop_reason is StopReason.LENGTH


@dataclass(slots=True)
class TokenBudget:
    """Context arithmetic against a real tokenizer (PRD D2, FR-CTX-6).

    `count` is injected so this type works with HF tokenizers, a fake counter in
    tests, and a character heuristic only as an explicit last resort.
    """

    max_context: int
    count: Callable[[str], int]
    reserve_output: int = 1500

    @property
    def input_limit(self) -> int:
        return max(0, self.max_context - self.reserve_output)

    def count_messages(self, messages: Iterable[Message]) -> int:
        # Chat templates add role markers and separators; ~4 tokens per turn is
        # the usual overhead and is included so the estimate is not optimistic.
        return sum(self.count(m.content) for m in messages) + 4 * len(list(messages))

    def check(self, messages: list[Message], *, max_new_tokens: int) -> int:
        """Return the input token count, or raise `BudgetError`."""
        used = self.count_messages(messages)
        cap = self.max_context - min(max_new_tokens, self.reserve_output)
        if used > cap:
            raise BudgetError(
                f"request is {used} tokens but the budget allows {cap} "
                f"(max_context={self.max_context}, reserved output={min(max_new_tokens, self.reserve_output)}). "
                "Evict observations or shorten the retrieval set — the request was not sent."
            )
        return used


class LLMBackend(ABC):
    """What the loop requires of any backend."""

    name: str = ""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable `model@quant` used in trailers and the status line."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Tokenizer truth for budget arithmetic (FR-CTX-6)."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        *,
        on_token: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        """Generate a completion. `on_token` streams; `cancel` polls a stop flag."""

    def load(self) -> None:  # noqa: B027 - optional hook, not every backend needs it
        """Load weights. Backends that need no load (tests) do nothing."""

    def unload(self) -> None:  # noqa: B027 - optional hook
        """Release the GPU. Required before `verify.gpu_commands` (DESIGN §4.6)."""

    def close(self) -> None:
        self.unload()


class FakeBackend(LLMBackend):
    """A scripted backend for tests and for `--dry-run`.

    Produces the queued responses in order; when the queue runs dry it repeats
    the last one, which is what makes loop-detection tests deterministic.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        model_id: str = "fake/coder@none",
        tokens_per_char: float = 0.25,
    ) -> None:
        self._responses = list(responses or ['```json\n{"tool": "done", "args": {}}\n```'])
        self._index = 0
        self._model_id = model_id
        self._tpc = tokens_per_char
        self.loaded = False
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text) * self._tpc))

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def queue(self, *responses: str) -> None:
        self._responses.extend(responses)

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        *,
        on_token: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()
        index = min(self._index, len(self._responses) - 1)
        text = self._responses[index]
        self._index += 1
        self.calls.append(
            {
                "messages": [m.to_dict() for m in messages],
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
            }
        )

        if on_token is not None:
            # Stream in small chunks so a UI test sees multiple updates.
            for i in range(0, len(text), 16):
                if cancel is not None and cancel():
                    emitted = text[:i]
                    return GenerationResult(
                        text=emitted,
                        stop_reason=StopReason.CANCELLED,
                        usage=Usage(self.count_tokens("".join(m.content for m in messages)), self.count_tokens(emitted)),
                        model=self._model_id,
                    )
                on_token(text[i : i + 16])

        stop = StopReason.LENGTH if len(text) > cfg.max_new_tokens else StopReason.STOP
        return GenerationResult(
            text=text,
            stop_reason=stop,
            usage=Usage(
                prompt_tokens=self.count_tokens("".join(m.content for m in messages)),
                completion_tokens=self.count_tokens(text),
            ),
            model=self._model_id,
        )
