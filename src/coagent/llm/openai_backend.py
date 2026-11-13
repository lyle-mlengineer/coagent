"""An OpenAI-compatible backend, for running coAgent against a hosted model.

Why this exists: the loop, the tools, the safety layer and the git plumbing are
all testable on a laptop, but exercising them against a real model used to
require a T4 and a 4 GB download. Any provider that speaks the OpenAI
`/chat/completions` shape — Groq, OpenRouter, NVIDIA NIM, Together, Fireworks,
LM Studio, Ollama, vLLM — can drive the same loop with a key or a local URL.

This is **opt-in and never a default**. PRD FR-MDL-7 ("No inference egress") is
a v1 guarantee: with the shipped config the model runs locally and makes no
network calls. Choosing a provider moves your source code to that provider, so
the agent says so out loud at session start rather than letting it happen
quietly (see `EgressNotice` in the loop).

Only the standard library is used. `urllib` is enough for JSON POST and a line
-delimited SSE stream, and it keeps `pip install co-agent-coder` free of an
HTTP client. Errors are mapped to actionable messages because "HTTP 401" alone
is the least useful thing a tool can say.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .base import (
    GenerationConfig,
    GenerationResult,
    LLMBackend,
    Message,
    StopReason,
    Usage,
)

__all__ = [
    "OpenAICompatibleBackend",
    "ProviderConfig",
    "PROVIDERS",
    "ProviderError",
    "ProviderSpec",
    "resolve_provider",
]

#: Hosted APIs rarely publish a tokenizer here; this is a safe default window.
DEFAULT_MAX_CONTEXT = 32768


class ProviderError(RuntimeError):
    """A hosted provider call failed, with a message worth showing a user."""


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """A known provider: its endpoint, key env var, and doc link."""

    name: str
    base_url: str
    api_key_env: str
    default_model: str
    notes: str = ""
    needs_key: bool = True


#: Presets so the common cases are one word. Anything else can be spelled out
#: with an explicit base_url, which is what makes vLLM/Ollama/local work.
PROVIDERS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="qwen/qwen3-32b",
        notes="fast and cheap; generous free tier",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="qwen/qwen3-coder",
        notes="one key for many vendors",
    ),
    "nvidia": ProviderSpec(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        default_model="qwen/qwen2.5-coder-32b-instruct",
        notes="NVIDIA NIM; free developer credits",
    ),
    "together": ProviderSpec(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        default_model="Qwen/Qwen2.5-Coder-32B-Instruct",
    ),
    "fireworks": ProviderSpec(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        default_model="accounts/fireworks/models/qwen2p5-coder-32b-instruct",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key_env="",
        default_model="qwen2.5-coder:7b",
        notes="fully local; no key, no egress off your machine",
        needs_key=False,
    ),
    "lmstudio": ProviderSpec(
        name="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key_env="",
        default_model="qwen2.5-coder-7b-instruct",
        notes="local app server; no key",
        needs_key=False,
    ),
    "vllm": ProviderSpec(
        name="vllm",
        base_url="http://localhost:8000/v1",
        api_key_env="",
        default_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        notes="self-hosted vLLM; no key",
        needs_key=False,
    ),
}


@dataclass(slots=True)
class ProviderConfig:
    """The resolved `[model.provider]` settings, ready to construct a backend."""

    base_url: str
    model: str
    api_key: str = ""
    name: str = "custom"
    max_context: int = 32768
    timeout_s: int = 120
    extra_headers: dict[str, str] = field(default_factory=dict)
    #: True when traffic leaves the machine (i.e. not a loopback URL).
    is_remote: bool = True

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


def resolve_provider(
    *,
    provider: str = "",
    base_url: str = "",
    model: str = "",
    api_key_env: str = "",
    api_key: str = "",
    max_context: int | None = None,
    timeout_s: int | None = None,
    extra_headers: dict[str, str] | None = None,
    environ: dict[str, str] | None = None,
) -> ProviderConfig:
    """Turn config/env into a `ProviderConfig`, or raise with the exact fix.

    Precedence is explicit config > preset > error. A key is read from the
    named environment variable (never the config file, which is committed).
    """
    import os

    env = environ if environ is not None else os.environ
    name = (provider or "").strip().lower()
    spec = PROVIDERS.get(name)

    if not base_url and spec is not None:
        base_url = spec.base_url
    if not model and spec is not None:
        model = spec.default_model
    if spec is not None and not api_key_env:
        api_key_env = spec.api_key_env

    if not base_url:
        known = ", ".join(sorted(PROVIDERS))
        raise ProviderError(
            "model.provider is enabled but no base_url is set. Name a known provider "
            f"({known}) or set [model.provider] base_url for your endpoint."
        )
    if not model:
        raise ProviderError(
            "model.provider is enabled but no model name is set. Set "
            "[model.provider] model, e.g. \"qwen/qwen3-coder\"."
        )

    lowered = base_url.lower()
    is_remote = not any(
        host in lowered
        for host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")
    )

    key = api_key or (env.get(api_key_env, "") if api_key_env else "")
    # A loopback endpoint is somebody's local server (Ollama, vLLM, LM Studio): it
    # has no key to give, and demanding one would make the easiest way to try
    # coAgent fail on a technicality. Only a real remote host requires a key.
    needs_key = (spec.needs_key if spec is not None else True) and is_remote
    if needs_key and not key:
        where = f"${api_key_env}" if api_key_env else "an API key"
        raise ProviderError(
            f"provider {name or base_url!r} needs an API key, but {where} is not set. "
            f"Export it first:  export {api_key_env or 'PROVIDER_API_KEY'}=…  "
            "(the key is read from the environment, never from the committed config)."
        )
    if not is_remote and needs_key and not key:
        # Unreachable while `needs_key` is gated on `is_remote`; kept so a future
        # auth-proxy case (local URL that really does want a token) is obvious.
        key = "not-needed"

    return ProviderConfig(
        base_url=base_url,
        model=model,
        api_key=key,
        name=name or "custom",
        max_context=max_context if max_context is not None else DEFAULT_MAX_CONTEXT,
        timeout_s=timeout_s if timeout_s is not None else 120,
        extra_headers=dict(extra_headers or {}),
        is_remote=is_remote,
    )


class OpenAICompatibleBackend(LLMBackend):
    """`/chat/completions` over urllib, streaming or not."""

    name = "openai-compatible"

    def __init__(self, provider: ProviderConfig) -> None:
        self.provider = provider
        self._cancelled = threading.Event()

    # ------------------------------------------------------------- identity

    @property
    def model_id(self) -> str:
        """`<provider>:<model>` — names the egress in every commit trailer."""
        return f"{self.provider.name}:{self.provider.model}"

    @property
    def max_context(self) -> int:
        return self.provider.max_context

    def load(self) -> None:  # noqa: B027 - nothing to load for a remote model
        """No weights; kept so callers need no special case."""

    def unload(self) -> None:  # noqa: B027
        """No VRAM held."""

    # ------------------------------------------------------------ tokenizer
    def count_tokens(self, text: str) -> int:
        """A conservative estimate: hosted APIs have no public tokenizer here.

        ~3.5 chars/token is deliberately pessimistic (over-estimating is the safe
        bias for a context cap), and it keeps the budget arithmetic working
        without a per-provider tokenizer dependency.
        """
        if not text:
            return 0
        return max(1, len(text) // 3)

    # ------------------------------------------------------------ generation

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.provider.api_key:
            headers["Authorization"] = f"Bearer {self.provider.api_key}"
        # OpenRouter wants these; harmless elsewhere.
        headers.setdefault("HTTP-Referer", "https://github.com/lyle-mlengineer/coagent")
        headers.setdefault("X-Title", "coAgent")
        headers.update(self.provider.extra_headers)
        return headers

    def _payload(
        self, messages: list[Message], cfg: GenerationConfig, *, stream: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.provider.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_new_tokens,
            "stream": stream,
        }
        if stream:
            # Ask for the usage block on the final chunk where supported.
            body["stream_options"] = {"include_usage": True}
        if cfg.stop:
            body["stop"] = list(cfg.stop)
        if cfg.seed:
            body["seed"] = cfg.seed
        return body

    def _open(self, body: dict[str, Any]) -> Any:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.provider.chat_url, data=data, headers=self._headers(), method="POST"
        )
        try:
            return urllib.request.urlopen(req, timeout=self.provider.timeout_s)
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"cannot reach {self.provider.name} at {self.provider.chat_url}: "
                f"{exc.reason}. Check the base_url and your network."
            ) from exc

    def _http_error(self, exc: urllib.error.HTTPError) -> ProviderError:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:600]
        except Exception:  # noqa: BLE001 - the message must survive a bad read
            body = ""
        hint = {
            401: f"the API key was rejected. Check ${PROVIDERS[self.provider.name].api_key_env if self.provider.name in PROVIDERS and PROVIDERS[self.provider.name].api_key_env else 'PROVIDER_API_KEY'}.",
            403: "the key is valid but not allowed to use this model.",
            404: f"no model {self.provider.model!r} at this endpoint, or the base_url is wrong.",
            422: "the request was rejected — usually an unknown model name or parameter.",
            429: "rate limited. Wait, or lower agent.max_steps.",
        }.get(exc.code, "")
        detail = f"\n{body}" if body else ""
        return ProviderError(
            f"{self.provider.name} returned HTTP {exc.code}. {hint}{detail}".strip()
        )

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        *,
        on_token: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()
        self._cancelled.clear()
        stream = on_token is not None
        started = time.monotonic()

        resp = self._open(self._payload(messages, cfg, stream=stream))
        try:
            if stream:
                return self._consume_stream(resp, on_token, cancel, started, cfg)
            return self._consume_once(resp, started, messages)
        finally:
            resp.close()

    # ----------------------------------------------------------------- paths

    def _consume_once(
        self, resp: Any, started: float, messages: list[Message]
    ) -> GenerationResult:
        raw = resp.read().decode("utf-8", errors="replace")
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.provider.name} returned non-JSON: {raw[:300]}") from exc

        choices = doc.get("choices") or []
        if not choices:
            raise ProviderError(
                f"{self.provider.name} returned no choices: {json.dumps(doc)[:300]}"
            )
        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason") or "stop"
        usage = self._usage(doc)
        if not usage.total:
            # Some endpoints omit usage; estimate so the session total is not zero.
            usage = Usage(
                prompt_tokens=sum(self.count_tokens(m.content) for m in messages),
                completion_tokens=self.count_tokens(text),
            )
        return GenerationResult(
            text=text,
            stop_reason=self._stop_reason(finish),
            usage=usage,
            model=doc.get("model") or self.provider.model,
        )

    def _consume_stream(
        self,
        resp: Any,
        on_token: Callable[[str], None],
        cancel: Callable[[], bool] | None,
        started: float,
        cfg: GenerationConfig,
    ) -> GenerationResult:
        pieces: list[str] = []
        finish = "stop"
        usage = Usage()
        model = self.provider.model

        for line in self._iter_lines(resp):
            if self._cancelled.is_set() or (cancel is not None and cancel()):
                return GenerationResult(
                    text="".join(pieces),
                    stop_reason=StopReason.CANCELLED,
                    usage=usage,
                    model=model,
                )
            line = line.strip()
            if not line or line.startswith(":"):
                continue  # SSE comment / keep-alive
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial or provider-specific chunk; keep streaming

            model = doc.get("model") or model
            if doc.get("usage"):
                usage = self._usage(doc)
            for choice in doc.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    pieces.append(piece)
                    on_token(piece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]

        text = "".join(pieces)
        return GenerationResult(
            text=text,
            stop_reason=self._stop_reason(finish, at_limit=len(text) >= cfg.max_new_tokens),
            usage=usage,
            model=model,
        )

    @staticmethod
    def _iter_lines(resp: Any) -> Iterator[str]:
        for raw in resp:
            yield raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    @staticmethod
    def _usage(doc: dict[str, Any]) -> Usage:
        u = doc.get("usage") or {}
        return Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
        )

    @staticmethod
    def _stop_reason(finish: str, *, at_limit: bool = False) -> StopReason:
        if finish == "length":
            return StopReason.LENGTH
        if finish in ("content_filter", "error"):
            return StopReason.ERROR
        # No finish_reason but the text hit max_tokens: honest truncation.
        return StopReason.LENGTH if at_limit else StopReason.STOP

    def close(self) -> None:
        self._cancelled.set()
