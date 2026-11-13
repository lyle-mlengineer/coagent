"""The OpenAI-compatible backend: streaming, usage, errors, and the egress gate.

These run against a real HTTP server on a loopback port, so the request body and
the SSE parsing are exercised for real — no network, no key, no GPU.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from coagent.llm.base import GenerationConfig, Message, StopReason
from coagent.llm.openai_backend import (
    PROVIDERS,
    OpenAICompatibleBackend,
    ProviderConfig,
    ProviderError,
    resolve_provider,
)

# ------------------------------------------------------------------ mock server


class _Server:
    """A one-shot OpenAI-compatible server whose reply the test controls."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.body: dict[str, Any] = {}
        self.chunks: list[str] = []
        self.stream_done = True
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers["content-length"])
                outer.requests.append(
                    {
                        "body": json.loads(self.rfile.read(length)),
                        "auth": self.headers.get("Authorization"),
                    }
                )
                if outer.status != 200:
                    payload = json.dumps(outer.body).encode()
                    self.send_response(outer.status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if outer.requests[-1]["body"].get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in outer.chunks:
                        self.wfile.write(chunk.encode())
                    if outer.stream_done:
                        self.wfile.write(b"data: [DONE]\n\n")
                    return
                payload = json.dumps(outer.body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def server():
    srv = _Server()
    try:
        yield srv
    finally:
        srv.close()


def _backend(server: _Server, **kwargs: Any) -> OpenAICompatibleBackend:
    cfg = ProviderConfig(
        base_url=server.base_url,
        model="mock-coder",
        api_key=kwargs.pop("api_key", "sk-test"),
        name=kwargs.pop("name", "custom"),
        is_remote=False,
        **kwargs,
    )
    return OpenAICompatibleBackend(cfg)


def _sse(content: str, *, finish: str | None = None, usage: dict | None = None) -> str:
    doc: dict[str, Any] = {
        "model": "mock-coder",
        "choices": [{"delta": ({"content": content} if content else {}), "finish_reason": finish}],
    }
    if usage:
        doc["usage"] = usage
    return f"data: {json.dumps(doc)}\n\n"


# ------------------------------------------------------------------ identity


def test_model_id_names_the_provider_and_model(server: _Server) -> None:
    b = _backend(server, name="groq")
    assert b.model_id == "groq:mock-coder"


def test_remote_vs_local_detection() -> None:
    remote = resolve_provider(provider="groq", environ={"GROQ_API_KEY": "k"})
    assert remote.is_remote is True
    local = resolve_provider(provider="ollama", environ={})
    assert local.is_remote is False


# ------------------------------------------------------------------ non-stream


def test_non_streaming_returns_text_and_usage(server: _Server) -> None:
    server.body = {
        "model": "mock-coder",
        "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
    }
    result = _backend(server).generate([Message("user", "ping")])

    assert result.text == "pong"
    assert result.stop_reason is StopReason.STOP
    assert result.usage.prompt_tokens == 7
    assert result.usage.completion_tokens == 2


def test_request_body_carries_model_and_sampling(server: _Server) -> None:
    server.body = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    cfg = GenerationConfig(max_new_tokens=64, temperature=0.2, top_p=0.9, stop=("</s>",))
    _backend(server).generate([Message("system", "s"), Message("user", "u")], cfg)

    sent = server.requests[-1]["body"]
    assert sent["model"] == "mock-coder"
    assert sent["max_tokens"] == 64
    assert sent["temperature"] == 0.2
    assert sent["stop"] == ["</s>"]
    assert sent["stream"] is False
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_bearer_token_is_sent(server: _Server) -> None:
    server.body = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    _backend(server, api_key="sk-secret").generate([Message("user", "u")])
    assert server.requests[-1]["auth"] == "Bearer sk-secret"


def test_length_finish_reason_maps_to_truncation(server: _Server) -> None:
    server.body = {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}
    result = _backend(server).generate([Message("user", "u")])
    assert result.stop_reason is StopReason.LENGTH
    assert result.truncated is True


# ------------------------------------------------------------------ streaming


def test_streaming_emits_tokens_in_order(server: _Server) -> None:
    server.chunks = [
        _sse("Hello"),
        _sse(", "),
        _sse("world"),
        _sse("", finish="stop", usage={"prompt_tokens": 11, "completion_tokens": 3}),
    ]
    seen: list[str] = []
    result = _backend(server).generate([Message("user", "u")], on_token=seen.append)

    assert seen == ["Hello", ", ", "world"]
    assert result.text == "Hello, world"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 3


def test_streaming_asks_for_usage_and_tolerates_keepalives(server: _Server) -> None:
    server.chunks = [": keep-alive\n\n", _sse("hi"), _sse("", finish="stop")]
    result = _backend(server).generate([Message("user", "u")], on_token=lambda _t: None)

    assert result.text == "hi"
    assert server.requests[-1]["body"]["stream"] is True
    assert server.requests[-1]["body"]["stream_options"] == {"include_usage": True}


def test_streaming_cancel_returns_cancelled(server: _Server) -> None:
    server.chunks = [_sse(f"t{i} ") for i in range(20)]
    state = {"n": 0}

    def cancel() -> bool:
        state["n"] += 1
        return state["n"] > 2

    seen: list[str] = []
    result = _backend(server).generate(
        [Message("user", "u")], on_token=seen.append, cancel=cancel
    )
    assert result.stop_reason is StopReason.CANCELLED
    assert len(seen) < 20  # it stopped early rather than draining the stream


def test_stream_without_finish_reason_at_max_tokens_is_length(server: _Server) -> None:
    server.chunks = [_sse("aaa"), _sse("bbb")]  # no finish_reason, no [DONE] sentinel
    server.stream_done = False
    result = _backend(server).generate(
        [Message("user", "u")], GenerationConfig(max_new_tokens=6), on_token=lambda _t: None
    )
    assert result.stop_reason is StopReason.LENGTH


# --------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "needle"),
    [
        (401, "API key"),
        (404, "base_url"),
        (429, "rate limited"),
    ],
)
def test_http_errors_are_actionable(server: _Server, status: int, needle: str) -> None:
    server.status = status
    server.body = {"error": {"message": "nope"}}
    with pytest.raises(ProviderError) as err:
        _backend(server, name="groq").generate([Message("user", "u")])
    assert needle in str(err.value)
    assert str(status) in str(err.value)


def test_unreachable_endpoint_is_actionable() -> None:
    cfg = ProviderConfig(
        base_url="http://127.0.0.1:9/v1", model="m", name="custom", timeout_s=2
    )
    with pytest.raises(ProviderError, match="cannot reach"):
        OpenAICompatibleBackend(cfg).generate([Message("user", "u")])


def test_no_choices_is_an_error(server: _Server) -> None:
    server.body = {"choices": []}
    with pytest.raises(ProviderError, match="no choices"):
        _backend(server).generate([Message("user", "u")])


# ---------------------------------------------------------- provider resolution


def test_a_preset_supplies_url_and_model() -> None:
    cfg = resolve_provider(provider="groq", environ={"GROQ_API_KEY": "k"})
    assert cfg.name == "groq"
    assert cfg.base_url == PROVIDERS["groq"].base_url
    assert cfg.model == PROVIDERS["groq"].default_model
    assert cfg.api_key == "k"
    assert cfg.chat_url.endswith("/chat/completions")


def test_explicit_values_override_the_preset() -> None:
    cfg = resolve_provider(
        provider="groq",
        base_url="https://example.test/v1",
        model="my/model",
        environ={"GROQ_API_KEY": "k"},
    )
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.model == "my/model"


def test_missing_key_names_the_variable_to_set() -> None:
    with pytest.raises(ProviderError) as err:
        resolve_provider(provider="groq", environ={})
    msg = str(err.value)
    assert "GROQ_API_KEY" in msg
    assert "export" in msg


def test_local_providers_need_no_key() -> None:
    cfg = resolve_provider(provider="ollama", environ={})
    assert cfg.api_key == ""
    assert cfg.is_remote is False


def test_unknown_provider_without_base_url_is_rejected() -> None:
    with pytest.raises(ProviderError, match="no base_url"):
        resolve_provider(provider="does-not-exist", environ={})


def test_missing_model_is_rejected() -> None:
    with pytest.raises(ProviderError, match="no model name"):
        resolve_provider(base_url="https://x.test/v1", environ={"X": "1"})


# ------------------------------------------------------------- the egress gate
#
# PRD FR-MDL-7: model execution makes no network calls in the shipped config.
# Provider mode is opt-in, and when it is on the agent says so.


def test_the_shipped_default_makes_no_egress() -> None:
    from coagent.config import Config

    assert Config().model.provider.enabled is False


def test_backend_from_config_uses_the_provider_only_when_enabled(monkeypatch) -> None:
    """No torch: the local branch is stubbed, so this asserts the *choice*."""
    import coagent.llm as llm_pkg
    from coagent.config import Config

    calls: list[str] = []

    def fake_make_backend(kind: str, **kwargs: object) -> object:
        calls.append(kind)
        return object()

    monkeypatch.setattr(llm_pkg, "make_backend", fake_make_backend)

    # Default config -> local transformers backend, despite there being no GPU.
    llm_pkg.backend_from_config(Config())
    assert calls == ["transformers"]

    # Provider enabled -> the OpenAI-compatible backend.
    cfg = Config()
    cfg.model.provider.enabled = True
    cfg.model.provider.name = "ollama"  # local provider: needs no key
    monkeypatch.setenv("OLLAMA_API_KEY", "")
    llm_pkg.backend_from_config(cfg)
    assert calls == ["transformers", "openai"]


def test_a_provider_error_is_raised_before_any_work(monkeypatch) -> None:
    """A missing key fails at construction, not after a clone."""
    from coagent.config import Config
    from coagent.llm import backend_from_config
    from coagent.llm.openai_backend import ProviderError

    cfg = Config()
    cfg.model.provider.enabled = True
    cfg.model.provider.name = "groq"  # needs GROQ_API_KEY
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        backend_from_config(cfg)


def test_t4_context_cap_does_not_apply_to_a_provider() -> None:
    """The 12288 cap fits a T4 KV cache; a hosted model has no such envelope."""
    from coagent.config import ConfigError, ModelConfig, ModelProviderConfig

    with pytest.raises(ConfigError, match="hard cap"):
        ModelConfig(max_context=99999)

    hosted = ModelConfig(
        max_context=99999, provider=ModelProviderConfig(enabled=True, name="groq")
    )
    assert hosted.max_context == 99999


# --------------------------------------------------- end to end through the loop


def test_a_provider_drives_the_whole_agent_loop_without_a_gpu(tmp_path) -> None:
    """The point of this feature: exercise the loop on a laptop, no T4, no torch."""
    import subprocess
    import sys

    from coagent.agent.loop import coAgent
    from coagent.config import load_config, write_default_config

    # A scripted OpenAI-compatible model: survey, plan, read, edit, report.
    script = [
        "Looking at the repo.",
        "# add a hello function\n- [ ] make the change\n- [ ] run the checks",
        '```json\n{"tool": "read_file", "args": {"path": "greet.py"}}\n```',
        '```json\n{"tool": "edit_file", "args": {"path": "greet.py", '
        '"search": "VALUE = 1", "replace": "VALUE = 2"}}\n```',
        "Report.",
    ]
    state = {"i": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            length = int(self.headers["content-length"])
            body = json.loads(self.rfile.read(length))
            i = min(state["i"], len(script) - 1)
            state["i"] += 1
            text = script[i]
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk = {
                    "model": body["model"],
                    "choices": [{"delta": {"content": text}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                return
            doc = {
                "model": body["model"],
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            payload = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    try:
        root = tmp_path / "proj"
        root.mkdir()

        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

        git("init", "-q", "-b", "main")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "T")
        (root / "greet.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / ".gitignore").write_text(
            ".coagent/*\n!.coagent/config.toml\n", encoding="utf-8"
        )
        git("add", "-A")
        git("commit", "-qm", "init")

        path = write_default_config(root)
        text = path.read_text(encoding="utf-8")
        text = text.replace('author_name = "coAgent"', 'author_name = "T"')
        text = text.replace('author_email = "coagent@localhost"', 'author_email = "t@example.com"')
        text = text.replace('mode = "push"', 'mode = "bundle"')
        text = text.replace(
            'commands = ["ruff check .", "pytest -q"]',
            "commands = [" + json.dumps(f'"{sys.executable}" -c "import greet"') + "]",
        )
        text += (
            "\n[model.provider]\n"
            "enabled = true\n"
            'name = "custom"\n'
            f'base_url = "http://127.0.0.1:{port}/v1"\n'
            'model = "mock-coder"\n'
            'api_key_env = ""\n'
            "max_context = 32768\n"
        )
        path.write_text(text, encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "config")

        events: list[tuple[str, dict]] = []
        agent = coAgent(
            root,
            config=load_config(path),
            auto=True,
            interactive=False,
            on_event=lambda kind, data: events.append((kind, data)),
        )
        try:
            assert agent.model_id == "custom:mock-coder"
            # A loopback endpoint is not egress: no notice, correctly.
            notices = [d for k, d in events if k == "egress_notice"]
            assert not notices, "a 127.0.0.1 provider must not claim egress"

            result = agent.run("add a hello function")
        finally:
            agent.close()

        assert result.done is True, result.blocked_reason
        assert result.stopped_by == "complete"
        assert "greet.py" in result.files_changed
        assert (root / "greet.py").read_text(encoding="utf-8").strip() == "VALUE = 2"
    finally:
        httpd.shutdown()


def test_a_remote_provider_announces_egress(monkeypatch) -> None:
    """Switching to a hosted model must be visible, not silent (FR-MDL-7)."""
    import coagent.llm as llm_pkg
    from coagent.config import Config
    from coagent.llm.openai_backend import OpenAICompatibleBackend

    cfg = Config()
    cfg.model.provider.enabled = True
    cfg.model.provider.name = "groq"
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")

    backend = llm_pkg.backend_from_config(cfg)
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.provider.is_remote is True
    assert backend.provider.base_url.startswith("https://api.groq.com")
