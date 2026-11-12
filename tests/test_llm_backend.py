"""Backend, budget, and preflight tests — all CPU-only (no torch required)."""

from __future__ import annotations

import pytest

from coagent.llm import (
    MODELS,
    BudgetError,
    FakeBackend,
    GenerationConfig,
    Message,
    StopReason,
    TokenBudget,
    get_model,
    get_model_by_name,
    make_backend,
    preflight_gpu,
)

# ------------------------------------------------------------------- budget


def test_budget_accepts_a_small_request() -> None:
    budget = TokenBudget(max_context=12288, count=lambda t: len(t.split()), reserve_output=1500)
    messages = [Message("system", "you are a coder"), Message("user", "fix the test")]
    used = budget.check(messages, max_new_tokens=1024)
    assert used > 0
    assert used <= budget.input_limit


def test_budget_rejects_an_over_cap_request() -> None:
    budget = TokenBudget(max_context=100, count=lambda t: len(t.split()), reserve_output=50)
    messages = [Message("user", "word " * 200)]
    with pytest.raises(BudgetError) as exc:
        budget.check(messages, max_new_tokens=40)
    assert "was not sent" in str(exc.value)


def test_budget_error_names_both_numbers() -> None:
    budget = TokenBudget(max_context=100, count=lambda t: len(t), reserve_output=40)
    with pytest.raises(BudgetError) as exc:
        budget.check([Message("user", "x" * 500)], max_new_tokens=30)
    assert "504 tokens" in str(exc.value)  # 500 + 4 tokens of chat-template overhead
    assert "budget allows 70" in str(exc.value)


def test_nowhere_near_the_cap_is_fine() -> None:
    budget = TokenBudget(max_context=1000, count=lambda t: 1, reserve_output=100)
    assert budget.check([Message("user", "hi")], max_new_tokens=900) == 5


# -------------------------------------------------------------------- fake


def test_fake_backend_returns_queued_responses() -> None:
    fake = FakeBackend(['{"tool":"a"}', '{"tool":"b"}'])
    msgs = [Message("user", "go")]
    assert fake.generate(msgs).text == '{"tool":"a"}'
    assert fake.generate(msgs).text == '{"tool":"b"}'


def test_fake_backend_repeats_last_response() -> None:
    """Deterministic repetition is what makes loop-detection tests meaningful."""
    fake = FakeBackend(['{"tool":"a"}'])
    msgs = [Message("user", "go")]
    for _ in range(5):
        assert fake.generate(msgs).text == '{"tool":"a"}'


def test_fake_backend_streams() -> None:
    fake = FakeBackend(["x" * 100])
    chunks: list[str] = []
    result = fake.generate([Message("user", "go")], on_token=chunks.append)
    assert len(chunks) > 1
    assert "".join(chunks) == result.text


def test_fake_backend_cancel_stops_streaming() -> None:
    fake = FakeBackend(["y" * 500])
    state = {"n": 0}

    def cancel() -> bool:
        state["n"] += 1
        return state["n"] > 2

    result = fake.generate([Message("user", "go")], on_token=lambda _c: None, cancel=cancel)
    assert result.stop_reason is StopReason.CANCELLED
    assert len(result.text) < 500


def test_fake_backend_reports_usage() -> None:
    fake = FakeBackend(["hello world"])
    result = fake.generate([Message("user", "go")])
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total == result.usage.prompt_tokens + result.usage.completion_tokens


def test_fake_backend_records_calls() -> None:
    fake = FakeBackend()
    fake.generate([Message("system", "s"), Message("user", "u")], GenerationConfig(temperature=0.7))
    assert len(fake.calls) == 1
    assert fake.calls[0]["temperature"] == 0.7
    assert fake.calls[0]["messages"][0] == {"role": "system", "content": "s"}


def test_fake_backend_load_unload() -> None:
    fake = FakeBackend()
    assert not fake.loaded
    fake.load()
    assert fake.loaded
    fake.unload()
    assert not fake.loaded


def test_make_backend_fake() -> None:
    assert isinstance(make_backend("fake"), FakeBackend)


def test_make_backend_unknown() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("gpt5")


def test_make_backend_transformers_is_lazy() -> None:
    """Constructing the real backend must not require torch to be installed."""
    backend = make_backend("transformers", model_key="qwen-coder-3b")
    assert backend.model_id == "Qwen2.5-Coder-3B-Instruct@nf4"
    assert not backend.loaded


# ---------------------------------------------------------------- registry


def test_models_are_the_two_validated_configs() -> None:
    assert set(MODELS) == {"qwen-coder-7b", "qwen-coder-3b"}
    seven = get_model("qwen-coder-7b")
    three = get_model("qwen-coder-3b")
    assert seven.max_context == 12288
    assert three.max_context == 8192
    # The fallback must never be bigger (PRD D6).
    assert three.vram_gb < seven.vram_gb
    assert three.max_context < seven.max_context


def test_trailer_id_format() -> None:
    assert get_model("qwen-coder-7b").trailer_id == "Qwen2.5-Coder-7B-Instruct@nf4"


def test_get_model_by_name() -> None:
    assert get_model_by_name("Qwen/Qwen2.5-Coder-7B-Instruct") is not None
    assert get_model_by_name("gpt-4") is None


def test_unknown_model_key() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        get_model("llama-70b")


# --------------------------------------------------------------- preflight


def test_preflight_without_cuda_returns_a_reason() -> None:
    result = preflight_gpu()
    if result.ok:
        pytest.skip("this machine has a usable T4")
    assert result.reason
    assert "T4" in result.reason or "torch" in result.reason
    assert result.describe() in {"no CUDA device detected"} or "GB" in result.describe()


def test_preflight_never_raises() -> None:
    # Whatever the hardware, a refusal must come back as a value, not an exception.
    preflight_gpu()
