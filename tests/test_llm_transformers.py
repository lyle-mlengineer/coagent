"""TransformersBackend tests with a stubbed `transformers` module.

The real backend orchestrates HF objects; testing it honestly on a CPU box means
faking those objects (no checkpoint, no CUDA), so we exercise the orchestration
— the budget, the streaming loop, the stop handling — without a GPU.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from coagent.llm.base import GenerationConfig, Message, StopReason
from coagent.llm.transformers_backend import (
    TransformersBackend,
    _is_cuda_oom,
    _oom_from,
)


class _Inputs(dict):
    """Mimics HF `BatchEncoding`: a dict that also supports `.to(device)`."""

    def to(self, device):
        return self


class _FakeTokenizer:
    def encode(self, text: str, **kw) -> list[int]:
        return [1] * len(text)

    def apply_chat_template(self, messages, **kw) -> str:
        return "[user] " + " ".join(m["content"] for m in messages)

    def __call__(self, prompt, **kw):
        return _Inputs(input_ids=MagicMock(shape=[1, 10]), attention_mask=MagicMock())

    eos_token_id = 2


class _FakeConfig:
    pass

def _install_fake_transformers(monkeypatch) -> MagicMock:
    """Substitute the backend's `_torch` / `_transformers` accessors with fakes."""
    module = sys.modules["coagent.llm.transformers_backend"]

    torch_stub = types.SimpleNamespace(
        float16="float16",
        manual_seed=lambda *a, **k: None,
        inference_mode=lambda: __import__("contextlib").nullcontext(),
        cuda=types.SimpleNamespace(
            is_available=lambda: False, empty_cache=lambda: None, ipc_collect=lambda: None
        ),
    )

    fake = types.ModuleType("transformers")
    fake.BitsAndBytesConfig = MagicMock(side_effect=lambda **kw: types.SimpleNamespace(**kw))
    fake.AutoTokenizer = MagicMock()
    fake.AutoTokenizer.from_pretrained = MagicMock(return_value=_FakeTokenizer())

    model_mock = MagicMock()
    model_mock.device = "cpu"
    model_mock.generate.return_value = None
    fake.AutoModelForCausalLM = MagicMock()
    fake.AutoModelForCausalLM.from_pretrained = MagicMock(return_value=model_mock)

    class _Streamer:
        def __init__(self, *a, **kw):
            pass

        def __iter__(self):
            yield from ["hello ", "world"]

        def end(self):
            pass

    fake.TextIteratorStreamer = _Streamer
    fake.StoppingCriteria = object
    fake.StoppingCriteriaList = lambda items: items

    monkeypatch.setattr(module, "_torch", lambda: torch_stub)
    monkeypatch.setattr(module, "_transformers", lambda: fake)
    # The bnb gate is satisfied: these tests simulate a working GPU box.
    monkeypatch.setattr(module, "_installed_bnb_version", lambda: "0.46.1")
    return fake


def test_backend_instantiates_in_spec() -> None:
    b = TransformersBackend(model_key="qwen-coder-3b")
    assert b.model_id == "Qwen2.5-Coder-3B-Instruct@nf4"
    assert b.max_context == 8192
    assert not b.loaded


def test_load_constructs_hf_objects(monkeypatch) -> None:
    from coagent.llm.transformers_backend import TransformersBackend

    fake = _install_fake_transformers(monkeypatch)
    b = TransformersBackend(model_key="qwen-coder-3b")
    b.load()
    assert b.loaded
    fake.AutoModelForCausalLM.from_pretrained.assert_called_once()    # The bitsandbytes config must request NF4 and fp16 compute for T4.
    kwargs = fake.AutoModelForCausalLM.from_pretrained.call_args.kwargs
    assert kwargs["attn_implementation"] == "sdpa"
    # NF4 4-bit and fp16 compute are the T4 constraints (PRD D3).
    quant = kwargs["quantization_config"]
    assert quant.load_in_4bit is True
    assert quant.bnb_4bit_quant_type == "nf4"
    assert quant.bnb_4bit_compute_dtype == "float16"


def test_unload_releases_model(monkeypatch) -> None:
    from coagent.llm.transformers_backend import TransformersBackend

    fake = _install_fake_transformers(monkeypatch)
    b = TransformersBackend(model_key="qwen-coder-3b")
    b.load()
    b.unload()
    assert not b.loaded
    assert b._model is None
    assert fake.AutoTokenizer.from_pretrained.called


def test_generate_uses_chat_template_and_streams(monkeypatch) -> None:
    from coagent.llm.transformers_backend import TransformersBackend

    _install_fake_transformers(monkeypatch)
    b = TransformersBackend(model_key="qwen-coder-3b")
    chunks: list[str] = []
    result = b.generate(
        [Message("system", "sys"), Message("user", "hi")],
        GenerationConfig(temperature=0.2),
        on_token=chunks.append,
    )
    assert result.model == "Qwen2.5-Coder-3B-Instruct@nf4"
    assert result.stop_reason is StopReason.STOP
    assert result.usage.prompt_tokens > 0
    assert chunks


def test_generate_calls_stop_criteria_when_cancel_provided(monkeypatch) -> None:
    from coagent.llm.transformers_backend import TransformersBackend

    _install_fake_transformers(monkeypatch)
    b = TransformersBackend(model_key="qwen-coder-3b")
    called = {"n": 0}

    def cancel() -> bool:
        called["n"] += 1
        return False

    result = b.generate([Message("user", "x")], cancel=cancel)
    assert result.stop_reason == StopReason.STOP
    # The flag is polled once at the end to decide the stop reason.
    assert called["n"] >= 1


def test_oom_detection() -> None:
    assert _is_cuda_oom(RuntimeError("CUDA out of memory."))
    assert _is_cuda_oom(RuntimeError("torch.cuda.OutOfMemoryError: exceeded"))
    assert not _is_cuda_oom(RuntimeError("something else"))


def test_oom_message_is_actionable() -> None:
    exc = _oom_from(RuntimeError("out of memory"))
    assert "unloaded" in str(exc)
    assert "3B" in str(exc)


def test_ladder_order() -> None:
    from coagent.llm.transformers_backend import OutOfMemoryLadder

    ladder = OutOfMemoryLadder()
    assert ladder.next_action() == "shrink_context"
    assert ladder.advance() == "shrink_context"
    assert ladder.next_action() == "halve_output"
    ladder.advance()
    assert ladder.next_action() == "fallback_model"
    ladder.advance()
    assert ladder.exhausted
    assert ladder.next_action() == "stop"


# --------------------------------------------------- bitsandbytes availability
#
# Reproduced on Colab: the notebook installed the package *without* the [colab]
# extra, so bitsandbytes was absent and `from_pretrained` raised a bare
# ImportError from inside transformers. These pin the early, actionable message.


def test_missing_bitsandbytes_says_install_the_extra(monkeypatch) -> None:
    import pytest

    module = sys.modules["coagent.llm.transformers_backend"]
    monkeypatch.setattr(module, "_installed_bnb_version", lambda: None)
    with pytest.raises(ImportError, match=r"co-agent-coder\[colab\]"):
        module._check_bitsandbytes()


def test_old_bitsandbytes_names_the_version_and_the_fix(monkeypatch) -> None:
    import pytest

    module = sys.modules["coagent.llm.transformers_backend"]
    monkeypatch.setattr(module, "_installed_bnb_version", lambda: "0.45.5")
    with pytest.raises(ImportError) as err:
        module._check_bitsandbytes()
    msg = str(err.value)
    assert "0.45.5" in msg  # what they have
    assert "0.46.1" in msg  # what they need
    assert "restart the kernel" in msg  # the CUDA lib stays loaded otherwise


def test_a_supported_bitsandbytes_passes(monkeypatch) -> None:
    module = sys.modules["coagent.llm.transformers_backend"]
    for ok in ("0.46.1", "0.47.0", "0.50.2"):
        monkeypatch.setattr(module, "_installed_bnb_version", lambda v=ok: v)
        module._check_bitsandbytes()  # must not raise


def test_load_refuses_before_touching_the_model(monkeypatch) -> None:
    """The gate runs before any weights are touched, so the failure is cheap."""
    import pytest

    module = sys.modules["coagent.llm.transformers_backend"]
    fake = _install_fake_transformers(monkeypatch)
    monkeypatch.setattr(module, "_installed_bnb_version", lambda: None)
    b = TransformersBackend(model_key="qwen-coder-3b")
    with pytest.raises(ImportError, match="bitsandbytes"):
        b.load()
    assert not fake.AutoModelForCausalLM.from_pretrained.called
    assert not b.loaded
