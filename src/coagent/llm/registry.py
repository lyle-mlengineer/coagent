"""Model registry and the T4 preflight (PRD D1, FR-MDL-1).

v1 supports exactly two configurations: 7B at NF4/12k, and 3B at NF4/8k as the
only fallback. Nothing here imports torch at module scope, so `coagent doctor`
and the test suite run on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MODELS",
    "ModelSpec",
    "PreflightResult",
    "get_model",
    "preflight_gpu",
    "T4_MIN_VRAM_GB",
    "T4_COMPUTE_CAPABILITY",
]

T4_MIN_VRAM_GB = 14.0
T4_COMPUTE_CAPABILITY = (7, 5)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    name: str
    quant: str
    max_context: int
    vram_gb: float
    notes: str = ""

    @property
    def trailer_id(self) -> str:
        """The `Coagent-Model` value: `Qwen2.5-Coder-7B-Instruct@nf4`."""
        short = self.name.rsplit("/", 1)[-1]
        return f"{short}@{self.quant}"


MODELS: dict[str, ModelSpec] = {
    "qwen-coder-7b": ModelSpec(
        key="qwen-coder-7b",
        name="Qwen/Qwen2.5-Coder-7B-Instruct",
        quant="nf4",
        max_context=12288,
        vram_gb=6.5,
        notes="the only validated default on a free T4",
    ),
    "qwen-coder-3b": ModelSpec(
        key="qwen-coder-3b",
        name="Qwen/Qwen2.5-Coder-3B-Instruct",
        quant="nf4",
        max_context=8192,
        vram_gb=3.0,
        notes="OOM fallback only; never a bigger model (PRD D6)",
    ),
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; expected one of {sorted(MODELS)}") from None


def get_model_by_name(name: str) -> ModelSpec | None:
    for spec in MODELS.values():
        if spec.name == name:
            return spec
    return None


@dataclass(slots=True)
class PreflightResult:
    ok: bool
    device: str = "none"
    total_vram_gb: float = 0.0
    compute_capability: tuple[int, int] = (0, 0)
    reason: str = ""
    override_allowed: bool = True

    def describe(self) -> str:
        if self.device == "none":
            return "no CUDA device detected"
        cap = f"{self.compute_capability[0]}.{self.compute_capability[1]}"
        return f"{self.device} · {self.total_vram_gb:.0f} GB · CC {cap}"


def preflight_gpu(*, device_index: int = 0) -> PreflightResult:
    """Check the free-tier T4 envelope without loading any weights.

    Returns a result rather than raising: the caller decides whether to refuse,
    and the message is what the user reads (FR-BOOT-2).
    """
    try:
        import torch
    except ImportError:
        return PreflightResult(
            ok=False,
            reason=(
                "torch is not installed. coAgent needs the GPU stack: "
                'pip install "coagent[llm]"'
            ),
        )

    if not torch.cuda.is_available():
        return PreflightResult(
            ok=False,
            reason=(
                "no CUDA device is available. coAgent runs on a free Colab T4; "
                "select Runtime → Change runtime type → T4 GPU. CPU-only is out of "
                "scope (PRD D-1)."
            ),
        )

    props = torch.cuda.get_device_properties(device_index)
    name = props.name.replace("NVIDIA ", "").strip()
    vram = props.total_memory / 1024**3
    cap = (props.major, props.minor)

    problems: list[str] = []
    if "T4" not in name.upper():
        problems.append(
            f"the GPU is {name}, not a T4. coAgent is validated only on the free-tier T4 "
            "(PRD D-1). Pass --allow-unsupported to run anyway, unsupported and unvalidated."
        )
    if cap < T4_COMPUTE_CAPABILITY:
        problems.append(
            f"compute capability {cap[0]}.{cap[1]} is below 7.5; fp16 and SDPA are the only "
            "safe path and this device does not guarantee them."
        )
    if vram < T4_MIN_VRAM_GB:
        problems.append(
            f"only {vram:.1f} GB of VRAM is visible; the 7B at NF4 needs ~14 GB free to be safe. "
            "The 3B fallback at 8k needs less — pass --allow-unsupported to try it."
        )

    return PreflightResult(
        ok=not problems,
        device=name,
        total_vram_gb=vram,
        compute_capability=cap,
        reason=" ".join(problems),
    )
