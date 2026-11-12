"""Configuration: load, validate, and version-check `.coagent/config.toml`.

The config is committed to the target repo, so it must be reproducible and
refuse to run when its schema is newer than this package understands
(PRD FR-DIS-5).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Colab is 3.11+, this is for laptop dev
    import tomli as tomllib

SCHEMA_VERSION = 1
CONFIG_RELPATH = Path(".coagent") / "config.toml"

__all__ = [
    "AgentConfig",
    "ArtifactsConfig",
    "ConfigError",
    "Config",
    "DurabilityConfig",
    "GitConfig",
    "ModelConfig",
    "SafetyConfig",
    "ToolsConfig",
    "UIConfig",
    "VerifyConfig",
    "default_config_text",
    "find_config",
    "load_config",
    "write_default_config",
]


class ConfigError(RuntimeError):
    """Raised when the config is missing, malformed, or an unsupported schema."""


@dataclass(slots=True)
class ModelConfig:
    name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    quant: str = "nf4"
    max_context: int = 12288
    max_new_tokens: int = 1024
    temperature_edit: float = 0.2
    temperature_plan: float = 0.7
    top_p: float = 0.95
    seed: int = 0  # 0 = pick a random seed at session start and log it
    constrained: bool = False
    fallback_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    fallback_max_context: int = 8192

    def __post_init__(self) -> None:
        if self.max_context > 12288:
            # PRD D2: the cap is hard, not advisory.
            raise ConfigError(
                f"model.max_context={self.max_context} exceeds the hard cap of 12288 "
                "for the free T4 envelope (PRD D2)."
            )
        if self.quant not in {"nf4", "fp16"}:
            raise ConfigError(f"model.quant must be 'nf4' or 'fp16', got {self.quant!r}.")


@dataclass(slots=True)
class VerifyConfig:
    commands: list[str] = field(default_factory=lambda: ["ruff check .", "pytest -q"])
    gpu_commands: list[str] = field(default_factory=list)
    timeout_s: int = 300
    vram_reserve_mb: int = 1024
    require_tests: bool = True
    feedback_max_tokens: int = 1500


@dataclass(slots=True)
class GitConfig:
    author_name: str = "coAgent"
    author_email: str = "coagent@users.noreply.github.com"
    remotes: list[str] = field(default_factory=lambda: ["origin"])
    clone_depth: int = 1
    push_ref_prefix: str = "refs/heads/coagent/"


@dataclass(slots=True)
class DurabilityConfig:
    mode: str = "push"
    remote: str = "origin"
    heartbeat_s: int = 180

    def __post_init__(self) -> None:
        if self.mode not in {"push", "bundle"}:
            raise ConfigError(f"durability.mode must be 'push' or 'bundle', got {self.mode!r}.")


@dataclass(slots=True)
class AgentConfig:
    max_steps: int = 60
    max_wall_s: int = 2700
    max_tokens: int = 200_000
    max_gpu_min: int = 40
    max_commits: int = 40
    max_consecutive_failures: int = 3
    replan_after_repeats: int = 3
    stop_after_repeats: int = 5


@dataclass(slots=True)
class ToolsConfig:
    allow_read: list[str] = field(default_factory=list)
    max_output_bytes: int = 65536
    shell_timeout_s: int = 120
    notebook_timeout_s: int = 300


@dataclass(slots=True)
class ArtifactsConfig:
    commit_under_mb: int = 5
    store: str = "drive"
    drive_dir: str = "coagent/artifacts"


@dataclass(slots=True)
class SafetyConfig:
    require_approval: list[str] = field(
        default_factory=lambda: ["network", "install", "delete", "write_outside_repo"]
    )
    deny: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UIConfig:
    mode: str = "show"
    gradio: bool = False
    gradio_port: int = 7860


@dataclass(slots=True)
class Config:
    """The whole `.coagent/config.toml`, parsed and validated."""

    schema_version: int = SCHEMA_VERSION
    model: ModelConfig = field(default_factory=ModelConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    git: GitConfig = field(default_factory=GitConfig)
    durability: DurabilityConfig = field(default_factory=DurabilityConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    source: Path | None = None

    def to_toml(self) -> str:
        return _dump_toml(self)


_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "verify": VerifyConfig,
    "git": GitConfig,
    "durability": DurabilityConfig,
    "agent": AgentConfig,
    "tools": ToolsConfig,
    "artifacts": ArtifactsConfig,
    "safety": SafetyConfig,
    "ui": UIConfig,
}


def _build_section(name: str, cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] must be a table, got {type(data).__name__}.")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"[{name}] has unknown key(s): {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(known))}."
        )
    return cls(**data)


def load_config(path: str | Path | None = None, *, start: str | Path | None = None) -> Config:
    """Load and validate a config file.

    With no `path`, searches upward from `start` (default: cwd) for
    `.coagent/config.toml`. A missing file yields defaults so that
    `coagent doctor` can run before the repo is initialised.
    """
    resolved = Path(path) if path is not None else find_config(start)
    if resolved is None:
        return Config()
    try:
        raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {resolved}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{resolved} is not valid TOML: {exc}") from exc

    version = raw.pop("schema_version", None)
    if version is None:
        raise ConfigError(f"{resolved} is missing schema_version (expected {SCHEMA_VERSION}).")
    if not isinstance(version, int):
        raise ConfigError(f"schema_version must be an integer, got {version!r}.")
    if version > SCHEMA_VERSION:
        raise ConfigError(
            f"{resolved} uses schema_version {version}, but this coagent build "
            f"understands {SCHEMA_VERSION}. Upgrade coagent, or regenerate the config "
            "with `coagent init`."
        )

    cfg = Config(schema_version=version, source=resolved)
    for key, value in raw.items():
        cls = _SECTIONS.get(key)
        if cls is None:
            raise ConfigError(f"unknown top-level section [{key}] in {resolved}.")
        setattr(cfg, key, _build_section(key, cls, value))
    return cfg


def find_config(start: str | Path | None = None) -> Path | None:
    """Walk upward from `start` looking for `.coagent/config.toml`."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        target = candidate / CONFIG_RELPATH
        if target.is_file():
            return target
    return None


def default_config_text() -> str:
    """The shipped template, with its comments intact."""
    return resources.files("coagent.templates").joinpath("config.toml").read_text(encoding="utf-8")


def write_default_config(repo_root: str | Path, *, force: bool = False) -> Path:
    """Write `.coagent/config.toml` into `repo_root`."""
    target = Path(repo_root) / CONFIG_RELPATH
    if target.exists() and not force:
        raise ConfigError(f"{target} already exists; pass force=True to overwrite.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(default_config_text(), encoding="utf-8")
    return target


def _dump_toml(cfg: Config) -> str:
    """Minimal TOML writer (no dependency); enough for round-tripping a Config."""
    lines = [f"schema_version = {cfg.schema_version}", ""]
    for name, cls in _SECTIONS.items():
        section = getattr(cfg, name)
        lines.append(f"[{name}]")
        for f in fields(cls):
            value = getattr(section, f.name)
            lines.append(f"{f.name} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '0'  # TOML has no null; only used defensively
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ConfigError(f"cannot serialise {type(value).__name__} to TOML")


def is_dataclass_config(obj: Any) -> bool:
    """Small helper used by tests and the doctor command."""
    return is_dataclass(obj)
