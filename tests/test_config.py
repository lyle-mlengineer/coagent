"""Config tests: schema versioning, hard caps, and template round-trip."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from coagent.config import (
    SCHEMA_VERSION,
    ConfigError,
    default_config_text,
    find_config,
    load_config,
    write_default_config,
)


def test_default_template_is_valid_toml_and_parses(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    cfg = load_config(tmp_path / ".coagent" / "config.toml")
    assert cfg.schema_version == SCHEMA_VERSION
    assert cfg.model.max_context == 12288
    assert cfg.model.quant == "nf4"
    assert "pytest -q" in cfg.verify.commands
    assert cfg.git.push_ref_prefix == "refs/heads/coagent/"


def test_missing_config_yields_defaults(tmp_path: Path) -> None:
    cfg = load_config(start=tmp_path)
    assert cfg.source is None
    assert cfg.model.name.startswith("Qwen/Qwen2.5-Coder-7B")


def test_find_config_walks_up(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_config(nested) == tmp_path / ".coagent" / "config.toml"


def test_context_cap_is_enforced(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    path = tmp_path / ".coagent" / "config.toml"
    path.write_text(path.read_text().replace("max_context = 12288", "max_context = 32768"))
    with pytest.raises(ConfigError, match="hard cap"):
        load_config(path)


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('schema_version = 99\n[model]\nname = "x"\n')
    with pytest.raises(ConfigError, match="schema_version 99"):
        load_config(path)


def test_unknown_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"schema_version = {SCHEMA_VERSION}\n[model]\nname = 'x'\nquantum = true\n")
    with pytest.raises(ConfigError, match="unknown key.*quantum"):
        load_config(path)


def test_round_trip_through_toml(tmp_path: Path) -> None:
    write_default_config(tmp_path)
    cfg = load_config(tmp_path / ".coagent" / "config.toml")
    again = tmp_path / "again.toml"
    again.write_text(cfg.to_toml())
    assert load_config(again).model == cfg.model


def test_template_has_no_null_seed_regression() -> None:
    # TOML has no null; the template must stay parseable.
    assert "seed = null" not in default_config_text()
    assert re.search(r"^seed = ", default_config_text(), re.MULTILINE)
