"""CLI surface tests."""

from __future__ import annotations

from pathlib import Path

from coagent.cli import main


def test_version_flag(capsys) -> None:
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_init_writes_config(tmp_path: Path, capsys) -> None:
    assert main(["init", str(tmp_path)]) == 0
    assert (tmp_path / ".coagent" / "config.toml").is_file()


def test_init_refuses_overwrite(tmp_path: Path, capsys) -> None:
    main(["init", str(tmp_path)])
    assert main(["init", str(tmp_path)]) == 1
    assert "already exists" in capsys.readouterr().err


def test_doctor_runs(tmp_path: Path, monkeypatch, capsys) -> None:
    main(["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in out


def test_tools_lists_phases(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["tools", "--phase", "all"]) == 0
    out = capsys.readouterr().out
    assert "plan     (0)" in out
    assert "edit_file" in out


def test_tools_json_is_valid(tmp_path: Path, monkeypatch, capsys) -> None:
    import json

    monkeypatch.chdir(tmp_path)
    assert main(["tools", "--phase", "act", "--json"]) == 0
    assert {s["name"] for s in json.loads(capsys.readouterr().out)} >= {"read_file", "edit_file"}


def test_models_lists_validated_configs(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "Qwen2.5-Coder-7B-Instruct@nf4" in out
    assert "Qwen2.5-Coder-3B-Instruct@nf4" in out
    assert "12288" in out


def test_models_gpu_preflight_never_crashes(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["models", "--gpu"]) == 0
    out = capsys.readouterr().out
    assert "gpu:" in out
    assert "verdict:" in out
