"""`coagent` command-line interface.

v0.1 scope: `coagent init`, `coagent doctor`, `coagent config`. The run/resume/
eval verbs land with the loop in milestone M1.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, load_config, write_default_config


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        target = args.path or "."
        path = write_default_config(Path(target), force=args.force)
    except ConfigError as exc:
        print(f"coagent: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    print("commit it so every session uses the same settings:")
    print(f"  git add {path} && git commit -m \"chore: add coagent config\"")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"coagent {__version__}  (python {platform.python_version()}, {platform.system()})")
    source = cfg.source or "(defaults — no .coagent/config.toml found)"
    print(f"config:  {source}")
    print(f"model:   {cfg.model.name} @ {cfg.model.quant}, ctx {cfg.model.max_context}")
    print(f"verify:  {cfg.verify.commands}")
    print(f"durable: {cfg.durability.mode} -> {cfg.durability.remote}")
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is None:
        print("torch:   not installed (install with `pip install coagent[llm]` in Colab)")
        return 0
    if not torch.cuda.is_available():
        print("GPU:     none detected — coAgent runs on a free Colab T4 (PRD D-1)")
        return 0
    props = torch.cuda.get_device_properties(0)
    name = props.name.replace("NVIDIA ", "")
    print(f"GPU:     {name} · {props.total_memory / 1024**3:.0f} GB · CC {props.major}.{props.minor}")
    if "T4" not in name or props.major < 7 or props.total_memory < 14 * 1024**3:
        print("verdict: NOT a free-tier T4 envelope — preflight will refuse unless --allow-unsupported")
    else:
        print("verdict: OK for the coAgent T4 envelope")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.path:
        print(cfg.source or "no .coagent/config.toml found")
    elif args.effective:
        print(cfg.to_toml())
    else:
        print(cfg.source or "no .coagent/config.toml found (using defaults)")
        print(cfg.to_toml())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coagent",
        description="Private AI coding agent for the free Google Colab T4.",
    )
    parser.add_argument("--version", action="version", version=f"coagent {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write a default .coagent/config.toml")
    p_init.add_argument("path", nargs="?", help="repo root (default: current directory)")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing config")
    p_init.set_defaults(func=_cmd_init)

    p_doctor = sub.add_parser("doctor", help="report versions, GPU, cache state, verdict")
    p_doctor.add_argument("--config", help="path to a config.toml (default: search upward)")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_config = sub.add_parser("config", help="print the effective config")
    p_config.add_argument("--config", help="path to a config.toml (default: search upward)")
    p_config.add_argument("--path", action="store_true", help="print only the config path")
    p_config.add_argument("--effective", action="store_true", help="print effective values as TOML")
    p_config.set_defaults(func=_cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())