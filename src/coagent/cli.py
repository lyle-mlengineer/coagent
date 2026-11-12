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
    added = _ensure_gitignore(Path(target))
    if added:
        print(f"added {len(added)} entr{'y' if len(added) == 1 else 'ies'} to .gitignore: {', '.join(added)}")
    print("commit them so every session uses the same settings:")
    print('  git add .coagent/config.toml .gitignore && git commit -m "chore: add coagent config"')
    return 0


_COAGENT_IGNORES = (
    ".coagent/tmp/",
    ".coagent/lock.json",
    ".coagent/sessions/",
    ".coagent/bundles/",
    ".coagent/cache/",
    ".coagent/secrets*",
)


def _ensure_gitignore(root: Path) -> list[str]:
    """Add coAgent runtime ignores to a repo's .gitignore.

    `coagent init` is an explicit setup action, so editing `.gitignore` here is
    expected rather than a surprise. The session path warns instead of editing
    (DESIGN §11).
    """
    ignore = root / ".gitignore"
    try:
        existing = ignore.read_text(encoding="utf-8") if ignore.is_file() else ""
    except OSError:
        return []
    added = [entry for entry in _COAGENT_IGNORES if entry not in existing]
    if not added:
        return []
    block = "\n# coAgent runtime state (config.toml and plans/ are committed)\n" + "\n".join(added) + "\n"
    try:
        with ignore.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(block)
    except OSError:
        return []
    return added


def _cmd_resume(args: argparse.Namespace) -> int:
    from .gitx import GitError, SessionError, build_briefing, open_session

    repo_path = Path(args.repo or ".")
    try:
        session = open_session(repo_path, session_id=args.session, force=args.force)
    except (SessionError, GitError) as exc:
        print(f"coagent: {exc}", file=sys.stderr)
        return 1
    briefing = build_briefing(session.repo, session.session_id, session.branch)
    # A briefing is read-only: never leave a lock behind from a diagnostic.
    session.close()
    print(briefing.render())
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "session": briefing.session_id,
                    "branch": briefing.branch,
                    "next_step": briefing.next_step,
                    "committed_steps": briefing.committed_steps,
                    "trailers": briefing.trailers,
                    "plan": str(briefing.plan_path) if briefing.plan_path else None,
                    "open_items": briefing.open_items,
                    "dirty_paths": briefing.dirty_paths,
                },
                indent=2,
            )
        )
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


def _cmd_tools(args: argparse.Namespace) -> int:
    """Print the tool schemas for a phase — the exact contract the model sees."""

    from .tools import PHASE_SAFETY, build_registry

    cfg = load_config(args.config)
    repo = Path(args.repo or ".").resolve()
    if not repo.is_dir():
        print(f"coagent: {repo} is not a directory", file=sys.stderr)
        return 1
    try:
        if args.phase == "all":
            for phase in PHASE_SAFETY:
                names = build_registry(repo, cfg, phase=phase).names()
                label = ", ".join(names) if names else "(none - plan cannot act)"
                print(f"{phase:8s} ({len(names)}): {label}")
            return 0
        reg = build_registry(repo, cfg, phase=args.phase)
    except ValueError as exc:
        print(f"coagent: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(reg.schemas_json())
        return 0
    print(f"phase {args.phase}: {len(reg)} tool(s)")
    for schema in reg.schemas():
        params = schema["parameters"]
        required = set(params.get("required", []))
        args_desc = ", ".join(
            name if name in required else f"{name}?" for name in params["properties"]
        )
        print(f"  {schema['name']:12s} [{schema['safety']:5s}] ({args_desc})")
        print(f"      {schema['description']}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .llm import MODELS, preflight_gpu

    cfg = load_config(args.config)
    preflight = preflight_gpu() if args.gpu else None
    if preflight is not None:
        print(f"gpu:     {preflight.describe()}")
        print(f"verdict: {'OK' if preflight.ok else 'REFUSED'}")
        if not preflight.ok:
            print(f"reason:  {preflight.reason}")
        print()
    print(f"configured: {cfg.model.name} @ {cfg.model.quant} · ctx {cfg.model.max_context}")
    print()
    for spec in MODELS.values():
        mark = "*" if spec.name == cfg.model.name else " "
        print(f" {mark} {spec.key:15s} {spec.trailer_id:40s} ctx {spec.max_context:>6d}  ~{spec.vram_gb:.1f} GB")
        if spec.notes:
            print(f"   {spec.notes}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """survey -> plan -> act -> verify -> checkpoint -> report, from the terminal."""
    from .agent import coAgent as Agent

    cfg = load_config(args.config)
    repo = Path(args.repo or ".").resolve()
    if not repo.is_dir():
        print(f"coagent: {repo} is not a directory", file=sys.stderr)
        return 1

    backend = None
    if args.dry_run:
        # A scripted model: proves the plumbing without a GPU or a download.
        from .llm import FakeBackend

        backend = FakeBackend(_dry_run_script(args.task))

    def on_event(kind: str, data: dict) -> None:
        if kind == "tool":
            mark = "ok  " if data.get("ok") else "FAIL"
            print(f"  {mark} {data.get('tool')}")
        elif kind == "plan":
            print(f"  plan  {data.get('title')}  ({len(data.get('steps', []))} steps)")
        elif kind == "commit":
            print(f"  commit {str(data.get('sha'))[:8]} step {data.get('step')}")
        elif kind == "sync":
            print(f"  sync  {'ok' if data.get('ok') else 'warn'}: {data.get('message')}")
        elif kind == "verify":
            print(f"  verify {'passed' if data.get('ok') else 'failed'}")

    try:
        agent = Agent(
            repo,
            config=cfg,
            backend=backend,
            auto=args.auto or args.dry_run,
            interactive=not args.auto,
            token=None,
            on_event=on_event,
            force=args.force,
            allow_unsupported=args.allow_unsupported,
        )
    except Exception as exc:  # noqa: BLE001 - every refusal is a message, not a traceback
        print(f"coagent: {exc}", file=sys.stderr)
        return 1

    print(agent.status_line())
    try:
        result = agent.run(args.task)
    except KeyboardInterrupt:
        print("coagent: interrupted", file=sys.stderr)
        return 130
    finally:
        agent.close()

    print()
    print(result.render())
    return 0 if result.done else 2


def _dry_run_script(task: str) -> list[str]:
    """A scripted session so `--dry-run` exercises the real loop.

    Deliberately ends without a further action: the run should stop cleanly on
    "no more actions" rather than trip the loop detector, so the output shows a
    normal early exit rather than a failure.
    """
    return [
        "Surveyed the repository map. This is a dry run, so no model is loaded.",
        "# dry run\n"
        "- [ ] list the repository root\n"
        "- [ ] run the configured checks",
        '```json\n{"tool": "list_dir", "args": {"path": "."}}\n```',
        # FakeBackend repeats its final response, so the script must say "stop"
        # explicitly instead of relying on the queue running dry.
        "The dry run is complete; no further action is needed.",
    ]


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

    p_tools = sub.add_parser("tools", help="list the tool schemas available per phase")
    p_tools.add_argument("--config", help="path to a config.toml (default: search upward)")
    p_tools.add_argument("--repo", help="repo root to jail against (default: cwd)")
    p_tools.add_argument(
        "--phase",
        default="act",
        choices=["survey", "plan", "act", "verify", "report", "all"],
        help="phase to inspect (default: act)",
    )
    p_tools.add_argument("--json", action="store_true", help="print raw JSON schemas")
    p_tools.set_defaults(func=_cmd_tools)

    p_resume = sub.add_parser("resume", help="print the resume briefing for a session")
    p_resume.add_argument("--repo", help="repo root (default: cwd)")
    p_resume.add_argument("--session", help="session id to resume (default: from the branch)")
    p_resume.add_argument("--force", action="store_true", help="take over a live lock")
    p_resume.add_argument("--json", action="store_true", help="print the briefing as JSON")
    p_resume.set_defaults(func=_cmd_resume)

    p_models = sub.add_parser("models", help="list the validated models and check the GPU envelope")
    p_models.add_argument("--config", help="path to a config.toml (default: search upward)")
    p_models.add_argument("--gpu", action="store_true", help="run the T4 preflight check")
    p_models.set_defaults(func=_cmd_models)

    p_run = sub.add_parser("run", help="run one task: survey, plan, act, verify, report")
    p_run.add_argument("task", help="what to do, in one sentence")
    p_run.add_argument("--repo", help="repo root (default: cwd)")
    p_run.add_argument("--config", help="path to a config.toml (default: search upward)")
    p_run.add_argument("--auto", action="store_true", help="pre-approve the plan")
    p_run.add_argument("--dry-run", action="store_true", help="use a scripted model (no GPU)")
    p_run.add_argument("--force", action="store_true", help="take over a live session lock")
    p_run.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="run on non-T4 hardware (unsupported, unvalidated)",
    )
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())