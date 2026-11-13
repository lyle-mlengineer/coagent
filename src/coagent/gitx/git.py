"""A thin, jailed `git` runner.

Every git call goes through here for four reasons:

1. **No tokens in the environment.** The child gets a scrubbed env; credentials
   arrive only through `GIT_ASKPASS`, never in a remote URL (DESIGN §9.2).
2. **No accidental prompts.** `GIT_TERMINAL_PROMPT=0` turns a missing credential
   into an error instead of a hang — a hang in a notebook looks like a freeze.
3. **Machine-readable output.** `-z`/`--porcelain` everywhere, so parsing never
   depends on locale or colour.
4. **One place for the push allowlist** (FR-GIT-8): `push()` refuses any ref
   outside `refs/heads/coagent/` and any unconfigured remote before spawn.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..safety.policy import Policy, Verdict

__all__ = ["GitError", "GitRepo", "GitResult", "format_askpass_script"]


class GitError(RuntimeError):
    """A git command failed. Carries git's own stderr for the report."""

    def __init__(self, message: str, *, argv: list[str] | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.argv = argv or []
        self.stderr = stderr


@dataclass(slots=True)
class GitResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def format_askpass_script() -> str:
    """A tiny POSIX `GIT_ASKPASS` helper that echoes the token from the env.

    Written to `.coagent/tmp/askpass.sh` (mode 0700) instead of embedding the
    token in the remote URL, which would persist in `.git/config`, appear in
    `git remote -v`, and be captured in every Drive bundle.
    """
    return (
        "#!/bin/sh\n"
        "# coAgent credential helper. The token is read from the environment of the\n"
        "# git child process only; it is never written to disk or to .git/config.\n"
        'case "$1" in\n'
        '  *Username*) printf \'%s\\n\' "${COAGENT_GIT_USERNAME:-x-access-token}" ;;\n'
        '  *Password*) printf \'%s\\n\' "${COAGENT_GIT_TOKEN:-}" ;;\n'
        '  *) printf \'%s\\n\' "${COAGENT_GIT_TOKEN:-}" ;;\n'
        "esac\n"
    )


class GitRepo:
    """Git operations scoped to one working tree."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: Policy | None = None,
        author_name: str = "coAgent",
        author_email: str = "coagent@users.noreply.github.com",
        token: str | None = None,
        timeout_s: int = 120,
    ) -> None:
        self.root = Path(root).resolve()
        self.policy = policy
        self.author_name = author_name
        self.author_email = author_email
        self._token = token
        self.timeout_s = timeout_s
        self._askpass_override: Path | None = None

    # --------------------------------------------------------------- plumbing

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if "TOKEN" not in k.upper() and "SECRET" not in k.upper()}
        env.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_AUTHOR_NAME": self.author_name,
                "GIT_AUTHOR_EMAIL": self.author_email,
                "GIT_COMMITTER_NAME": self.author_name,
                "GIT_COMMITTER_EMAIL": self.author_email,
                # Deterministic plumbing: no pager, no colour, no quoting.
                "GIT_PAGER": "cat",
                "PAGER": "cat",
                "LC_ALL": "C",
                # No pager/colour/quoting. Crucially we do NOT override
                # core.autocrlf: whatever the machine and the repo already use
                # is what the index was built with, and forcing a different
                # value makes every CRLF file look permanently modified.
            }
        )
        # Preserve the identity so `git commit` is attributable, but never the
        # credential: that is injected per-call only for the ops that need it.
        if self._token:
            helper = getattr(self, "_askpass_override", None) or self._askpass_path()
            if Path(helper).exists():
                env["GIT_ASKPASS"] = str(helper)
                env["COAGENT_GIT_TOKEN"] = self._token
        return env

    def _askpass_path(self) -> Path:
        return self._askpass_dir() / "askpass.sh"

    def _askpass_dir(self) -> Path:
        return self.root / ".coagent" / "tmp"

    def install_askpass(self, directory: Path | None = None) -> Path:
        """Write the credential helper. Called once at bootstrap (DESIGN §11).

        `directory` overrides the default `<root>/.coagent/tmp` location. A clone
        must pass one that lives *outside* the destination: `git clone` refuses a
        non-empty target, so writing the helper into the repo-to-be makes every
        clone fail with exit 128 before it starts.
        """
        path = Path(directory) / "askpass.sh" if directory is not None else self._askpass_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_askpass_script(), encoding="utf-8")
        with contextlib.suppress(OSError):  # Windows/ACL edge case
            path.chmod(0o700)
        return path

    def use_askpass(self, path: Path) -> None:
        """Point credential injection at an already-installed helper."""
        self._askpass_override = Path(path)

    def run(self, *args: str, cwd: Path | None = None, check: bool = True) -> GitResult:
        argv = ["git", *args]
        proc = subprocess.run(
            argv,
            cwd=str(cwd or self.root),
            env=self._env(),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=self.timeout_s,
        )
        result = GitResult(argv=argv, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if check and not result.ok:
            raise GitError(
                f"git {' '.join(args)} failed (exit {result.returncode}): {proc.stderr.strip()[:800]}",
                argv=argv,
                stderr=proc.stderr,
            )
        return result

    # ------------------------------------------------------------------ state

    def is_repo(self) -> bool:
        return (self.root / ".git").exists() and self.run(
            "rev-parse", "--git-dir", check=False
        ).ok

    def head(self) -> str:
        return self.run("rev-parse", "HEAD").stdout.strip()

    def current_branch(self) -> str:
        return self.run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def branch_exists(self, name: str) -> bool:
        return self.run("rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).ok

    def remote_branch_exists(self, remote: str, name: str) -> bool:
        return self.run(
            "rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}/{name}", check=False
        ).ok

    def remotes(self) -> list[str]:
        out = self.run("remote", check=False).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]

    def status_porcelain(self, *, ignore: tuple[str, ...] = ()) -> list[str]:
        """Porcelain entries, optionally ignoring paths by prefix.

        coAgent's own runtime state (`.coagent/lock.json`, `tmp/`, `sessions/`)
        must not count as user dirt, or every session would refuse to start on
        its own lock file.
        """
        out = self.run("status", "--porcelain=v1", "-z", "--untracked-files=all").stdout
        entries = [entry for entry in out.split("\x00") if entry.strip()]
        if not ignore:
            return entries
        kept: list[str] = []
        for entry in entries:
            path = entry[3:] if len(entry) > 3 else entry
            if any(path == prefix or path.startswith(prefix) for prefix in ignore):
                continue
            kept.append(entry)
        return kept

    def is_dirty(self, *, ignore: tuple[str, ...] = ()) -> bool:
        return bool(self.status_porcelain(ignore=ignore))

    def changed_paths(self, *, ignore: tuple[str, ...] = ()) -> list[str]:
        """Paths from porcelain entries, with the two-char status stripped."""
        paths: list[str] = []
        for entry in self.status_porcelain(ignore=ignore):
            # Format: "XY <path>" (renames have an extra NUL-separated source).
            path = entry[3:] if len(entry) > 3 else entry
            if path:
                paths.append(path)
        return paths

    def staged_paths(self) -> list[str]:
        out = self.run("diff", "--cached", "--name-only", "-z").stdout
        return [p for p in out.split("\x00") if p]

    def has_staged_changes(self) -> bool:
        return self.run("diff", "--cached", "--quiet", check=False).returncode == 1

    def diff_stat(self, *, staged: bool = True) -> str:
        args = ["diff", "--stat"] + (["--cached"] if staged else [])
        return self.run(*args, check=False).stdout.strip()

    def diff(self, *, staged: bool = True) -> str:
        args = ["diff"] + (["--cached"] if staged else [])
        return self.run(*args, check=False).stdout

    # -------------------------------------------------------------- identity

    def ensure_identity(self) -> None:
        """Set user.name/user.email locally if unset — otherwise the first
        commit fails and the whole session looks broken (DESIGN §11)."""
        for key, value in (("user.name", self.author_name), ("user.email", self.author_email)):
            current = self.run("config", "--get", key, check=False)
            if not current.ok or not current.stdout.strip():
                self.run("config", key, value)

    # ------------------------------------------------------------------ write

    def switch(self, branch: str, *, create: bool = False) -> None:
        if create:
            self.run("switch", "-c", branch)
        else:
            self.run("switch", branch)

    def add(self, *paths: str) -> None:
        self.run("add", "--", *paths)

    def add_all_tracked(self) -> None:
        """Stage modifications and deletions, but not untracked files."""
        self.run("add", "-u")

    def commit(self, message: str) -> str:
        """Plain commit with a literal message (used mostly by tests)."""
        self._commit_stdin(message if message.endswith("\n") else message + "\n")
        return self.head()

    def _scrub_commit_editor(self) -> None:
        """Nothing to do; kept as the seam if a hooks policy is added later."""

    def _commit_stdin(self, message: str, *, argv_extra: list[str] | None = None) -> None:
        argv = ["git", "commit", "--no-verify", "--cleanup=verbatim", "-F", "-", *(argv_extra or [])]
        proc = subprocess.run(
            argv,
            cwd=str(self.root),
            env=self._env(),
            input=message,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            raise GitError(
                f"git commit failed (exit {proc.returncode}): {proc.stderr.strip()[:800]}",
                argv=argv,
                stderr=proc.stderr,
            )

    def commit_with_trailers(self, subject: str, body: str, trailers: list[tuple[str, str]]) -> str:
        """Commit with a message built from a subject, body, and trailers.

        Built in Python and passed via `-F -` on stdin so a trailer containing
        newlines or quotes cannot break the argument list.
        """
        self._commit_stdin(build_commit_message(subject, body, trailers))
        return self.head()

    def log_oneline(self, *, grep: str = "", max_count: int = 20, rev: str = "HEAD") -> list[str]:
        args = ["log", "--oneline", f"-n{max_count}"]
        if grep:
            args.append(f"--grep={grep}")
        args.append(rev)
        return [line for line in self.run(*args, check=False).stdout.splitlines() if line.strip()]

    # ------------------------------------------------------------------- push

    def push(self, remote: str, ref: str, *, policy: Policy | None = None) -> GitResult:
        """Push one ref, after the allowlist check (FR-GIT-8)."""
        active = policy or self.policy
        if active is not None:
            decision = active.check_push(remote, ref)
            if decision.verdict is not Verdict.ALLOW:
                raise GitError(f"push denied: {decision.reason} ({decision.rule})")
        else:  # no policy configured: fall back to the narrow default
            normalized = ref if ref.startswith("refs/") else f"refs/heads/{ref}"
            if not normalized.startswith("refs/heads/coagent/"):
                raise GitError(f"push denied: only coagent/* refs may be pushed, not {normalized}")
        if not self._token:
            raise GitError(
                "no GitHub token is loaded; add one to Colab secrets to push. "
                "Fall back to durability.mode = 'bundle' to checkpoint offline."
            )
        branch = ref.removeprefix("refs/heads/")
        return self.run("push", "--set-upstream", remote, f"HEAD:refs/heads/{branch}", check=False)

    def fetch(self, remote: str, *, depth: int | None = None) -> GitResult:
        args = ["fetch", remote]
        if depth:
            args.append(f"--depth={depth}")
        return self.run(*args, check=False)

    def clone(self, url: str, dest: Path, *, depth: int = 1) -> None:
        """Clone into `dest`. The URL must not carry credentials (DESIGN §9.2)."""
        if "@" in url.split("://", 1)[-1].split("/", 1)[0]:
            raise GitError(
                "the clone URL contains embedded credentials. Use a bare https URL; "
                "coAgent injects the token through GIT_ASKPASS instead."
            )
        argv = ["git", "clone", f"--depth={depth}", url, str(dest)]
        proc = subprocess.run(
            argv,
            cwd=str(dest.parent),
            env=self._env(),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=max(self.timeout_s, 600),
        )
        if proc.returncode != 0:
            raise GitError(
                f"git clone failed (exit {proc.returncode}): {proc.stderr.strip()[:800]}",
                argv=argv,
                stderr=proc.stderr,
            )


def build_commit_message(subject: str, body: str, trailers: list[tuple[str, str]]) -> str:
    """Subject, blank line, body, blank line, trailers — in that order.

    `--cleanup=verbatim` keeps the trailers' exact spelling, which the
    `--grep="Coagent-Step"` queries in the PRD depend on.
    """
    parts = [subject.strip()]
    if body.strip():
        parts.append(body.strip())
    if trailers:
        parts.append("\n".join(f"{key}: {value}" for key, value in trailers))
    return "\n\n".join(parts) + "\n"
