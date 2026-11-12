"""Policy: the approval matrix, enforced in code (PRD FR-TOL-4, DESIGN §9.1).

Two things live here:

* `SafetyClass` — how a tool is classified (read / write / exec / vcs / ui).
* `Policy.decide()` — the one function that answers "may this run?".

Rules, in order (deny before allow, per FR-GIT-9):

1. A denylisted command is denied, always.
2. `git push` to a ref outside `git.push_ref_prefix`, or to an unconfigured
   remote, is denied.
3. Reads inside the jail are allowed.
4. Writes inside the repo on the session branch are allowed.
5. Everything in `require_approval` asks in interactive mode and is **denied**
   in non-interactive mode. The default flips to deny, not allow.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from .denylist import DEFAULT_DENIED, Denial, deny_reasons

__all__ = [
    "Decision",
    "Policy",
    "RiskClass",
    "SafetyClass",
    "Verdict",
]


class SafetyClass(str, enum.Enum):
    """How a tool is classified for the approval matrix."""

    READ = "read"
    WRITE = "write"
    EXEC = "exec"
    VCS = "vcs"
    UI = "ui"


class RiskClass(str, enum.Enum):
    """The action classes named in config `[safety] require_approval`."""

    NETWORK = "network"
    INSTALL = "install"
    DELETE = "delete"
    WRITE_OUTSIDE_REPO = "write_outside_repo"


class Verdict(str, enum.Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    reason: str
    rule: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    @property
    def needs_approval(self) -> bool:
        return self.verdict is Verdict.ASK


@dataclass
class Policy:
    """The session's enforcement state."""

    repo_root: Path
    interactive: bool = True
    allow_read: list[Path] = field(default_factory=list)
    push_ref_prefix: str = "refs/heads/coagent/"
    allowed_remotes: tuple[str, ...] = ("origin",)
    require_approval: tuple[str, ...] = (
        RiskClass.NETWORK.value,
        RiskClass.INSTALL.value,
        RiskClass.DELETE.value,
        RiskClass.WRITE_OUTSIDE_REPO.value,
    )
    deny_rules: tuple[str, ...] = DEFAULT_DENIED
    # Origins allowed to every write: the repo and the ephemeral scratch dir.
    scratch: Path | None = None

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.allow_read = [Path(p).resolve() for p in self.allow_read]
        if self.scratch is not None:
            self.scratch = Path(self.scratch).resolve()

    # ------------------------------------------------------------------ reads

    def check_read(self, path: Path) -> Decision:
        """Reads are jailed too (DESIGN §6.2) — after `mount_drive` an unjailed
        read is access to the user's entire Drive."""
        resolved = Path(path).resolve()
        if self._inside(resolved, self.repo_root):
            if ".git" in resolved.relative_to(self.repo_root).parts:
                # .git is not readable through read_file; git tools open it directly.
                return Decision(Verdict.DENY, "reading .git through the file tools is not allowed", "git-internal")
            return Decision(Verdict.ALLOW, "inside the repo", "read-in-jail")
        if self.scratch and self._inside(resolved, self.scratch):
            return Decision(Verdict.ALLOW, "inside the scratch dir", "read-scratch")
        for extra in self.allow_read:
            if self._inside(resolved, extra):
                return Decision(Verdict.ALLOW, "inside an allow_read entry", "read-allowed")
        return Decision(
            Verdict.DENY,
            f"{resolved} is outside the repo jail; add it to [tools] allow_read to permit it",
            "read-outside-jail",
        )

    # ----------------------------------------------------------------- writes

    def check_write(self, path: Path) -> Decision:
        resolved = Path(path).resolve()
        if self._inside(resolved, self.repo_root):
            return Decision(Verdict.ALLOW, "inside the repo on the session branch", "write-in-repo")
        if self.scratch and self._inside(resolved, self.scratch):
            return Decision(Verdict.ALLOW, "inside the scratch dir", "write-scratch")
        return self._risk(
            RiskClass.WRITE_OUTSIDE_REPO,
            f"{resolved} is outside the repo; writing there needs approval",
            "write-outside-repo",
        )

    # ------------------------------------------------------------------ shell

    def check_shell(self, cmd: str) -> Decision:
        denial: Denial | None = next(iter(deny_reasons(cmd, self.deny_rules)), None)
        if denial is not None:
            return Decision(Verdict.DENY, denial.reason, f"deny:{denial.rule}")
        if _is_delete(cmd):
            return self._risk(RiskClass.DELETE, "delete/move/chmod needs approval", "exec-delete")
        if _is_install(cmd):
            return self._risk(RiskClass.INSTALL, "installing packages needs approval", "exec-install")
        if _is_network(cmd):
            return self._risk(RiskClass.NETWORK, "network access needs approval", "exec-network")
        return Decision(Verdict.ALLOW, "shell command", "exec-allow")

    # -------------------------------------------------------------------- git

    def check_push(self, remote: str, ref: str) -> Decision:
        """Only configured remotes, only coagent/* refs (FR-GIT-8)."""
        if remote not in self.allowed_remotes:
            return Decision(
                Verdict.DENY,
                f"remote {remote!r} is not in [git] remotes {list(self.allowed_remotes)}",
                "push-remote",
            )
        normalized = ref if ref.startswith("refs/") else f"refs/heads/{ref}"
        if not normalized.startswith(self.push_ref_prefix):
            return Decision(
                Verdict.DENY,
                f"only {self.push_ref_prefix}* may be pushed, not {normalized}",
                "push-ref",
            )
        return Decision(Verdict.ALLOW, "checkpoint push to the session branch", "push-session")

    # ------------------------------------------------------------------ help

    def _risk(self, risk: RiskClass, why: str, rule: str) -> Decision:
        if risk.value in self.require_approval:
            if self.interactive:
                return Decision(Verdict.ASK, why, rule)
            return Decision(Verdict.DENY, f"{why} (non-interactive mode defaults to deny)", rule)
        return Decision(Verdict.ALLOW, why, rule)

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True


_INSTALL_HINTS = ("pip install", "pip3 install", "apt-get", "apt ", "conda install", "npm install", "uv pip")
_NETWORK_HINTS = (
    "curl",
    "wget",
    "git clone",
    "git fetch",
    "git pull",
    "requests.get",
    "http://",
    "https://",
    "huggingface-cli download",
)
_DELETE_HINTS = ("rm ", "rmdir", "mv ", "chmod", "chown", "unlink", "shutil.rmtree")


def _first_word(cmd: str) -> str:
    return cmd.strip().split(maxsplit=1)[0] if cmd.strip() else ""


def _is_install(cmd: str) -> bool:
    return any(hint in cmd for hint in _INSTALL_HINTS)


def _is_network(cmd: str) -> bool:
    word = _first_word(cmd)
    if word in {"curl", "wget", "ssh", "scp", "nc"}:
        return True
    return any(hint in cmd for hint in _NETWORK_HINTS)


def _is_delete(cmd: str) -> bool:
    return any(hint in cmd for hint in _DELETE_HINTS)
