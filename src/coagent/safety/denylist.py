"""Denylist of destructive commands (PRD FR-TOL-5, FR-GIT-9).

This is a seatbelt, not a sandbox — the reality of a root-ish Colab shell.
The real control is the approval matrix in `policy.py`. What this module does
is evaluate a command *before* anything runs, deny rules before allow rules,
and return an actionable reason.

Matching strategy: a permissive tokenizer (shlex) so `git push --force` and
`git push --force=true` and `git push -f` are all caught, but a commit message
quoting `reset --hard` is not.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

__all__ = ["Denial", "Denylist", "DEFAULT_DENIED", "deny_reasons"]


@dataclass(frozen=True, slots=True)
class Denial:
    rule: str
    reason: str


# Pattern: token sequence of a dangerous command. Tokens are matched in order
# and may skip over "=" joins: `git push --force` and `git push --force=true`.
_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("git-force-push", ("git", "push", "--force*"), "force-push rewrites remote history (FR-GIT-9)."),
    ("git-force-push-short", ("git", "push", "-f"), "force-push rewrites remote history (FR-GIT-9)."),
    ("git-push-main", ("git", "push", "*", "main*"), "pushing main is never allowed (FR-GIT-8)."),
    ("git-reset-hard", ("git", "reset", "--hard*"), "hard reset discards work (FR-GIT-9)."),
    (
        "git-branch-delete",
        ("git", "branch", "-D*"),
        "deleting branches is destructive (FR-GIT-9).",
    ),
    ("git-clean-fd", ("git", "clean", "-f*"), "git clean -f deletes untracked files."),
    ("rm-root", ("rm", "-rf", "/"), "rm -rf / is never valid here (FR-TOL-5)."),
    ("rm-home", ("rm", "-rf", "~"), "rm -rf ~ is never valid here."),
    ("chmod-root", ("chmod", "-R", "777", "/"), "chmod -R 777 / destroys a system."),
    ("chmod-home", ("chmod", "-R", "777", "~"), "chmod -R 777 ~ destroys a system."),
    (
        "pipe-to-shell",
        ("*", "|", "bash"),
        "piping to a shell executes untrusted input; requires approval at a higher level.",
    ),
    ("curl-pipe-sh", ("curl*", "|", "sh"), "curl | sh executes remote code (FR-TOL-5)."),
    ("wget-pipe-sh", ("wget*", "|", "sh"), "wget | sh executes remote code (FR-TOL-5)."),
    ("dd-disk", ("dd", "*", "of=/dev/*"), "dd to a device overwrites disks."),
    ("fork-bomb", ("*", ":(){", "|", "*", "&"), "fork bombs violate Colab fair use."),
    ("shutdown", ("shutdown", "*"), "shutting down the VM kills the session (FR-REL-2)."),
    ("reboot", ("reboot", "*"), "rebooting the VM kills the session (FR-REL-2)."),
    ("rm-var", ("rm", "-rf", "/var"), "deleting /var destroys the runtime."),
    ("mkfs", ("mkfs*", "*"), "mkfs formats disks."),
)


def _tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd)
    except ValueError:
        # Unbalanced quotes — split on whitespace instead so a denial still
        # applies; the command would fail to run anyway.
        return cmd.split()


def _token_matches(actual: str, rule: str) -> bool:
    if rule.endswith("*") and len(rule) > 1:
        prefix = rule[:-1]
        if actual.startswith(prefix):
            return True
        # Flags are case-sensitive in git, but `-d` (merged-only delete) is the
        # common lowercase slip; treat any `-d`/`-D` variant as destructive.
        if prefix == "-D":
            return actual.startswith("-d") or actual.startswith("-D")
        return False
    if rule == "*":
        return True
    return actual == rule


DEFAULT_DENIED: tuple[str, ...] = tuple(rule[0] for rule in _RULES)


def deny_reasons(cmd: str, rules: tuple[str, ...] = DEFAULT_DENIED) -> tuple[Denial, ...]:
    """Return every denial that applies to `cmd`.

    Deny rules are evaluated before allow rules (PRD FR-GIT-9): a force-push to
    a coagent/* ref is still denied.
    """
    tokens = _tokens(cmd)
    if not tokens:
        return ()
    reasons: list[Denial] = []
    for rule_id, pattern, reason in _RULES:
        if rule_id not in rules:
            continue
        if _match_pattern(tokens, pattern):
            reasons.append(Denial(rule=rule_id, reason=reason))
    return tuple(reasons)


def _match_pattern(tokens: list[str], pattern: tuple[str, ...]) -> bool:
    """Sequence match with wildcards.

    * a trailing `*` on a rule token is a prefix match, so `--force=true`
      matches the rule `--force*`;
    * the rule token `"*"` matches exactly one token;
    * the rule token `"|"` means the *rest* of the pattern must match
      somewhere later in the token list (a shell pipe).
    """
    if not tokens or not pattern:
        return False

    def match_from(tok_i: int, pat_i: int) -> bool:
        while pat_i < len(pattern):
            rule = pattern[pat_i]
            if rule == "|":
                rest = pattern[pat_i + 1 :]
                # A pipe: try every later token as the start of the rest.
                for start in range(tok_i, len(tokens)):
                    if match_from(start, pat_i + 1) if rest else True:
                        return True
                return False
            if tok_i >= len(tokens):
                return False
            if not _token_matches(tokens[tok_i], rule):
                return False
            tok_i += 1
            pat_i += 1
        return True

    return match_from(0, 0)


class Denylist:
    """Holds the configurable deny rules for one session."""

    def __init__(self, rules: tuple[str, ...] | list[str] | None = None) -> None:
        self.rules: tuple[str, ...] = tuple(rules) if rules is not None else DEFAULT_DENIED

    def check(self, cmd: str) -> Denial | None:
        reasons = deny_reasons(cmd, self.rules)
        return reasons[0] if reasons else None
