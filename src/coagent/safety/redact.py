"""Secret redaction (PRD FR-PRV-2 / FR-BOOT-5).

Redaction happens *before* a string reaches a log, a display, a report, or a
committed file. Every entry point that writes text goes through `redact()`.

The output keeps the last four characters so a log stays useful for "which
token was that" without being usable as a credential.
"""

from __future__ import annotations

import re

__all__ = ["MASK", "Redactor", "redact", "redact_obj", "contains_secret", "SECRET_PATTERNS"]

MASK = "•••"

# Ordered most-specific first. Each entry is (label, compiled pattern, group).
# The group selects the value to redact when the overall match includes context.
_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    # Fine-grained GitHub PATs: github_pat_11ABCDEFG0abc...
    ("github_pat_fine", re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), 0),
    # Classic PATs and OAuth tokens.
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"), 0),
    # Hugging Face
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{16,}"), 0),
    # OpenAI-style
    ("openai", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), 0),
    # Anthropic
    ("anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"), 0),
    # AWS access key id
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0),
    # Google API key: AIza + 35 chars (case-sensitive alphabet)
    ("google_api", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,40}\b"), 0),
    # Slack
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"), 0),
    # Stripe
    ("stripe", re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}"), 0),
    # JWT: three base64url segments, the first of which decodes to {"alg":...
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), 0),
    # Credentials embedded in a URL (https://user:token@host) — clone URLs leak
    # into .git/config and remote -v output.
    ("url_credentials", re.compile(r"(://)[^/\s:@]+:[^/\s@]+(@)"), 2),
    # Bearer / token headers in captured output.
    ("bearer", re.compile(r"(?i)\b(?:bearer|token)\s+([A-Za-z0-9._\-]{16,})"), 1),
    # Private key blocks.
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        0,
    ),
    # Key=value assignments for known secret-ish names, e.g. GH_TOKEN=ghp_...
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|CREDENTIAL)[A-Z0-9_]*)"
            r"\s*[=:]\s*[\"']?([^\s\"',}]{8,})"
        ),
        2,
    ),
]

SECRET_PATTERNS = tuple(label for label, _, _ in _PATTERNS)


def _mask_value(value: str) -> str:
    """`MASK` plus the last 4 characters, never the first."""
    tail = value[-4:] if len(value) > 8 else ""
    return f"{MASK}{tail}"


class Redactor:
    """A reusable redactor with a set of literal secrets known for this session.

    Registering a token means it is removed even when it does not match any
    pattern — e.g. a short PAT pasted from a different provider.
    """

    __slots__ = ("_literals",)

    def __init__(self, literals: list[str] | None = None) -> None:
        self._literals: list[str] = [s for s in (literals or []) if s and len(s) >= 8]

    def add(self, secret: str | None) -> None:
        if secret and len(secret) >= 8 and secret not in self._literals:
            self._literals.append(secret)
        # Longest first so a token that contains another redacts as one unit.
        self._literals.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        if not text:
            return text
        for literal in self._literals:
            if literal in text:
                text = text.replace(literal, _mask_value(literal))
        for _label, pattern, group in _PATTERNS:
            text = pattern.sub(lambda m, g=group: _replace(m, g), text)
        return text

    def redact_obj(self, obj: object) -> object:
        return _walk(obj, self.redact)

    def contains_secret(self, text: str) -> bool:
        """True if redaction would change `text` — used by the leak tests."""
        return self.redact(text) != text


def _replace(match: re.Match[str], group: int) -> str:
    whole = match.group(0)
    if group == 0:
        return whole[:0] + _mask_value(whole)
    start, end = match.span(group)
    offset = match.start()
    rel_start, rel_end = start - offset, end - offset
    return whole[:rel_start] + _mask_value(whole[rel_start:rel_end]) + whole[rel_end:]


_DEFAULT = Redactor()


def redact(text: str) -> str:
    """Redact using the default pattern set."""
    return _DEFAULT.redact(text)


def redact_obj(obj: object) -> object:
    """Recursively redact strings inside dicts/lists/tuples."""
    return _DEFAULT.redact_obj(obj)


def contains_secret(text: str) -> bool:
    return _DEFAULT.contains_secret(text)


def _walk(obj: object, fn) -> object:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {fn(k) if isinstance(k, str) else k: _walk(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_walk(v, fn) for v in obj)
    if isinstance(obj, set):
        return {_walk(v, fn) for v in obj}
    return obj
