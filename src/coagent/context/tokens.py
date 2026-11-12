"""Token accounting for the context ledger (DESIGN §5.1).

Estimating tokens must be **cheap and deterministic**, not exact. The loop
makes dozens of these decisions per task and there is no tokenizer available on
the CPU path — the model may not even be loaded yet. A ~4 chars/token heuristic
is within 10–15% for source code and, crucially, it is *monotone*: if the
estimate says a block fits, the real block is very unlikely to blow the window.

The budget here is the repo map's slot only (2000 tokens per §5.1). Everything
that packs text into a fixed number of tokens goes through :func:`fit`.
"""

from __future__ import annotations

__all__ = ["estimate_tokens", "fit", "truncate_middle"]

#: Mean characters per token for Python/JS source. Code is denser in punctuation
#: than prose, so it tokenizes closer to 3.5–4 chars than the 4-of-English rule.
CHARS_PER_TOKEN = 4

#: Marker appended when a block was cut; it must not be silently mistaken for
#: real map content, so it is counted in every fitting decision.
ELLIPSIS = "…"


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of ``text``.

    Counts characters, not bytes: a non-ASCII identifier costs the same as an
    ASCII one under a BPE tokenizer for the blends we target, and using bytes
    would double-count UTF-8 continuation bytes.

    Always returns at least 1 for non-empty text so that a one-character line
    does not look free.
    """
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def truncate_middle(text: str, max_tokens: int) -> str:
    """Keep the head and tail of ``text``, dropping the middle.

    Head/tail is deliberate: for a repo map the head is the highest-ranked file
    and the tail is the most recently added context, while the middle is the
    least interesting. That matches the observation-truncation rule in
    DESIGN §5.1 and keeps the two ends, which is where a model looks first.
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = max_tokens * CHARS_PER_TOKEN
    marker = f"\n{ELLIPSIS} [truncated] {ELLIPSIS}\n"
    if max_chars <= len(marker):
        return text[:max_chars]

    room = max_chars - len(marker)
    head = int(room * 0.7)
    tail = room - head
    return text[:head] + marker + text[len(text) - tail :]


def fit(blocks: list[str], max_tokens: int, *, separator: str = "\n") -> str:
    """Pack ``blocks`` in order until ``max_tokens`` is reached.

    Order is the caller's ranking order, so this is a greedy pack (DESIGN §5.2:
    "pack greedily into the 2k budget"). A block that does not fit is not
    allowed to preempt a smaller one later in the list — the ranking decided the
    order and we stop at the first overflow, which keeps the output stable
    across runs instead of reshuffling on every estimate.
    """
    if max_tokens <= 0 or not blocks:
        return ""

    out: list[str] = []
    used = 0
    for block in blocks:
        if not block:
            continue
        cost = estimate_tokens(block) + (1 if out else 0)
        if used + cost > max_tokens:
            break
        out.append(block)
        used += cost

    if not out:
        # Nothing fit whole: fall back to the first block, truncated. A partial
        # top-ranked file beats an empty map.
        return truncate_middle(blocks[0], max_tokens)

    return separator.join(out)
