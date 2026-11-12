"""Context layer: repo map, token budget, truncation (DESIGN §5).

Context is the scarcest resource on a 12k window, so it gets its own ledger
rather than being assembled ad hoc at each call site.
"""

from __future__ import annotations

from .repo_map import DEFAULT_MAX_TOKENS, FileMap, RepoMap, RepoMapConfig, Signature
from .retrieval import Chunk, Recalled, RetrievalConfig, Retriever
from .tokens import estimate_tokens, fit, truncate_middle

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "Chunk",
    "FileMap",
    "Recalled",
    "RepoMap",
    "RepoMapConfig",
    "RetrievalConfig",
    "Retriever",
    "Signature",
    "estimate_tokens",
    "fit",
    "truncate_middle",
]
