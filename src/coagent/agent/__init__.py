"""The agent: phases, budgets, state, and the loop."""

from __future__ import annotations

from .budget import Budget, BudgetExhausted, BudgetLimits, LoopDetector, LoopVerdict
from .loop import AgentResult, ApprovalRequest, Approver, coAgent
from .prompts import SYSTEM_CORE, PlanStep, format_observations
from .state import Observation, PlanFile, SessionState, slugify

__all__ = [
    "SYSTEM_CORE",
    "AgentResult",
    "ApprovalRequest",
    "Approver",
    "Budget",
    "BudgetExhausted",
    "BudgetLimits",
    "LoopDetector",
    "LoopVerdict",
    "Observation",
    "PlanFile",
    "PlanStep",
    "SessionState",
    "coAgent",
    "format_observations",
    "slugify",
]
