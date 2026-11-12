"""Tool registry and the contract every tool implements (PRD FR-TOL-1).

The model may only do what is registered here, with only the arguments the
schema declares. Unknown tool names and unknown argument keys are rejected with
a corrective message rather than a crash — a malformed tool call costs a turn,
and the cheapest recovery is to tell the model exactly what was wrong.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..safety.policy import SafetyClass

__all__ = [
    "ArgSpec",
    "BaseTool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSchemaError",
]


class ToolError(RuntimeError):
    """A tool ran but failed. The message is fed back to the model verbatim."""


class ToolSchemaError(ToolError):
    """The call was malformed. Rejected before the tool body runs (FR-TOL-1)."""


@dataclass(frozen=True, slots=True)
class ArgSpec:
    """One declared argument."""

    name: str
    type: str  # "string" | "integer" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: Any = None
    enum: tuple[str, ...] | None = None

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass(slots=True)
class ToolResult:
    """What a tool returns to the loop."""

    ok: bool
    content: str
    tool: str = ""
    # Structured extras for the UI (diffs, paths, images) — never sent to the model.
    meta: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def success(cls, content: str, **meta: Any) -> ToolResult:
        return cls(ok=True, content=content, meta=meta)

    @classmethod
    def failure(cls, content: str, **meta: Any) -> ToolResult:
        return cls(ok=False, content=content, meta=meta)


class BaseTool(ABC):
    """One registered action."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    safety: ClassVar[SafetyClass] = SafetyClass.READ
    args: ClassVar[tuple[ArgSpec, ...]] = ()

    def __init__(self, ctx: Any = None) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------- schema

    @classmethod
    def parameters_schema(cls) -> dict[str, Any]:
        props = {a.name: a.json_schema() for a in cls.args}
        required = [a.name for a in cls.args if a.required and a.default is None]
        return {"type": "object", "properties": props, "required": required}

    @classmethod
    def tool_schema(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "description": cls.description,
            "safety": cls.safety.value,
            "parameters": cls.parameters_schema(),
        }

    @classmethod
    def validate_args(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Reject unknown keys and missing required args with a corrective message."""
        if not isinstance(args, dict):
            raise ToolSchemaError(
                f"arguments for {cls.name} must be an object, got {type(args).__name__}"
            )
        known = {a.name for a in cls.args}
        unknown = set(args) - known
        if unknown:
            raise ToolSchemaError(
                f"unknown argument(s) for {cls.name}: {', '.join(sorted(unknown))}. "
                f"Valid arguments are: {', '.join(sorted(known)) or '(none)'}."
            )
        cleaned: dict[str, Any] = {}
        for spec in cls.args:
            if spec.name not in args:
                if spec.required and spec.default is None:
                    raise ToolSchemaError(
                        f"missing required argument {spec.name!r} for {cls.name}."
                    )
                cleaned[spec.name] = spec.default
                continue
            cleaned[spec.name] = _coerce(cls.name, spec, args[spec.name])
        return cleaned

    # --------------------------------------------------------------- body

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Raise `ToolError` for an expected failure."""


def _coerce(tool: str, spec: ArgSpec, value: Any) -> Any:
    """Check the declared type, correcting the common JSON mistakes."""
    want = spec.type
    if want == "string":
        if not isinstance(value, str):
            raise ToolSchemaError(f"{tool}.{spec.name} must be a string, got {type(value).__name__}")
        if spec.enum and value not in spec.enum:
            raise ToolSchemaError(
                f"{tool}.{spec.name} must be one of {list(spec.enum)}, got {value!r}"
            )
        return value
    if want == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ToolSchemaError(f"{tool}.{spec.name} must be a boolean")
    if want == "integer":
        if isinstance(value, bool):
            raise ToolSchemaError(f"{tool}.{spec.name} must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        raise ToolSchemaError(f"{tool}.{spec.name} must be an integer")
    if want == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolSchemaError(f"{tool}.{spec.name} must be a number")
        return float(value)
    if want == "array":
        if not isinstance(value, list):
            raise ToolSchemaError(f"{tool}.{spec.name} must be an array")
        return value
    if want == "object":
        if not isinstance(value, dict):
            raise ToolSchemaError(f"{tool}.{spec.name} must be an object")
        return value
    return value


class ToolRegistry:
    """The set of tools available in the current phase."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} has no name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolSchemaError(
                f"unknown tool {name!r}. Available tools: {', '.join(sorted(self._tools)) or '(none)'}."
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, *, safety: set[SafetyClass] | None = None) -> list[dict[str, Any]]:
        """JSON tool schemas, optionally filtered to a phase's allowed classes."""
        return [
            t.tool_schema()
            for t in self._tools.values()
            if safety is None or t.safety in safety
        ]

    def schemas_json(self, *, safety: set[SafetyClass] | None = None) -> str:
        return json.dumps(self.schemas(safety=safety), indent=None, separators=(",", ":"))

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Validate then dispatch. Schema errors never reach the tool body."""
        tool = self.get(name)
        cleaned = tool.validate_args(args)
        result = tool.run(**cleaned)
        result.tool = name
        return result
