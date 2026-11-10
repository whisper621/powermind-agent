"""Unified tool return protocol.

Every core tool MUST return ToolResult (or raise a domain exception).
Plain strings expressing error results are forbidden in active runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Standard return type for every core tool.

    success: True if the tool completed its primary function.
    data:    Structured output (dict / list / Pydantic model).
    error:   Error message when success=False.  MUST be empty on success.
    evidence: List of evidence records this tool produced.
    metadata: Extra context (latency, model version, cache hit, etc.).
    """

    success: bool
    data: dict | None = None
    error: str = ""
    evidence: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def ok(cls, data: dict | None = None, *, evidence: list[dict] | None = None,
           metadata: dict | None = None) -> "ToolResult":
        return cls(
            success=True,
            data=data or {},
            evidence=evidence or [],
            metadata=metadata or {},
        )

    @classmethod
    def fail(cls, error: str, *, data: dict | None = None,
             metadata: dict | None = None) -> "ToolResult":
        return cls(
            success=False,
            data=data or {},
            error=error,
            metadata=metadata or {},
        )
