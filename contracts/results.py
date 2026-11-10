"""Unified component success/failure protocol.

Every LangGraph node returns structured results through these types.
No component may disguise failure as a success string.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComponentResult:
    """Standardised return type for every graph node.

    A node that completes successfully MUST return success=True.
    A node that fails MUST raise ComponentExecutionError or return success=False.
    Error strings MUST NOT be placed in ``data`` to simulate success.
    """

    component: str
    success: bool
    data: dict = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ComponentExecutionError(RuntimeError):
    """Structured exception for component failures.

    Raised when a node cannot produce a valid result.  The fault policy
    reads these fields to decide retry / switch / reduce / human / block.
    """

    component: str
    failure_type: str
    retryable: bool
    alternatives: tuple[str, ...]
    details: dict

    def __init__(
        self,
        component: str,
        message: str,
        *,
        failure_type: str = "unknown",
        retryable: bool = False,
        alternatives: tuple[str, ...] = (),
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.component = component
        self.failure_type = failure_type
        self.retryable = retryable
        self.alternatives = alternatives
        self.details = details or {}
