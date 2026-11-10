"""Minimum viable LLM Gateway.

Agents depend on this protocol, not on concrete model classes.
Adding a new model provider only requires a new Provider implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class LLMCallResult:
    content: str
    model: str = ""
    model_version: str = ""
    token_count: int = 0
    duration_ms: float = 0.0
    error: str = ""
    tool_calls: list[dict] = field(default_factory=list)


@runtime_checkable
class LLMGateway(Protocol):
    """Protocol that every Agent depends on.

    Agents MUST NOT read .env or instantiate ChatDeepSeek directly.
    """

    def invoke(
        self,
        *,
        agent_name: str,
        messages: list,
        tools: list | None = None,
        task_context: dict | None = None,
    ) -> LLMCallResult:
        ...

    def invoke_with_tools(
        self,
        *,
        agent_name: str,
        messages: list,
        tools: list,
        task_context: dict | None = None,
    ) -> LLMCallResult:
        ...

    def health_check(self) -> bool:
        ...
