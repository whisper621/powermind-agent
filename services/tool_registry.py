"""Tool Registry — single source of truth for available tools.

Agents fetch tools through the registry rather than importing them directly.
Adding a new tool = adding a ToolSpec — no agent code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ToolSpec:
    name: str
    version: str
    domain: str
    allowed_agents: tuple[str, ...] = ()
    supported_task_types: tuple[str, ...] = ()
    high_risk: bool = False
    enabled: bool = True
    handler: Optional[Callable] = None
    description: str = ""
    metadata: dict = field(default_factory=dict)


class ToolRegistry:
    """Central registry of ToolSpec instances.

    Usage:
        registry = ToolRegistry()
        registry.register(ToolSpec(name="predict_price", ...))
        tools_for_quant = registry.get_for_agent("quant")
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for s in specs:
            self.register(s)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def get_for_agent(self, agent_name: str) -> list:
        """Return callable handlers for an agent's allowed tools."""
        handlers = []
        for spec in self._specs.values():
            if not spec.enabled:
                continue
            if not spec.allowed_agents or agent_name in spec.allowed_agents:
                if spec.handler is not None:
                    handlers.append(spec.handler)
        return handlers

    def get_for_task(self, task_type: str) -> list:
        """Return callable handlers for a task type."""
        handlers = []
        for spec in self._specs.values():
            if not spec.enabled:
                continue
            if not spec.supported_task_types or task_type in spec.supported_task_types:
                if spec.handler is not None:
                    handlers.append(spec.handler)
        return handlers

    def get_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def get_enabled_handlers(self) -> list:
        return [s.handler for s in self._specs.values() if s.enabled and s.handler is not None]

    def health_check(self) -> dict:
        return {
            "total": len(self._specs),
            "enabled": sum(1 for s in self._specs.values() if s.enabled),
            "by_domain": {
                d: sum(1 for s in self._specs.values() if s.domain == d)
                for d in sorted({s.domain for s in self._specs.values()})
            },
        }
