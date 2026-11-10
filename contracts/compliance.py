"""Compliance data model — PRD-mandated statuses and deterministic gate.

The compliance GATE reads structured ComplianceResult.status — it MUST NOT
parse natural-language "PASS"/"FAIL" strings from agent output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ComplianceStatus(str, Enum):
    ALLOWED = "允许"
    NEEDS_MODIFICATION = "需修改"
    NEEDS_REVIEW = "需复核"
    PROHIBITED = "禁止"
    BLOCKED = "阻止继续"
    INSUFFICIENT_EVIDENCE = "资料不足"


# Statuses that MUST route to safe_stop (never enter Strategy)
_COMPLIANCE_BLOCKING = {ComplianceStatus.PROHIBITED, ComplianceStatus.BLOCKED}


@dataclass
class ComplianceCheck:
    """A single compliance check result."""
    rule_id: str
    rule_name: str
    passed: bool
    severity: str  # "info" | "warning" | "critical"
    detail: str = ""
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class ComplianceResult:
    """Standardised compliance output from the Compliance Agent node.

    The compliance_gate node reads ONLY this structured object — it never
    asks an LLM whether to proceed.
    """

    status: ComplianceStatus
    summary: str = ""
    applicable_rules: list[dict] = field(default_factory=list)
    scope: dict = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    risks: list[dict] = field(default_factory=list)
    required_changes: list[str] = field(default_factory=list)
    execution_preconditions: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    requires_human: bool = False

    @property
    def is_blocking(self) -> bool:
        """True when the gate MUST prevent Strategy execution."""
        return self.status in _COMPLIANCE_BLOCKING


def compliance_gate_route(result: ComplianceResult) -> str:
    """Deterministic routing decision — NO LLM involved.

    Returns LangGraph node keys matching the conditional edge map:
        "step_dispatcher" | "human_wait" | "safe_stop" | "context_assembly"

    IMPORTANT: ALLOWED and NEEDS_MODIFICATION return "step_dispatcher" (NOT "strategy").
    The step_dispatcher reads the plan to find the next step — for workflows that
    include strategy it will be strategy, for policy_search it will be
    evidence_pipeline/verifier.  The gate MUST NOT assume every task needs Strategy.
    """
    mapping = {
        ComplianceStatus.ALLOWED: "step_dispatcher",
        ComplianceStatus.NEEDS_MODIFICATION: "step_dispatcher",  # forward with modifications noted
        ComplianceStatus.NEEDS_REVIEW: "human_wait",
        ComplianceStatus.PROHIBITED: "safe_stop",
        ComplianceStatus.BLOCKED: "safe_stop",
        ComplianceStatus.INSUFFICIENT_EVIDENCE: "context_assembly",
    }
    return mapping.get(result.status, "safe_stop")
