"""Deterministic fault-handling policy for the unified LangGraph workflow.

The policy does not generate business conclusions.  It selects one approved
control action from the current failure context:

    retry -> switch capability -> reduce scope -> human review -> block/terminate

The selection is dynamic at runtime, but the available actions and their
conditions are deterministic and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class FailureType(str, Enum):
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    INVALID_PLAN = "invalid_plan"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    DATA_MISSING = "data_missing"
    TOOL_FAILURE = "tool_failure"
    AGENT_FAILURE = "agent_failure"
    MEMORY_UNAVAILABLE = "memory_unavailable"
    VERIFICATION_UNAVAILABLE = "verification_unavailable"
    PERMISSION_DENIED = "permission_denied"
    SECURITY_BLOCK = "security_block"
    AUDIT_UNAVAILABLE = "audit_unavailable"
    EXTERNAL_STATE_UNKNOWN = "external_state_unknown"
    UNKNOWN = "unknown"


class DegradationAction(str, Enum):
    RETRY = "retry"
    SWITCH_CAPABILITY = "switch_capability"
    REDUCE_SCOPE = "reduce_scope"
    HUMAN_REVIEW = "human_review"
    BLOCK = "block"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class FailureContext:
    component: str
    failure_type: FailureType
    error: str
    task_type: str = "general"
    risk_level: str = "medium"
    critical: bool = True
    attempt: int = 0
    max_retries: int = 1
    target_node: str = ""
    alternatives: tuple[str, ...] = ()
    external_side_effect_possible: bool = False
    external_state_known: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DegradationDecision:
    action: DegradationAction
    reason: str
    target_node: str = ""
    selected_alternative: str = ""
    allowed_use: str = "none"
    requires_human: bool = False
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data


@dataclass(frozen=True)
class DegradationEvent:
    component: str
    failure_type: str
    error: str
    action: str
    reason: str
    target_node: str = ""
    selected_alternative: str = ""
    allowed_use: str = "none"
    limitations: tuple[str, ...] = ()
    occurred_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_decision(
        cls,
        context: FailureContext,
        decision: DegradationDecision,
    ) -> "DegradationEvent":
        return cls(
            component=context.component,
            failure_type=context.failure_type.value,
            error=context.error,
            action=decision.action.value,
            reason=decision.reason,
            target_node=decision.target_node,
            selected_alternative=decision.selected_alternative,
            allowed_use=decision.allowed_use,
            limitations=decision.limitations,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PolicyRule = Callable[[FailureContext], DegradationDecision | None]


class DegradationPolicy:
    """Select an approved fault-handling branch from runtime context."""

    def __init__(self, rules: list[PolicyRule] | None = None) -> None:
        self._rules = rules or [
            self._external_state_rule,
            self._control_plane_rule,
            self._retry_rule,
            self._switch_rule,
            self._reduce_scope_rule,
            self._critical_failure_rule,
            self._unknown_failure_rule,
        ]

    def decide(self, context: FailureContext) -> DegradationDecision:
        for rule in self._rules:
            decision = rule(context)
            if decision is not None:
                return decision
        return DegradationDecision(
            action=DegradationAction.TERMINATE,
            reason="没有匹配到已批准的故障处置规则",
            target_node=context.target_node,
        )

    @staticmethod
    def _external_state_rule(context: FailureContext) -> DegradationDecision | None:
        if (
            context.failure_type == FailureType.EXTERNAL_STATE_UNKNOWN
            or (
                context.external_side_effect_possible
                and not context.external_state_known
            )
        ):
            return DegradationDecision(
                action=DegradationAction.HUMAN_REVIEW,
                reason="外部动作可能已经发生且状态无法确认，禁止自动重试",
                target_node="external_status_check",
                requires_human=True,
                limitations=("不得重复提交", "核对外部回执后才能继续"),
            )
        return None

    @staticmethod
    def _control_plane_rule(context: FailureContext) -> DegradationDecision | None:
        if context.failure_type in {
            FailureType.PERMISSION_DENIED,
            FailureType.SECURITY_BLOCK,
            FailureType.AUDIT_UNAVAILABLE,
        }:
            return DegradationDecision(
                action=DegradationAction.BLOCK,
                reason="权限、安全或审计控制面不可用，正式任务不得继续",
                target_node=context.target_node,
                limitations=("禁止交付正式结果", "禁止进入真实执行"),
            )
        return None

    @staticmethod
    def _retry_rule(context: FailureContext) -> DegradationDecision | None:
        if (
            context.failure_type
            in {FailureType.TRANSIENT, FailureType.RATE_LIMIT}
            and context.attempt < context.max_retries
            and not context.external_side_effect_possible
        ):
            return DegradationDecision(
                action=DegradationAction.RETRY,
                reason=f"暂时性故障，允许第{context.attempt + 1}次受控重试",
                target_node=context.target_node or context.component,
                allowed_use="unchanged",
            )
        return None

    @staticmethod
    def _switch_rule(context: FailureContext) -> DegradationDecision | None:
        if context.alternatives:
            alternative = context.alternatives[0]
            return DegradationDecision(
                action=DegradationAction.SWITCH_CAPABILITY,
                reason="存在已登记并通过测试的等价替代能力",
                target_node=context.target_node or context.component,
                selected_alternative=alternative,
                allowed_use="unchanged",
            )
        return None

    @staticmethod
    def _reduce_scope_rule(context: FailureContext) -> DegradationDecision | None:
        if not context.critical:
            return DegradationDecision(
                action=DegradationAction.REDUCE_SCOPE,
                reason="失败能力不是当前任务的必要条件，可以删除受影响内容后继续",
                target_node=context.target_node,
                allowed_use="reference_or_review",
                limitations=("必须说明缺失能力和影响", "不得直接用于自动执行"),
            )
        return None

    @staticmethod
    def _critical_failure_rule(context: FailureContext) -> DegradationDecision | None:
        if context.critical:
            return DegradationDecision(
                action=DegradationAction.HUMAN_REVIEW,
                reason="关键能力失败且没有可用的已批准替代能力",
                target_node=context.target_node,
                allowed_use="none",
                requires_human=True,
                limitations=("不得输出可执行结论",),
            )
        return None

    @staticmethod
    def _unknown_failure_rule(context: FailureContext) -> DegradationDecision | None:
        return DegradationDecision(
            action=DegradationAction.TERMINATE,
            reason="未知故障没有已批准的继续路径",
            target_node=context.target_node,
        )


def classify_failure(component: str, error: BaseException | str) -> FailureType:
    """Classify a technical error without asking an LLM to make the decision."""

    text = str(error).lower()
    component_text = component.lower()

    if any(token in text for token in ("permission", "forbidden", "unauthorized", "越权", "无权限")):
        return FailureType.PERMISSION_DENIED
    if any(token in text for token in ("audit", "审计")):
        return FailureType.AUDIT_UNAVAILABLE
    if any(token in text for token in ("rate limit", "429", "限流")):
        return FailureType.RATE_LIMIT
    if any(token in text for token in ("timeout", "timed out", "connection", "temporarily", "超时", "连接")):
        return FailureType.TRANSIENT
    if any(token in text for token in ("cycle", "dependency", "invalid plan", "计划", "依赖")):
        return FailureType.INVALID_PLAN
    if "memory" in component_text or "memory" in text:
        return FailureType.MEMORY_UNAVAILABLE
    if "verif" in component_text or "验证" in text:
        return FailureType.VERIFICATION_UNAVAILABLE
    if "retriev" in component_text or "rag" in component_text or "检索" in text:
        return FailureType.RETRIEVAL_UNAVAILABLE
    if "tool" in component_text or "工具" in text:
        return FailureType.TOOL_FAILURE
    if "agent" in component_text:
        return FailureType.AGENT_FAILURE
    return FailureType.UNKNOWN
