"""Fixed workflows — approved, pre-validated workflow specs.

New workflows are added as new WorkflowSpec entries — the main graph
structure never changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkflowSpec:
    name: str
    task_types: tuple[str, ...]
    entry_node: str
    required_nodes: tuple[str, ...]
    high_risk: bool = False
    version: str = "1.0.0"
    description: str = ""


# ── Current demo workflows ──────────────────────────────────────────

FIXED_WORKFLOWS: dict[str, WorkflowSpec] = {
    "policy_search": WorkflowSpec(
        name="policy_search",
        task_types=("policy_search", "policy_query"),
        entry_node="quant",
        required_nodes=("quant", "compliance"),
        high_risk=False,
        version="1.0.0",
        description="政策查询：Quant获取市场数据 + Compliance政策检索",
    ),
    "daily_pulse": WorkflowSpec(
        name="daily_pulse",
        task_types=("daily_pulse", "market_analysis", "market_report"),
        entry_node="quant",
        required_nodes=("quant", "expert", "compliance", "strategy"),
        high_risk=False,
        version="1.0.0",
        description="市场日报/分析：Q→E→C→S 完整流程",
    ),
    "bidding_advice": WorkflowSpec(
        name="bidding_advice",
        task_types=("bidding_advice", "pricing_strategy"),
        entry_node="quant",
        required_nodes=("quant", "expert", "compliance", "strategy"),
        high_risk=True,
        version="1.0.0",
        description="报价与交易策略：Q→E→C→ComplianceGate→S，高风险",
    ),
    "long_term_locking": WorkflowSpec(
        name="long_term_locking",
        task_types=("long_term_locking",),
        entry_node="quant",
        required_nodes=("quant", "expert", "compliance", "strategy"),
        high_risk=True,
        version="1.0.0",
        description="中长期锁仓策略：Q→E→C→ComplianceGate→S，高风险",
    ),
    "real_time_alert": WorkflowSpec(
        name="real_time_alert",
        task_types=("real_time_alert", "real_time_trading"),
        entry_node="quant",
        required_nodes=("quant", "expert", "compliance", "strategy"),
        high_risk=True,
        version="1.0.0",
        description="实时交易/日内调整：存储快照→Q→E→C→ComplianceGate→S，高风险",
    ),
}


def resolve_workflow(task_type: str) -> WorkflowSpec | None:
    """Find the fixed workflow for a task type, if one exists."""
    for spec in FIXED_WORKFLOWS.values():
        if task_type in spec.task_types:
            return spec
    return None


def get_required_agent_order(spec: WorkflowSpec) -> list[str]:
    """Return the ordered agent names for a fixed workflow."""
    agent_map = {
        "quant": "QUANT",
        "expert": "EXPERT",
        "compliance": "COMPLIANCE",
        "strategy": "STRATEGY",
    }
    return [
        agent_map[node]
        for node in spec.required_nodes
        if node in agent_map
    ]
