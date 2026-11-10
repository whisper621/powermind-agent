"""Unified PowerMind task State — the single source of truth across all nodes.

Every node reads/writes only its assigned fields.  Condition routes read
structured state, never natural-language strings.
"""

from __future__ import annotations

from typing import TypedDict, Optional


class PowerMindState(TypedDict, total=False):
    # ── Request identity ──
    request_id: str
    thread_id: str
    task_mode: str
    task_type: str

    # ── User input ──
    user_query: str
    confirmed_query: str
    province: str
    target_date: str
    client_id: str
    tenant_context: dict

    # ── Intent & routing ──
    intent: dict
    risk_flags: list[dict]
    missing_fields: list[str]
    clarification: dict

    # ── Workflow plan & dispatch ──
    workflow_type: str
    plan: dict
    plan_index: int
    current_step: str
    next_step: str
    completed_steps: list[str]

    # ── Retrieval ──
    retrieval_result: dict
    structured_data: list[dict]
    context_packages: dict
    channel_status: dict

    # ── Agent outputs ──
    quant_result: dict
    expert_result: dict
    compliance_result: dict
    strategy_result: dict
    reflection_result: dict

    # ── Evidence & verification ──
    evidence_records: list[dict]
    verification_result: dict

    # ── Task outcome ──
    task_status: str
    result_version: int
    execution_profile: str
    outcome: str
    warnings: list[str]
    result_limitations: list[str]
    final_response: str

    # ── Failure handling ──
    failure: dict
    fault_decision: dict
    degradation_events: list[dict]
    retry_counters: dict[str, int]

    # ── Human-in-the-loop ──
    pending_human_action: dict
    resume_data: dict
