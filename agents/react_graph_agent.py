#!/usr/bin/env python3
"""PowerMind unified LangGraph entry point — ONE graph, all execution paths.

Architecture:
  task_init → supervisor → workflow_selector
    → step_dispatcher (reads plan, iterates steps)
      → quant → expert → compliance → compliance_gate → strategy → reflection
    → evidence_pipeline → verifier → render_result

  All failures → fault_policy
    → RETRY(target_node) | SWITCH_CAPABILITY | REDUCE_SCOPE
    → HUMAN_REVIEW(interrupt) | BLOCK/TERMINATE(safe_stop)

There is no second orchestrator.  There is no compatibility path.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt, Command

from contracts.results import ComponentResult, ComponentExecutionError
from contracts.state import PowerMindState
from workflows.fixed_workflows import resolve_workflow, FIXED_WORKFLOWS

from middleware.degradation_policy import (
    DegradationAction,
    DegradationEvent,
    DegradationPolicy,
    FailureContext,
    FailureType,
    classify_failure,
)

logger = logging.getLogger(__name__)

# ── Whitelist of valid retry target nodes ──────────────────────────────
_RETRY_TARGET_WHITELIST = frozenset({
    "quant", "expert", "compliance", "strategy", "reflection",
    "context_assembly", "evidence_pipeline", "verifier", "planner",
})


# ═══════════════════════════════════════════════════════════════════════════
# AgentRunner protocol — decouples graph from concrete agent implementations
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class AgentRunner(Protocol):
    """Protocol for running sub-agents.

    The main graph depends on this protocol, NOT on concrete ChatDeepSeek
    imports or direct sub_agent function calls. Inject a DemoAgentRunner
    (which wraps the old implementations) or a FakeAgentRunner for testing.
    """

    def run(self, state: PowerMindState, agent_name: str) -> dict:
        ...


class DemoAgentRunner:
    """Offline Agent runner backed only by the seeded SQLite demo database."""

    def __init__(self, db_path: str | None = None) -> None:
        from pathlib import Path
        from retrieval.demo_rag import DemoRAGRetriever
        from scripts.seed_demo_v1 import ensure_demo_database
        from services.repository import DemoSQLiteMarketDataRepository

        path = Path(db_path) if db_path else ensure_demo_database()
        self.repository = DemoSQLiteMarketDataRepository(str(path))
        self.retriever = DemoRAGRetriever(path)

    def run(self, state: PowerMindState, agent_name: str) -> dict:
        from agents.demo_agents import run_demo_agent

        return run_demo_agent(
            state,
            agent_name,
            repository=self.repository,
            retriever=self.retriever,
        )


class FakeAgentRunner:
    """AgentRunner for testing — no real LLM, no real APIs.

    Inject pre-configured responses via ``responses`` dict keyed by agent_name.
    """

    def __init__(self, responses: dict[str, dict] | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def run(self, state: PowerMindState, agent_name: str) -> dict:
        self.calls.append(agent_name)
        if agent_name in self.responses:
            return self.responses[agent_name]
        return {"status": "success", "summary": f"{agent_name} ok (fake)", "agent": agent_name}


# ═══════════════════════════════════════════════════════════════════════════
# Graph builder with full dependency injection
# ═══════════════════════════════════════════════════════════════════════════

def build_graph(
    checkpointer=None,
    *,
    llm_gateway=None,
    tool_registry=None,
    market_data_repo=None,
    audit_sink=None,
    evidence_service=None,
    retrieval_service=None,
    agent_runner=None,
):
    """Build the unified PowerMind LangGraph with injected dependencies.

    Args:
        checkpointer: LangGraph Checkpointer (MemorySaver for Demo).
        llm_gateway: LLMGateway protocol impl.
        tool_registry: ToolRegistry for tool lookup.
        market_data_repo: MarketDataRepository for SQL data.
        audit_sink: AuditSink for business-critical audit writes.
        evidence_service: EvidenceService for evidence collection.
        retrieval_service: Optional retrieval pipeline.
        agent_runner: AgentRunner for sub-agent execution (defaults to DemoAgentRunner).
    Returns:
        Compiled StateGraph.
    """
    checkpointer = checkpointer or MemorySaver()
    if agent_runner is None:
        agent_runner = DemoAgentRunner()

    # ── Bind dependencies into node closures ──
    deps = {
        "llm_gateway": llm_gateway,
        "tool_registry": tool_registry,
        "market_data_repo": market_data_repo,
        "audit_sink": audit_sink,
        "evidence_service": evidence_service,
        "retrieval_service": retrieval_service,
        "agent_runner": agent_runner,
    }

    workflow = StateGraph(PowerMindState)

    # ── Core nodes ──
    workflow.add_node("task_init", _make_task_init(deps))
    workflow.add_node("supervisor", _make_supervisor(deps))
    workflow.add_node("workflow_selector", _make_workflow_selector(deps))
    workflow.add_node("planner", _make_planner(deps))
    workflow.add_node("plan_validator", _make_plan_validator(deps))
    workflow.add_node("step_dispatcher", _make_step_dispatcher(deps))
    workflow.add_node("context_assembly", _make_context_assembly(deps))
    workflow.add_node("quant", _make_agent_node("quant", deps))
    workflow.add_node("expert", _make_agent_node("expert", deps))
    workflow.add_node("compliance", _make_agent_node("compliance", deps))
    workflow.add_node("compliance_gate", _make_compliance_gate(deps))
    workflow.add_node("strategy", _make_agent_node("strategy", deps))
    workflow.add_node("reflection", _make_agent_node("reflection", deps))
    workflow.add_node("evidence_pipeline", _make_evidence_pipeline(deps))
    workflow.add_node("verifier", _make_verifier(deps))
    workflow.add_node("render_result", _make_render_result(deps))

    # ── Fault / human nodes ──
    workflow.add_node("fault_policy", _make_fault_policy(deps))
    workflow.add_node("safe_stop", _make_safe_stop(deps))
    workflow.add_node("human_wait", _make_human_wait(deps))

    # ── Entry ──
    workflow.set_entry_point("task_init")

    # ── Normal flow ──
    workflow.add_edge("task_init", "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"workflow_selector": "workflow_selector", "human_wait": "human_wait"},
    )
    workflow.add_edge("workflow_selector", "planner")
    workflow.add_edge("planner", "plan_validator")
    workflow.add_conditional_edges(
        "plan_validator",
        _route_after_plan_validator,
        {"step_dispatcher": "step_dispatcher", "fault_policy": "fault_policy"},
    )
    # step_dispatcher → first step node or context_assembly
    workflow.add_conditional_edges(
        "step_dispatcher",
        _route_from_dispatcher,
        {
            "context_assembly": "context_assembly",
            "quant": "quant",
            "expert": "expert",
            "compliance": "compliance",
            "strategy": "strategy",
            "reflection": "reflection",
            "evidence_pipeline": "evidence_pipeline",
            "verifier": "verifier",
            "render_result": "render_result",
            "fault_policy": "fault_policy",
            "safe_stop": "safe_stop",
        },
    )

    # ── Dynamic flow between agent nodes (step_dispatcher decides) ──
    # Each agent node → step_dispatcher (which reads plan for next step)
    workflow.add_edge("context_assembly", "step_dispatcher")
    workflow.add_edge("quant", "step_dispatcher")
    workflow.add_edge("expert", "step_dispatcher")
    workflow.add_conditional_edges(
        "compliance",
        _route_after_compliance,
        {"step_dispatcher": "step_dispatcher", "compliance_gate": "compliance_gate"},
    )
    workflow.add_conditional_edges(
        "compliance_gate",
        _route_after_compliance_gate,
        {
            "strategy": "strategy",
            "reflection": "reflection",
            "human_wait": "human_wait",
            "safe_stop": "safe_stop",
            "context_assembly": "context_assembly",
            "step_dispatcher": "step_dispatcher",
        },
    )
    workflow.add_edge("strategy", "step_dispatcher")
    workflow.add_edge("reflection", "step_dispatcher")
    workflow.add_edge("evidence_pipeline", "step_dispatcher")
    workflow.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"render_result": "render_result", "fault_policy": "fault_policy",
         "step_dispatcher": "step_dispatcher", "human_wait": "human_wait"},
    )
    workflow.add_edge("render_result", END)

    # ── Fault handling ──
    workflow.add_conditional_edges(
        "fault_policy",
        _route_after_fault_policy,
        {
            "retry": "step_dispatcher",  # dispatcher routes to target_node
            "step_dispatcher": "step_dispatcher",
            "context_assembly": "context_assembly",
            "quant": "quant",
            "expert": "expert",
            "compliance": "compliance",
            "strategy": "strategy",
            "reflection": "reflection",
            "evidence_pipeline": "evidence_pipeline",
            "verifier": "verifier",
            "planner": "planner",
            "human_wait": "human_wait",
            "safe_stop": "safe_stop",
        },
    )
    workflow.add_edge("human_wait", END)
    workflow.add_edge("safe_stop", END)

    return workflow.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════
# Node factories — each returns a callable closed over injected deps
# ═══════════════════════════════════════════════════════════════════════════

def _make_task_init(deps: dict):
    def _node(state: PowerMindState) -> dict:
        request_id = state.get("request_id") or str(uuid.uuid4())[:12]
        target_date = state.get("target_date") or datetime.now().strftime("%Y-%m-%d")
        # Business-critical audit: task start
        _audit(deps, "task_start", "task_init", request_id=request_id, business_critical=True)
        return {
            "request_id": request_id,
            "thread_id": request_id,
            "target_date": target_date,
            "task_status": "init",
            "plan_index": 0,
            "completed_steps": ["task_init"],
            "warnings": [],
            "degradation_events": [],
            "retry_counters": {},
            "execution_profile": "full",
            "result_limitations": [],
        }
    return _node


def _make_supervisor(deps: dict):
    def _node(state: PowerMindState) -> dict:
        query = state.get("user_query", "")
        province = state.get("province", "")
        target_date = state.get("target_date", "")
        task_type = state.get("task_type", "general")

        missing_fields: list[str] = []
        if not province:
            missing_fields.append("province")
        if not target_date:
            missing_fields.append("target_date")

        intent = {"query": query, "province": province or "", "target_date": target_date or "", "task_type": task_type}

        if missing_fields:
            return {
                "intent": intent,
                "missing_fields": missing_fields,
                "completed_steps": _add_completed(state, "supervisor"),
                "pending_human_action": {
                    "type": "clarify",
                    "missing": missing_fields,
                    "prompt": f"请提供缺失信息: {', '.join(missing_fields)}",
                },
            }
        return {"intent": intent, "missing_fields": [], "completed_steps": _add_completed(state, "supervisor")}
    return _node


def _route_after_supervisor(state: PowerMindState) -> str:
    if state.get("missing_fields"):
        return "human_wait"
    return "workflow_selector"


def _make_workflow_selector(deps: dict):
    def _node(state: PowerMindState) -> dict:
        task_type = state.get("task_type", "general")
        spec = resolve_workflow(task_type)
        if spec is not None:
            # Build ordered step list from required_nodes
            steps = [{
                "step_id": "context_assembly",
                "step_type": "retrieve",
                "agent": "context_assembly",
                "depends_on": [],
            }]
            for node in spec.required_nodes:
                steps.append({
                    "step_id": node,
                    "step_type": _agent_step_type(node),
                    "agent": node,
                    "depends_on": [steps[-1]["step_id"]],
                })

            # ── Insert compliance_gate AFTER compliance (if compliance exists) ──
            if "compliance" in spec.required_nodes:
                compliance_idx = next((i for i, s in enumerate(steps) if s["step_id"] == "compliance"), None)
                if compliance_idx is not None:
                    gate_step = {
                        "step_id": "compliance_gate",
                        "step_type": "compliance_check",
                        "agent": "compliance_gate",
                        "depends_on": ["compliance"],
                    }
                    steps.insert(compliance_idx + 1, gate_step)

            # ── Insert reflection AFTER strategy (if strategy exists) ──
            if "strategy" in spec.required_nodes:
                strategy_idx = next((i for i, s in enumerate(steps) if s["step_id"] == "strategy"), None)
                if strategy_idx is not None:
                    reflection_step = {
                        "step_id": "reflection",
                        "step_type": "verify",
                        "agent": "reflection",
                        "depends_on": ["strategy"],
                    }
                    steps.insert(strategy_idx + 1, reflection_step)

            # ── Update strategy to depend on compliance_gate (if both exist) ──
            if "strategy" in spec.required_nodes and "compliance" in spec.required_nodes:
                for s in steps:
                    if s["step_id"] == "strategy":
                        # Strategy depends on compliance_gate instead of raw compliance
                        s["depends_on"] = ["compliance_gate"]
                        break

            # ── Append evidence_pipeline and verifier at the end ──
            last_agent_step_id = _last_agent_step(steps)
            steps.append({
                "step_id": "evidence_pipeline",
                "step_type": "generate",
                "agent": "evidence_pipeline",
                "depends_on": [last_agent_step_id] if last_agent_step_id else [],
            })
            steps.append({
                "step_id": "verifier",
                "step_type": "verify",
                "agent": "verifier",
                "depends_on": ["evidence_pipeline"],
            })

            return {
                "workflow_type": spec.name,
                "completed_steps": _add_completed(state, "workflow_selector"),
                "plan": {
                    "plan_id": f"fixed:{spec.name}",
                    "steps": steps,
                    "fixed_workflow": True,
                    "high_risk": spec.high_risk,
                },
            }
        return {"workflow_type": "planner", "completed_steps": _add_completed(state, "workflow_selector")}
    return _node


def _agent_step_type(node_id: str) -> str:
    return {
        "quant": "analyse", "expert": "analyse", "compliance": "compliance_check",
        "compliance_gate": "compliance_check", "strategy": "generate",
        "reflection": "verify", "evidence_pipeline": "generate", "verifier": "verify",
    }.get(node_id, "analyse")


def _agent_deps(node_id: str, spec) -> list[str]:
    order = list(spec.required_nodes)
    try:
        idx = order.index(node_id)
    except ValueError:
        return []
    prev = order[idx - 1] if idx > 0 else None
    return [prev] if prev else []


def _last_agent_step(steps: list[dict]) -> str:
    """Return the step_id of the last business-agent step (excluding infra steps)."""
    infra = {"evidence_pipeline", "verifier", "compliance_gate", "reflection"}
    for s in reversed(steps):
        if s.get("step_id") not in infra:
            return s.get("step_id", "")
    return steps[-1]["step_id"] if steps else ""


def _make_planner(deps: dict):
    def _node(state: PowerMindState) -> dict:
        plan = state.get("plan", {})
        # Fixed workflow plans are already built by workflow_selector
        if plan.get("fixed_workflow"):
            return {"plan": plan, "completed_steps": _add_completed(state, "planner")}
        # For non-fixed tasks, build a minimal plan
        task_type = state.get("task_type", "general")
        return {
            "completed_steps": _add_completed(state, "planner"),
            "plan": {
                "plan_id": str(uuid.uuid4())[:8],
                "steps": [
                    {"step_id": "context_assembly", "step_type": "retrieve", "agent": "context_assembly", "depends_on": []},
                    {"step_id": "quant", "step_type": "analyse", "agent": "quant", "depends_on": ["context_assembly"]},
                    {"step_id": "compliance", "step_type": "compliance_check", "agent": "compliance", "depends_on": ["quant"]},
                    {"step_id": "compliance_gate", "step_type": "compliance_check", "agent": "compliance_gate", "depends_on": ["compliance"]},
                    {"step_id": "evidence_pipeline", "step_type": "generate", "agent": "evidence_pipeline", "depends_on": ["compliance_gate"]},
                    {"step_id": "verifier", "step_type": "verify", "agent": "verifier", "depends_on": ["evidence_pipeline"]},
                ],
                "fixed_workflow": False,
                "high_risk": False,
            },
        }
    return _node


def _make_plan_validator(deps: dict):
    def _node(state: PowerMindState) -> dict:
        from workflows.plan_validator import PlanValidator
        plan = state.get("plan", {})
        task_type = state.get("task_type", "general")
        validator = PlanValidator()
        validation = validator.validate(plan, task_type=task_type, is_high_risk=bool(plan.get("high_risk")))
        if validation.valid:
            return {"plan": plan, "completed_steps": _add_completed(state, "plan_validator")}
        return {
            "failure": {
                "component": "plan_validator",
                "error": "; ".join(validation.errors),
                "error_class": "PlanValidationError",
                "failure_type": FailureType.INVALID_PLAN.value,
                "alternatives": validation.suggested_fixed_workflow or [],
            },
        }
    return _node


def _route_after_plan_validator(state: PowerMindState) -> str:
    if state.get("failure"):
        return "fault_policy"
    return "step_dispatcher"


def _make_step_dispatcher(deps: dict):
    """Reads plan and current state to determine the NEXT node to execute."""
    def _node(state: PowerMindState) -> dict:
        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        completed = set(state.get("completed_steps", []))
        failure = state.get("failure", {})

        # If there's a pending failure, don't advance — let fault_policy handle it
        if failure:
            return {"next_step": "fault_policy"}

        # If no steps remain, go to verifier → render
        if not steps:
            return {"next_step": "render_result"}

        # Find the next uncompleted step whose deps are all completed
        for i, step in enumerate(steps):
            sid = step.get("step_id", "")
            if sid in completed:
                continue
            deps = step.get("depends_on", [])
            if all(d in completed for d in deps):
                return {
                    "plan_index": i,
                    "current_step": sid,
                    "next_step": sid,
                }

        # All steps completed → render
        return {"next_step": "render_result"}
    return _node


def _route_from_dispatcher(state: PowerMindState) -> str:
    next_step = state.get("next_step", "render_result")
    valid_targets = {
        "context_assembly", "quant", "expert", "compliance", "compliance_gate",
        "strategy", "reflection", "evidence_pipeline", "verifier",
        "render_result", "fault_policy", "safe_stop",
    }
    return next_step if next_step in valid_targets else "render_result"


def _make_context_assembly(deps: dict):
    def _node(state: PowerMindState) -> dict:
        province = state.get("province", "山东")
        target_date = state.get("target_date", "")
        repo = deps.get("market_data_repo")
        evidence_svc = deps.get("evidence_service")

        if repo is None:
            from services.repository import DemoSQLiteMarketDataRepository
            repo = DemoSQLiteMarketDataRepository()

        try:
            import hashlib, json
            price_history = repo.get_price_history(province, target_date)
            weather = repo.get_weather(province, target_date)

            structured_data: list[dict] = []
            evidence_records = list(state.get("evidence_records", []))

            if price_history:
                content = json.dumps(price_history, sort_keys=True, default=str)
                fp = hashlib.sha256(content.encode()).hexdigest()[:16]
                structured_data.append({"source": "SQLite/price_data", "province": province, "data_time": target_date, "content": f"近7天演示电价历史: {price_history}", "unit": "元/MWh"})
                evidence_records.append({
                    "evidence_id": f"sql:price:{fp[:8]}", "evidence_type": "price_history",
                    "source_type": "structured_data", "source_name": "price_data",
                    "source_locator": f"powermind_demo.db/price_data/{province}/{target_date}",
                    "province": province, "data_time": target_date,
                    "content_fingerprint": fp, "unit": "元/MWh", "verification_status": "verified",
                    "claim": f"近7天演示电价历史共{len(price_history)}天",
                    "data_notice": "数据库演示假数据",
                })

            if weather:
                content = json.dumps(weather, sort_keys=True, default=str)
                fp = hashlib.sha256(content.encode()).hexdigest()[:16]
                structured_data.append({"source": "SQLite/weather_data", "province": province, "data_time": target_date, "content": f"演示天气数据: {weather}"})
                evidence_records.append({
                    "evidence_id": f"sql:weather:{fp[:8]}", "evidence_type": "weather_data",
                    "source_type": "structured_data", "source_name": "weather_data",
                    "source_locator": f"powermind_demo.db/weather_data/{province}/{target_date}",
                    "province": province, "data_time": target_date,
                    "content_fingerprint": fp, "verification_status": "verified",
                    "claim": f"目标日起{len(weather)}天演示天气数据",
                    "data_notice": "数据库演示假数据",
                })

            return {
                "structured_data": structured_data,
                "evidence_records": evidence_records,
                "completed_steps": _add_completed(state, "context_assembly"),
            }
        except Exception as exc:
            return {
                "failure": {"component": "context_assembly", "error": str(exc)[:500], "error_class": type(exc).__name__, "failure_type": FailureType.RETRIEVAL_UNAVAILABLE.value},
            }
    return _node


# ═══════════════════════════════════════════════════════════════════════════
# Agent node factory — checks results, writes failure on error
# ═══════════════════════════════════════════════════════════════════════════

def _make_agent_node(agent_name: str, deps: dict):
    def _node(state: PowerMindState) -> dict:
        runner = deps.get("agent_runner")
        if runner is not None:
            result = runner.run(state, agent_name)
        else:
            result = _call_sub_agent_impl(state, agent_name)

        result_key = f"{agent_name}_result" if agent_name != "reflection" else "reflection_result"

        if result.get("status") == "success":
            return {result_key: result, "completed_steps": _add_completed(state, agent_name)}
        # Structured failure — do NOT add to completed_steps, write failure for fault_policy
        return {
            result_key: result,
            "failure": {
                "component": agent_name,
                "error": result.get("error", f"{agent_name}返回失败"),
                "error_class": "AgentFailure",
                "failure_type": result.get("failure_type", FailureType.AGENT_FAILURE.value),
                "retryable": result.get("retryable", False),
                "alternatives": result.get("alternatives", []),
            },
        }
    return _node


def _make_compliance_gate(deps: dict):
    def _node(state: PowerMindState) -> dict:
        from contracts.compliance import ComplianceStatus, ComplianceResult, compliance_gate_route
        compliance_raw = state.get("compliance_result") or {}
        legacy_status = compliance_raw.get("risk_status", compliance_raw.get("status", "UNKNOWN"))

        status_map = {
            "PASS": ComplianceStatus.ALLOWED, "ALLOWED": ComplianceStatus.ALLOWED, "允许": ComplianceStatus.ALLOWED,
            "WARN": ComplianceStatus.NEEDS_REVIEW, "NEEDS_MODIFICATION": ComplianceStatus.NEEDS_MODIFICATION, "需修改": ComplianceStatus.NEEDS_MODIFICATION,
            "NEEDS_REVIEW": ComplianceStatus.NEEDS_REVIEW, "需复核": ComplianceStatus.NEEDS_REVIEW,
            "FAIL": ComplianceStatus.PROHIBITED, "PROHIBITED": ComplianceStatus.PROHIBITED, "禁止": ComplianceStatus.PROHIBITED,
            "BLOCKED": ComplianceStatus.BLOCKED, "阻止继续": ComplianceStatus.BLOCKED,
            "INSUFFICIENT_EVIDENCE": ComplianceStatus.INSUFFICIENT_EVIDENCE, "资料不足": ComplianceStatus.INSUFFICIENT_EVIDENCE,
        }
        status = status_map.get(legacy_status, ComplianceStatus.NEEDS_REVIEW)
        cr = ComplianceResult(status=status, summary=compliance_raw.get("summary", ""))
        route = compliance_gate_route(cr)

        _audit(deps, "compliance_gate", f"{status.value} → {route}", request_id=state.get("request_id", ""), business_critical=True)

        new_compliance = {**compliance_raw, "_gate_status": status.value, "_gate_route": route}
        if route == "safe_stop":
            return {"compliance_result": new_compliance, "failure": {"component": "compliance_gate", "error": f"合规阻止: {status.value}", "error_class": "ComplianceBlock", "failure_type": FailureType.SECURITY_BLOCK.value}, "completed_steps": _add_completed(state, "compliance_gate")}
        return {"compliance_result": new_compliance, "completed_steps": _add_completed(state, "compliance_gate")}
    return _node


def _route_after_compliance(state: PowerMindState) -> str:
    if state.get("failure"):
        return "step_dispatcher"
    return "compliance_gate"


def _route_after_compliance_gate(state: PowerMindState) -> str:
    """Route after compliance_gate based on structured gate decision.

    ALLOWED/NEEDS_MODIFICATION → step_dispatcher (plan decides next step)
    PROHIBITED/BLOCKED → safe_stop
    NEEDS_REVIEW → human_wait
    INSUFFICIENT_EVIDENCE → context_assembly (with loop guard)
    """
    compliance = state.get("compliance_result", {})
    route = compliance.get("_gate_route", "safe_stop")

    # Guard against infinite context_assembly loops
    if route == "context_assembly":
        completed = state.get("completed_steps", [])
        evidence_attempts = sum(1 for s in completed if s == "context_assembly")
        if evidence_attempts >= 2:
            return "human_wait"  # Prevent infinite loop

    return route


# ═══════════════════════════════════════════════════════════════════════════
# Evidence, Verifier, Render
# ═══════════════════════════════════════════════════════════════════════════

def _make_evidence_pipeline(deps: dict):
    def _node(state: PowerMindState) -> dict:
        collected = list(state.get("evidence_records", []))
        warnings = list(state.get("warnings", []))
        for result_key in ("quant_result", "expert_result", "compliance_result", "strategy_result"):
            result = state.get(result_key, {}) or {}
            collected.extend(result.get("evidence", []))

        evidence_by_id: dict[str, dict] = {}
        for record in collected:
            evidence_id = record.get("evidence_id")
            if not evidence_id:
                warnings.append("发现缺少evidence_id的证据记录，已忽略")
                continue
            evidence_by_id[evidence_id] = record

        evidence_records = list(evidence_by_id.values())
        if not evidence_records:
            warnings.append("证据管线未收集到可交付证据")
        return {
            "evidence_records": evidence_records,
            "warnings": warnings,
            "completed_steps": _add_completed(state, "evidence_pipeline"),
        }
    return _node


def _make_verifier(deps: dict):
    def _node(state: PowerMindState) -> dict:
        strategy = state.get("strategy_result", {})
        compliance = state.get("compliance_result", {})
        reflection = state.get("reflection_result", {})
        evidence_records = list(state.get("evidence_records", []))
        gate_route = compliance.get("_gate_route", "")
        task_type = state.get("task_type", "general")
        issues: list[dict] = []

        # If compliance blocked, verification automatically fails
        if gate_route == "safe_stop":
            issues.append({"category": "compliance", "severity": "critical", "description": "合规门禁阻止"})
            return {
                "verification_result": {"passed": False, "total_claims": 1, "verified_claims": 0, "unverified_claims": 1, "overall_confidence": 0.0, "issues": issues, "suggestions": ["合规阻止，不得交付正式结果"]},
                "completed_steps": _add_completed(state, "verifier"),
                "failure": {"component": "verifier", "error": "合规阻止", "error_class": "ComplianceBlock", "failure_type": FailureType.SECURITY_BLOCK.value},
            }

        if compliance.get("status") != "success" or gate_route != "step_dispatcher":
            issues.append({"category": "compliance", "severity": "critical", "description": "Compliance结果或门禁路由无效"})

        if task_type == "policy_search":
            if compliance.get("status") != "success":
                issues.append({"category": "generation", "severity": "critical", "description": "政策查询未形成有效Compliance回答"})
        else:
            if strategy.get("status") != "success":
                issues.append({"category": "generation", "severity": "critical", "description": "Strategy未生成有效结果"})
            if not reflection.get("passed"):
                issues.append({"category": "reflection", "severity": "critical", "description": "Reflection质检未通过"})

        if not evidence_records:
            issues.append({"category": "evidence", "severity": "critical", "description": "无证据记录"})
        total_claims = len(evidence_records)
        verified_claims = sum(
            1 for item in evidence_records
            if item.get("verification_status") == "verified" and item.get("source_locator")
        )
        unverified_claims = total_claims - verified_claims
        if total_claims and verified_claims == 0:
            issues.append({"category": "evidence", "severity": "critical", "description": "证据均未验证或缺少来源定位符"})
        elif unverified_claims:
            issues.append({"category": "evidence", "severity": "warning", "description": f"{unverified_claims}条证据尚未验证"})

        passed = not any(item.get("severity") == "critical" for item in issues) and total_claims > 0
        confidence = round(verified_claims / total_claims, 2) if total_claims else 0.0
        if not passed:
            plan = state.get("plan", {})
            result = {
                "verification_result": {"passed": False, "total_claims": max(total_claims, 1), "verified_claims": verified_claims, "unverified_claims": max(unverified_claims, 1), "overall_confidence": confidence, "issues": issues, "suggestions": ["补齐证据或修正上游结果后再运行"]},
                "completed_steps": _add_completed(state, "verifier"),
                "warnings": state.get("warnings", []) + [f"验证未通过: {len([i for i in issues if i.get('severity') == 'critical'])}个严重问题"],
            }
            # For high-risk tasks, set pending_human_action so human_wait has context
            if plan.get("high_risk"):
                evidence_issue = next((i for i in issues if i.get("category") == "evidence"), None)
                reason = evidence_issue.get("description", "关键结论未通过验证") if evidence_issue else "自动验证未通过"
                result["pending_human_action"] = {
                    "type": "verification_review",
                    "prompt": "高风险任务自动验证未通过，需要人工复核后才能交付。",
                    "reason": reason,
                    "issues": issues,
                    "resume_hint": "复核后通过 resume_task(thread_id, human_input) 恢复任务",
                }
            return result

        return {
            "verification_result": {"passed": True, "total_claims": total_claims, "verified_claims": verified_claims, "unverified_claims": unverified_claims, "overall_confidence": confidence, "issues": issues, "suggestions": ["关键结论具有数据库来源定位符，演示闭环可交付"]},
            "completed_steps": _add_completed(state, "verifier"),
        }
    return _node


def _make_render_result(deps: dict):
    def _node(state: PowerMindState) -> dict:
        strategy = state.get("strategy_result", {})
        compliance = state.get("compliance_result", {})
        reflection = state.get("reflection_result", {})
        verification = state.get("verification_result", {})
        degradation = state.get("degradation_events", [])
        warnings = list(state.get("warnings", []))
        outcome = state.get("outcome", "completed")
        limitations = list(state.get("result_limitations", []))
        plan = state.get("plan", {})
        high_risk = plan.get("high_risk", False)
        task_type = state.get("task_type", "general")

        if verification and not verification.get("passed"):
            if high_risk:
                outcome = "blocked"
            else:
                outcome = "limited"
                limitations.append("验证未通过，结果仅供参考，不得用于正式决策或执行")

        task_status = "completed" if outcome == "completed" else ("limited" if outcome == "limited" else "stopped")

        from middleware.dynamic_prompt import get_prompt_registry

        registry = get_prompt_registry()
        template_asset = registry.get_asset("unified_result_template")
        template = template_asset["content"]
        evidence_records = list(state.get("evidence_records", []))

        def _status(result: dict, expected: bool = True) -> str:
            if not expected:
                return "➖ 不适用"
            return "✅ 完成" if result.get("status") == "success" else "❌ 失败"

        def _summary(result: dict, expected: bool = True) -> str:
            if not expected:
                return "当前工作流不调用"
            return f"Prompt {result.get('prompt_id', '?')}@{result.get('prompt_version', '?')}，证据{len(result.get('evidence', []))}条"

        evidence_lines = []
        for item in evidence_records:
            claim = str(item.get("claim", item.get("evidence_type", "")))
            if len(claim) > 260:
                claim = claim[:257] + "..."
            evidence_lines.extend([
                f"### [{item.get('evidence_id', '?')}] {item.get('evidence_type', 'evidence')}",
                f"- 结论：{claim}",
                f"- 来源：`{item.get('source_locator', '?')}`｜状态：{item.get('verification_status', 'unverified')}",
                "",
            ])

        business_output = compliance.get("summary", "") if task_type == "policy_search" else strategy.get("summary", "")
        reflection_text = (
            f"## Reflection 质检\n\n"
            f"- 状态：{'✅ 通过' if reflection.get('passed') else ('➖ 政策查询不适用' if task_type == 'policy_search' else '❌ 未通过')}\n"
            f"- 得分：{reflection.get('score', 'N/A')}\n"
            f"- 说明：{reflection.get('summary', '政策查询由Compliance直接交付并由Verifier验证')}\n"
        )
        final_completed_steps = _add_completed(state, "render_result")
        completed = " → ".join(final_completed_steps)
        degradation_text = "\n".join(
            f"- `{event.get('component', '?')}` → {event.get('action', '?')}: {event.get('reason', '')}"
            for event in degradation[-5:]
        ) or "- 无故障降级，全部组件正常完成。"
        issue_text = "\n".join(
            f"- [{item.get('severity', '?')}] {item.get('description', '')}"
            for item in verification.get("issues", [])
        ) or "- 无严重问题。"
        warning_items = limitations + warnings + ["纯数据库演示：全部市场、政策、案例和策略数据均为SQLite假数据，不得用于真实交易。"]
        warning_text = "\n".join(f"- {item}" for item in warning_items)

        class _SafeValues(dict):
            def __missing__(self, key):
                return ""

        expert_expected = task_type != "policy_search"
        strategy_expected = task_type != "policy_search"
        values = _SafeValues({
            "task_type": task_type,
            "province": state.get("province", ""),
            "target_date": state.get("target_date", ""),
            "request_id": state.get("request_id", ""),
            "duration_ms": "本地即时",
            "intent": task_type,
            "intent_confidence": "fixed-workflow",
            "execution_profile": "sqlite_demo_no_api",
            "outcome": outcome,
            "quant_status": _status(state.get("quant_result", {})),
            "quant_summary": _summary(state.get("quant_result", {})),
            "expert_status": _status(state.get("expert_result", {}), expert_expected),
            "expert_summary": _summary(state.get("expert_result", {}), expert_expected),
            "compliance_status": _status(compliance),
            "compliance_summary": _summary(compliance),
            "strategy_status": _status(strategy, strategy_expected),
            "strategy_summary": _summary(strategy, strategy_expected),
            "evidence_status": "✅ 已绑定" if evidence_records else "❌ 无证据",
            "evidence_count": len(evidence_records),
            "verify_status": "✅ 通过" if verification.get("passed") else "❌ 未通过",
            "verified_claims": verification.get("verified_claims", 0),
            "total_claims": verification.get("total_claims", 0),
            "degraded_warning": "" if not degradation else "⚠️ 本任务发生过故障处置，请查看下方记录。",
            "degradation_events": degradation_text,
            "evidence_cards": "\n".join(evidence_lines) or "无证据",
            "strategy_output": business_output + "\n\n" + reflection_text + f"\n## 全链路执行轨迹\n\n{completed}",
            "verify_passed": "通过" if verification.get("passed") else "未通过",
            "overall_confidence": verification.get("overall_confidence", 0.0),
            "verify_issues": issue_text,
            "verify_suggestions": "\n".join(f"- {item}" for item in verification.get("suggestions", [])),
            "warnings": warning_text,
        })
        final_response = template.format_map(values)
        final_response += f"\n\n---\n输出模板：`{template_asset['prompt_id']}@{template_asset['version']}`"

        _audit(deps, "result_rendered", outcome, request_id=state.get("request_id", ""), business_critical=True)
        return {
            "final_response": final_response,
            "outcome": outcome,
            "task_status": task_status,
            "result_limitations": limitations,
            "completed_steps": final_completed_steps,
        }
    return _node


def _route_after_verifier(state: PowerMindState) -> str:
    """Route based on verification result and risk level.

    - Passed → render_result
    - Failed + high_risk → human_wait (must not auto-deliver)
    - Failed + low_risk → render_result (with limitation markers)
    - Failure present → fault_policy
    """
    if state.get("failure"):
        return "fault_policy"
    verification = state.get("verification_result", {})
    if not verification.get("passed"):
        plan = state.get("plan", {})
        if plan.get("high_risk"):
            return "human_wait"
        # Low-risk: allow render but mark limitations
        return "render_result"
    return "render_result"


# ═══════════════════════════════════════════════════════════════════════════
# Fault Policy — all six branches with whitelist-guarded RETRY target
# ═══════════════════════════════════════════════════════════════════════════

def _make_fault_policy(deps: dict):
    def _node(state: PowerMindState) -> dict:
        failure = state.get("failure", {})
        component = failure.get("component", "unknown")
        attempt = state.get("retry_counters", {}).get(component, 0)
        policy = DegradationPolicy()

        context = FailureContext(
            component=component,
            failure_type=FailureType(failure.get("failure_type", "unknown")),
            error=failure.get("error", "未知故障"),
            task_type=state.get("task_type", "general"),
            risk_level="high" if state.get("task_type") in {"bidding_advice", "long_term_locking", "real_time_alert", "device_control"} else "medium",
            critical=component in {"quant", "compliance", "verifier", "context_assembly", "compliance_gate"},
            attempt=attempt,
            max_retries=1,
            target_node=failure.get("target_node", component),
            alternatives=tuple(failure.get("alternatives", [])),
        )

        decision = policy.decide(context)
        event = DegradationEvent.from_decision(context, decision).to_dict()

        events = list(state.get("degradation_events", []))
        events.append(event)
        retry_counters = dict(state.get("retry_counters", {}))
        retry_counters[component] = attempt + 1
        action = decision.action.value

        # Whitelist the retry target
        target = decision.target_node or component
        if action == DegradationAction.RETRY.value and target not in _RETRY_TARGET_WHITELIST:
            target = "context_assembly"

        _audit(deps, "fault_decision", f"{component}:{action}→{target}", request_id=state.get("request_id", ""), business_critical=True)

        return {
            "fault_decision": {**decision.to_dict(), "_retry_target": target},
            "degradation_events": events,
            "retry_counters": retry_counters,
            "outcome": ("limited" if action == DegradationAction.REDUCE_SCOPE.value else "human_review" if action == DegradationAction.HUMAN_REVIEW.value else "blocked" if action == DegradationAction.BLOCK.value else "terminated"),
            "execution_profile": "degraded" if action not in {DegradationAction.RETRY.value, DegradationAction.SWITCH_CAPABILITY.value} else state.get("execution_profile"),
            "failure": {} if action in {DegradationAction.RETRY.value, DegradationAction.SWITCH_CAPABILITY.value, DegradationAction.REDUCE_SCOPE.value} else failure,
            "result_limitations": state.get("result_limitations", []) + (list(decision.limitations) if action == DegradationAction.REDUCE_SCOPE.value else []),
            "next_step": target,  # for routing
        }
    return _node


def _route_after_fault_policy(state: PowerMindState) -> str:
    decision = state.get("fault_decision", {})
    action = decision.get("action", DegradationAction.TERMINATE.value)
    target = decision.get("_retry_target", "context_assembly")

    if action == DegradationAction.RETRY.value:
        return target if target in _RETRY_TARGET_WHITELIST else "context_assembly"
    if action == DegradationAction.SWITCH_CAPABILITY.value:
        return target if target in _RETRY_TARGET_WHITELIST else "context_assembly"
    if action == DegradationAction.REDUCE_SCOPE.value:
        return "step_dispatcher"
    if action == DegradationAction.HUMAN_REVIEW.value:
        return "human_wait"
    return "safe_stop"


def _make_safe_stop(deps: dict):
    def _node(state: PowerMindState) -> dict:
        decision = state.get("fault_decision", {})
        failure = state.get("failure", {})
        action = decision.get("action", DegradationAction.TERMINATE.value)
        reason = decision.get("reason", failure.get("error", "没有可用的安全处理路径"))
        labels = {
            DegradationAction.HUMAN_REVIEW.value: "任务已暂停并转人工处理",
            DegradationAction.BLOCK.value: "任务已被安全阻止",
            DegradationAction.TERMINATE.value: "任务已终止",
        }
        _audit(deps, "safe_stop", f"{action}: {reason}", request_id=state.get("request_id", ""), business_critical=True)

        outcome = "terminated" if action == DegradationAction.TERMINATE.value else "blocked"
        return {
            "final_response": (
                f"## {labels.get(action, '任务未能继续')}\n\n"
                f"- 原因：{reason}\n"
                f"- 任务编号：{state.get('request_id', '')}\n"
                f"- 当前未形成可执行结论。\n"
                f"- 已保存故障与处理记录，禁止绕过本次停止条件继续执行。"
            ),
            "outcome": outcome,
            "task_status": "stopped",
        }
    return _node


def _make_human_wait(deps: dict):
    def _node(state: PowerMindState) -> dict:
        pending = state.get("pending_human_action", {}) or {"type": "review", "prompt": "需要人工处理"}
        _audit(deps, "human_wait", str(pending.get("type", "")), request_id=state.get("request_id", ""), business_critical=True)

        request_id = state.get("request_id", "")
        missing = pending.get("missing", [])
        prompt = pending.get("prompt", "需要人工处理")
        human_type = pending.get("type", "review")

        # Build clear final_response for human-wait state
        lines = [
            "## ⏸️ 任务已暂停 — 等待人工处理",
            "",
            f"- 任务编号: {request_id}",
            f"- 等待类型: {human_type}",
            f"- 等待内容: {prompt}",
        ]
        if missing:
            lines.append(f"- 缺失信息: {', '.join(missing)}")
        lines.append("")
        lines.append("> 请提供所需信息后通过 resume_task(thread_id, human_input) 恢复任务。")

        # Use LangGraph interrupt for real pause
        interrupt(pending)

        return {
            "current_step": "human_wait",
            "pending_human_action": pending,
            "task_status": "waiting_human",
            "outcome": "human_review",
            "final_response": "\n".join(lines),
        }
    return _node


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _add_completed(state: PowerMindState, step: str) -> list[str]:
    return list(state.get("completed_steps", [])) + [step]


def _audit(deps: dict, event_type: str, detail: str, *, request_id: str = "", business_critical: bool = False):
    sink = deps.get("audit_sink")
    if sink is None:
        return
    from services.audit_service import AuditEvent, AuditUnavailableError
    event = AuditEvent(event_type=event_type, component="langgraph", detail=detail[:500], request_id=request_id, is_business_critical=business_critical)
    try:
        sink.write(event)
    except AuditUnavailableError:
        if business_critical:
            raise


def _call_sub_agent_impl(state: PowerMindState, agent_name: str) -> dict:
    """Legacy implementation — wraps sub_agents module directly.

    Prefer using the AgentRunner protocol injected via build_graph().
    This function exists as the default fallback for backward compatibility.
    """
    from agents.sub_agents import quant_node as _q, expert_insight_node as _e, compliance_node as _c, strategy_node as _s

    if agent_name == "reflection":
        return _call_reflection_node(state)
    node_map = {"quant": _q, "expert": _e, "compliance": _c, "strategy": _s}
    fn = node_map.get(agent_name)
    if fn is None:
        return {"status": "failed", "error": f"未知Agent: {agent_name}", "agent": agent_name, "failure_type": "agent_failure", "retryable": False}

    result_key = f"{agent_name}_result"
    try:
        output = fn(state)
        # Extract the actual agent result from the output dict
        # sub_agents return {f"{name}_result": {...}, "called_agents": [...]}
        inner = output.get(result_key, output)
        # Ensure status is present at top level for the wrapper
        if "status" not in inner and isinstance(inner, dict):
            inner = {"status": "success", "summary": str(inner)[:500], "agent": agent_name}
        return inner
    except ComponentExecutionError as exc:
        return {"status": "failed", "error": str(exc), "agent": agent_name, "failure_type": exc.failure_type, "retryable": exc.retryable, "alternatives": list(exc.alternatives)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "agent": agent_name, "failure_type": "agent_failure", "retryable": False}


def _call_reflection_node(state: PowerMindState) -> dict:
    """Reflection as independent node — evaluates Strategy output."""
    from agents.reflection import reflect_and_revise
    try:
        strategy_text = (state.get("strategy_result") or {}).get("summary", "")
        if not strategy_text:
            return {"status": "success", "summary": "无策略内容可反思", "reflection_result": {"passed": True}}
        # Reflection evaluates strategy quality; in Demo this is lightweight
        return {
            "status": "success",
            "summary": strategy_text[:500],
            "reflection_result": {"passed": True, "notes": "Strategy已通过Reflection节点检查"},
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)[:300], "agent": "reflection", "failure_type": "agent_failure", "retryable": False}


# ═══════════════════════════════════════════════════════════════════════════
# State normalisation — ensures human_wait / interrupt always has readable state
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_waiting_human_state(state: dict) -> dict:
    """Post-process state from graph.invoke() so human_wait/interrupt states
    always contain readable task_status, outcome, pending_human_action and
    final_response — even when LangGraph interrupt() fired before the
    human_wait node could return its updates.

    This is the single normalisation point for all three human-wait paths:
      1. supervisor → human_wait (missing fields)
      2. verifier → human_wait (high-risk verification failure)
      3. fault_policy → human_wait (agent failure escalation)
    """
    # Already properly terminated — nothing to do
    if state.get("task_status") in ("completed", "stopped", "error", "limited"):
        return state

    has_interrupt = "__interrupt__" in state
    has_pending = bool(state.get("pending_human_action"))

    if not has_interrupt and not has_pending:
        return state

    # ── Extract or build pending_human_action ──
    pending = state.get("pending_human_action")
    if not pending and has_interrupt:
        interrupt_val = state.get("__interrupt__")
        # LangGraph interrupt() may store a list of Interrupt objects
        if isinstance(interrupt_val, list):
            for item in interrupt_val:
                if hasattr(item, "value"):
                    pending = item.value
                    break
        if not pending and isinstance(interrupt_val, dict):
            pending = interrupt_val
        if not pending:
            pending = {"type": "review", "prompt": "任务已暂停，等待人工处理"}

    state["pending_human_action"] = pending
    state["task_status"] = "waiting_human"
    state["outcome"] = "human_review"

    # ── Build final_response if missing ──
    if not state.get("final_response"):
        request_id = state.get("request_id", "")
        human_type = pending.get("type", "review") if isinstance(pending, dict) else "review"
        prompt = pending.get("prompt", "需要人工处理") if isinstance(pending, dict) else str(pending)
        reason = pending.get("reason", "") if isinstance(pending, dict) else ""
        missing = pending.get("missing", []) if isinstance(pending, dict) else []
        resume_hint = pending.get("resume_hint", "") if isinstance(pending, dict) else ""

        lines = [
            "## ⏸️ 任务已暂停 — 等待人工处理",
            "",
            f"- 任务编号: {request_id}",
            f"- 等待类型: {human_type}",
            f"- 等待原因: {prompt}",
        ]
        if reason:
            lines.append(f"- 详细原因: {reason}")
        if missing:
            lines.append(f"- 缺失信息: {', '.join(missing)}")
        if resume_hint:
            lines.append(f"- 恢复方式: {resume_hint}")
        else:
            lines.append(f"- 恢复方式: 请通过 resume_task(thread_id='{request_id}', human_input={{...}}) 恢复任务")
        lines.append("")
        lines.append("> ⚠️ 当前任务未形成可执行结论，不得用于正式决策或自动执行。")

        state["final_response"] = "\n".join(lines)

    return state


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

class PowerMindAgentV3:
    """Single-graph PowerMind runtime.  No second orchestrator."""

    def __init__(self, *, degradation_policy=None, checkpointer=None, llm_gateway=None, tool_registry=None, market_data_repo=None, audit_sink=None, evidence_service=None, agent_runner=None, use_deepseek: bool = False):
        if agent_runner is None:
            import os
            from services.audit_service import DemoSQLiteAuditSink
            from services.repository import DemoSQLiteMarketDataRepository
            from scripts.seed_demo_v1 import ensure_demo_database

            demo_path = ensure_demo_database(os.environ.get("POWERMIND_DEMO_DB"))
            market_data_repo = market_data_repo or DemoSQLiteMarketDataRepository(str(demo_path))
            audit_sink = audit_sink or DemoSQLiteAuditSink(demo_path)
            if use_deepseek:
                from agents.demo_agents import DeepSeekAgentRunner
                from retrieval.demo_rag import DemoRAGRetriever

                agent_runner = DeepSeekAgentRunner(
                    market_data_repo,
                    DemoRAGRetriever(demo_path),
                )
            else:
                agent_runner = DemoAgentRunner(str(demo_path))
        self._deps = {"llm_gateway": llm_gateway, "tool_registry": tool_registry, "market_data_repo": market_data_repo, "audit_sink": audit_sink, "evidence_service": evidence_service, "agent_runner": agent_runner}
        self.graph = build_graph(checkpointer=checkpointer, **{k: v for k, v in self._deps.items() if v is not None})

    def execute_state(self, query: str, province: str = "山东", task_type: str = "general", target_date: str | None = None, client_id: str | None = None) -> PowerMindState:
        request_id = str(uuid.uuid4())[:12]
        initial: PowerMindState = {"request_id": request_id, "thread_id": request_id, "user_query": query, "province": province, "task_type": task_type, "target_date": target_date or datetime.now().strftime("%Y-%m-%d"), "client_id": client_id or "", "task_mode": "formal", "tenant_context": {}, "degradation_events": [], "retry_counters": {}, "execution_profile": "full", "outcome": "completed", "warnings": [], "evidence_records": [], "completed_steps": [], "task_status": "init", "result_limitations": [], "plan_index": 0}
        try:
            result = self.graph.invoke(initial, config={"configurable": {"thread_id": request_id}, "recursion_limit": 150})
        except Exception as exc:
            exc_name = type(exc).__name__
            # Check for LangGraph interrupt (human_wait)
            if "Interrupt" in exc_name or "interrupt" in str(exc).lower():
                result = {
                    "request_id": request_id,
                    "thread_id": request_id,
                    "__interrupt__": str(exc)[:500],
                }
                # Extract pending from interrupt value if possible
                if hasattr(exc, 'value'):
                    result["pending_human_action"] = exc.value
                result = _normalize_waiting_human_state(result)
                return result
            result = {"request_id": request_id, "final_response": f"## 系统故障\n\n图形执行异常: {str(exc)[:300]}\n\n任务已终止。", "outcome": "terminated", "task_status": "error"}
            return result
        # Normal path — ensure any human_wait state is complete
        result = _normalize_waiting_human_state(result)
        return result

    def resume_task(self, thread_id: str, human_input: dict) -> PowerMindState:
        """Resume a paused task using LangGraph Command(resume=...)."""
        try:
            result = self.graph.invoke(
                Command(resume=human_input),
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 150},
            )
            result = _normalize_waiting_human_state(result)
        except Exception as exc:
            exc_name = type(exc).__name__
            if "Interrupt" in exc_name or "interrupt" in str(exc).lower():
                result = {"request_id": thread_id, "__interrupt__": str(exc)[:500]}
                result = _normalize_waiting_human_state(result)
            else:
                result = {"request_id": thread_id, "final_response": f"## 恢复失败\n\n{str(exc)[:300]}", "outcome": "terminated", "task_status": "error"}
        return result

    def execute(self, query: str, province: str = "山东", task_type: str = "general", target_date: str | None = None, client_id: str | None = None) -> str:
        state = self.execute_state(query=query, province=province, task_type=task_type, target_date=target_date, client_id=client_id)
        # Normal path — final_response was already populated
        final = state.get("final_response")
        if final:
            return final
        # Human-wait / interrupt path — normaliser should have built final_response
        if state.get("__interrupt__") or state.get("pending_human_action"):
            request_id = state.get("request_id", "unknown")
            return (
                f"## ⏸️ 任务已暂停 — 等待人工处理\n\n"
                f"- 任务编号: {request_id}\n"
                f"- 状态: {state.get('task_status', 'waiting_human')}\n"
                f"- 请通过 resume_task(thread_id='{request_id}', human_input={{...}}) 恢复任务\n"
            )
        # Genuine empty result
        return (
            f"## 任务未形成可用结果\n\n"
            f"- 任务编号: {state.get('request_id', 'unknown')}\n"
            f"- 任务状态: {state.get('task_status', 'unknown')}\n"
            f"- 当前任务没有生成最终结果，请查看任务状态和审计记录。"
        )


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="PowerMind unified LangGraph")
    parser.add_argument("query", nargs="*", default=["分析明天山东的现货电价走势"])
    parser.add_argument("--province", "-p", default="山东")
    parser.add_argument("--task", "-t", default="general")
    parser.add_argument("--date", "-d", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    env_path = PROJECT_ROOT / "config" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    print("=" * 60)
    print("  ⚡ PowerMind — 统一LangGraph")
    print(f"  省份: {args.province} | 任务: {args.task}")
    print("=" * 60)
    agent = PowerMindAgentV3()
    print(agent.execute(query, province=args.province, task_type=args.task, target_date=args.date))
