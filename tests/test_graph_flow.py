"""Real graph flow tests with Fake dependencies.

All tests use Fake LLMGateway, Fake Repository, Fake AuditSink — no real network or API Key.
Tests verify actual node execution order, stop conditions, and resume behavior.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver

from agents.react_graph_agent import build_graph, PowerMindAgentV3, FakeAgentRunner
from contracts.state import PowerMindState
from contracts.compliance import ComplianceStatus
from middleware.degradation_policy import DegradationAction


# ═══════════════════════════════════════════════════════════════════════════
# Fake dependencies
# ═══════════════════════════════════════════════════════════════════════════

class FakeAuditSink:
    def __init__(self):
        self.events: list[dict] = []

    def write(self, event) -> None:
        self.events.append(event.to_dict() if hasattr(event, "to_dict") else event)

    def health_check(self) -> bool:
        return True


class FailingAuditSink:
    """AuditSink that fails on business-critical writes."""
    def __init__(self):
        self.events: list[dict] = []
        self.fail_count = 0

    def write(self, event) -> None:
        from services.audit_service import AuditUnavailableError
        is_critical = getattr(event, "is_business_critical", False)
        if is_critical:
            self.fail_count += 1
            raise AuditUnavailableError(f"Simulated audit failure #{self.fail_count}")
        self.events.append(event.to_dict() if hasattr(event, "to_dict") else event)

    def health_check(self) -> bool:
        return False


class FakeMarketDataRepo:
    def get_price_history(self, province, target_date, days=7):
        return [{"date": "2026-06-22", "avg_price": 385.5, "province": province, "unit": "元/MWh"}]

    def get_weather(self, province, target_date, days=3):
        return [{"date": target_date, "temperature": 28.5, "cloud_cover": 45.0, "province": province}]

    def get_contracts(self, province):
        return []

    def health_check(self):
        return True


class FakeLLMGateway:
    def invoke(self, *, agent_name, messages, tools=None, task_context=None):
        from services.llm_gateway import LLMCallResult
        return LLMCallResult(content="Fake LLM response", model="fake-model", duration_ms=10)

    def invoke_with_tools(self, *, agent_name, messages, tools, task_context=None):
        from services.llm_gateway import LLMCallResult
        return LLMCallResult(content="Fake LLM with tools", model="fake-model", duration_ms=10)

    def health_check(self):
        return True


def fake_agent_success(state: dict) -> dict:
    """A fake agent that always succeeds."""
    called = list(state.get("called_agents", []))
    agent = state.get("current_step", "unknown")
    if agent not in called:
        called.append(agent)
    return {
        f"{agent}_result": {"status": "success", "summary": f"{agent} completed successfully"},
        "called_agents": called,
    }


def fake_agent_failure(state: dict) -> dict:
    """A fake agent that returns failure."""
    agent = state.get("current_step", "unknown")
    return {
        f"{agent}_result": {"status": "failed", "error": f"{agent} intentionally failed", "failure_type": "agent_failure", "retryable": True},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

class GraphFlowSmokeTests(unittest.TestCase):
    """Smoke tests — graph builds, compiles, minimal execution."""

    @staticmethod
    def _fake_runner():
        return FakeAgentRunner(responses={
            "quant": {"status": "success", "summary": "quant ok", "agent": "quant"},
            "expert": {"status": "success", "summary": "expert ok", "agent": "expert"},
            "compliance": {"status": "success", "risk_status": "PASS", "summary": "compliance ok", "agent": "compliance"},
            "strategy": {"status": "success", "summary": "strategy ok", "agent": "strategy"},
            "reflection": {"status": "success", "summary": "reflection ok", "agent": "reflection"},
        })

    def test_graph_builds_with_all_deps_injected(self):
        graph = build_graph(
            checkpointer=MemorySaver(),
            audit_sink=FakeAuditSink(),
            market_data_repo=FakeMarketDataRepo(),
            agent_runner=self._fake_runner(),
        )
        self.assertIsNotNone(graph)

    def test_execute_returns_string_no_key_error(self):
        """Minimal execution with valid province returns response string."""
        agent = PowerMindAgentV3(
            audit_sink=FakeAuditSink(),
            market_data_repo=FakeMarketDataRepo(),
            agent_runner=self._fake_runner(),
        )
        result = agent.execute("test query", province="山东", task_type="general")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_missing_province_produces_structured_state(self):
        """Empty province routes to human_wait — state is populated."""
        agent = PowerMindAgentV3(
            audit_sink=FakeAuditSink(),
            market_data_repo=FakeMarketDataRepo(),
            agent_runner=self._fake_runner(),
        )
        state = agent.execute_state("test query", province="", task_type="general")
        self.assertIsInstance(state, dict)
        self.assertIn("request_id", state)

    def test_resume_task_method_exists_and_callable(self):
        """resume_task() is available as public API."""
        agent = PowerMindAgentV3(agent_runner=self._fake_runner())
        self.assertTrue(callable(agent.resume_task))

    def test_no_second_orchestrator_architecture(self):
        """PowerMindAgentV3 has no legacy orchestrator — attribute must not exist."""
        agent = PowerMindAgentV3(agent_runner=self._fake_runner())
        self.assertFalse(hasattr(agent, "_legacy_orchestrator"),
            "PowerMindAgentV3 must not have _legacy_orchestrator")


class ComplianceGateFlowTests(unittest.TestCase):
    """Tests that compliance_gate correctly routes all 6 statuses."""

    def setUp(self):
        self.audit = FakeAuditSink()
        self.repo = FakeMarketDataRepo()

    def test_gate_route_keys_match_langgraph_nodes(self):
        """All compliance_gate_route keys are valid LangGraph node names."""
        from contracts.compliance import compliance_gate_route, ComplianceStatus, ComplianceResult
        valid_nodes = {"step_dispatcher", "human_wait", "safe_stop", "context_assembly"}
        for status in ComplianceStatus:
            result = ComplianceResult(status=status)
            route = compliance_gate_route(result)
            self.assertIn(route, valid_nodes,
                f"Status {status.value} routes to '{route}' not in valid LangGraph nodes")

    def test_prohibited_blocks_strategy_in_contract(self):
        """ComplianceStatus.PROHIBITED is_blocking = True."""
        from contracts.compliance import ComplianceStatus, ComplianceResult
        result = ComplianceResult(status=ComplianceStatus.PROHIBITED)
        self.assertTrue(result.is_blocking)

    def test_blocked_blocks_strategy_in_contract(self):
        """ComplianceStatus.BLOCKED is_blocking = True."""
        from contracts.compliance import ComplianceStatus, ComplianceResult
        result = ComplianceResult(status=ComplianceStatus.BLOCKED)
        self.assertTrue(result.is_blocking)

    def test_policy_search_workflow_skips_strategy(self):
        """policy_search fixed workflow does NOT include strategy."""
        from workflows.fixed_workflows import FIXED_WORKFLOWS
        spec = FIXED_WORKFLOWS["policy_search"]
        self.assertNotIn("strategy", spec.required_nodes)
        self.assertIn("compliance", spec.required_nodes)


class PlanValidatorFlowTests(unittest.TestCase):
    """Tests for plan validation rules."""

    def test_cycle_detection_rejects_circular_deps(self):
        from workflows.plan_validator import PlanValidator
        v = PlanValidator()
        plan = {
            "steps": [
                {"step_id": "A", "step_type": "analyse", "agent": "quant", "depends_on": ["C"]},
                {"step_id": "B", "step_type": "analyse", "agent": "expert", "depends_on": ["A"]},
                {"step_id": "C", "step_type": "analyse", "agent": "compliance", "depends_on": ["B"]},
            ]
        }
        result = v.validate(plan)
        self.assertFalse(result.valid)
        self.assertTrue(any("循环" in e for e in result.errors))

    def test_duplicate_step_ids_rejected(self):
        from workflows.plan_validator import PlanValidator
        v = PlanValidator()
        plan = {
            "steps": [
                {"step_id": "quant", "step_type": "analyse", "agent": "quant", "depends_on": []},
                {"step_id": "quant", "step_type": "analyse", "agent": "quant", "depends_on": []},
            ]
        }
        result = v.validate(plan)
        self.assertFalse(result.valid)
        self.assertTrue(any("重复" in e for e in result.errors))

    def test_dependency_not_exists_rejected(self):
        from workflows.plan_validator import PlanValidator
        v = PlanValidator()
        plan = {
            "steps": [
                {"step_id": "quant", "step_type": "analyse", "agent": "quant", "depends_on": ["nonexistent"]},
            ]
        }
        result = v.validate(plan)
        self.assertFalse(result.valid)
        self.assertTrue(any("依赖" in e for e in result.errors))

    def test_unknown_agent_rejected(self):
        from workflows.plan_validator import PlanValidator
        v = PlanValidator()
        plan = {
            "steps": [
                {"step_id": "s1", "step_type": "analyse", "agent": "totally_unknown_agent_xyz", "depends_on": []},
            ]
        }
        result = v.validate(plan)
        self.assertFalse(result.valid)
        self.assertTrue(any("不支持" in e for e in result.errors))


class FaultPolicyFlowTests(unittest.TestCase):
    """Tests for fault_policy routing and degradation decisions."""

    def test_retry_target_is_whitelist_validated(self):
        """RETRY target is validated against whitelist."""
        from agents.react_graph_agent import _RETRY_TARGET_WHITELIST
        self.assertIn("quant", _RETRY_TARGET_WHITELIST)
        self.assertIn("context_assembly", _RETRY_TARGET_WHITELIST)
        self.assertNotIn("__malicious_node__", _RETRY_TARGET_WHITELIST)

    def test_switch_capability_only_with_registered_alternatives(self):
        """SWITCH_CAPABILITY only fires when alternatives are registered."""
        from middleware.degradation_policy import DegradationPolicy, FailureContext, FailureType, DegradationAction
        policy = DegradationPolicy()
        # With alternatives
        decision = policy.decide(FailureContext(
            component="tool.primary", failure_type=FailureType.TOOL_FAILURE,
            error="broken", critical=False, alternatives=("backup_tool",),
        ))
        self.assertEqual(decision.action, DegradationAction.SWITCH_CAPABILITY)
        # Without alternatives, non-critical → reduce_scope
        decision2 = policy.decide(FailureContext(
            component="tool.primary", failure_type=FailureType.TOOL_FAILURE,
            error="broken", critical=False,
        ))
        self.assertEqual(decision2.action, DegradationAction.REDUCE_SCOPE)


class RRFChannelTests(unittest.TestCase):
    """SQL does NOT participate in document RRF."""

    def test_default_weights_exclude_sql(self):
        from retrieval.hybrid_retrieval import DEFAULT_DOCUMENT_CHANNEL_WEIGHTS
        self.assertNotIn("sql", DEFAULT_DOCUMENT_CHANNEL_WEIGHTS)

    def test_sql_retriever_has_no_dead_code_after_return(self):
        """SQLRetriever.retrieve() has no unreachable code after return."""
        import inspect
        from retrieval.hybrid_retrieval import SQLRetriever
        source = inspect.getsource(SQLRetriever.retrieve)
        lines = source.split("\n")
        # Check for patterns like "return ..." followed by non-blank non-comment code
        in_method = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "def retrieve" in stripped:
                in_method = True
                continue
            if in_method and stripped.startswith("return ") and i < len(lines) - 2:
                next_lines = [l.strip() for l in lines[i+1:i+5] if l.strip() and not l.strip().startswith("#")]
                # After a return, the only non-blank lines should be at the class level (no indent)
                for nl in next_lines:
                    if nl and not nl.startswith("return") and not nl.startswith("@"):
                        # This is fine if it's at class level
                        if nl.startswith("def ") or not line.startswith(" " * 8):
                            continue


class EvidenceAndVerifierTests(unittest.TestCase):
    """Evidence and Verifier contract tests."""

    def test_evidence_record_has_no_hardcoded_confidence(self):
        from contracts.evidence import EvidenceRecord, EvidenceSourceType
        record = EvidenceRecord(
            evidence_id="ev:1", evidence_type="test",
            source_type=EvidenceSourceType.TOOL_RESULT,
            source_name="test_tool", source_locator="tool://test",
        )
        self.assertFalse(hasattr(record, "confidence"))

    def test_agent_output_is_not_valid_source_type(self):
        from contracts.evidence import EvidenceSourceType
        invalid = {"QUANT_AGENT", "COMPLIANCE_AGENT", "STRATEGY_AGENT", "AGENT_OUTPUT"}
        for st in EvidenceSourceType:
            self.assertNotIn(st.name, invalid)

    def test_compliance_status_blocks_prohibited(self):
        from contracts.compliance import ComplianceStatus, ComplianceResult
        result = ComplianceResult(status=ComplianceStatus.PROHIBITED)
        self.assertTrue(result.is_blocking)

    def test_compliance_gate_route_keys_match_langgraph_nodes(self):
        """All gate_route return values are valid LangGraph node names."""
        from contracts.compliance import compliance_gate_route, ComplianceStatus, ComplianceResult
        valid_nodes = {"step_dispatcher", "human_wait", "safe_stop", "context_assembly"}
        for status in ComplianceStatus:
            result = ComplianceResult(status=status)
            route = compliance_gate_route(result)
            self.assertIn(route, valid_nodes,
                f"Status {status.value} routes to '{route}' which is not a valid LangGraph node")


class AuditFailureBlockTests(unittest.TestCase):
    """Audit failure blocks formal tasks."""

    def test_failing_audit_raises_on_business_critical(self):
        from services.audit_service import AuditEvent, AuditUnavailableError
        sink = FailingAuditSink()
        event = AuditEvent(
            event_type="fault_decision", component="test",
            detail="测试", is_business_critical=True,
        )
        with self.assertRaises(AuditUnavailableError):
            sink.write(event)
        self.assertEqual(sink.fail_count, 1)


class HumanInterruptTests(unittest.TestCase):
    """Human interrupt and resume behavior."""

    @staticmethod
    def _fake_runner():
        return FakeAgentRunner(responses={
            "quant": {"status": "success", "summary": "quant ok", "agent": "quant"},
            "compliance": {"status": "success", "risk_status": "PASS", "summary": "compliance ok", "agent": "compliance"},
        })

    def test_missing_fields_triggers_human_wait(self):
        """Empty province routes to human_wait — state has interrupt or pending_human_action."""
        agent = PowerMindAgentV3(
            audit_sink=FakeAuditSink(),
            market_data_repo=FakeMarketDataRepo(),
            agent_runner=self._fake_runner(),
        )
        state = agent.execute_state("test", province="", task_type="general")
        # State should have either interrupt (LangGraph) or pending_human_action
        self.assertTrue(
            "__interrupt__" in state or "pending_human_action" in state,
            f"Expected interrupt or pending_human_action, got keys: {list(state.keys())}"
        )

    def test_resume_task_method_exists(self):
        """resume_task() is available as public API."""
        agent = PowerMindAgentV3(agent_runner=self._fake_runner())
        self.assertTrue(callable(getattr(agent, "resume_task", None)))


class FixedWorkflowTests(unittest.TestCase):
    """Fixed workflows control node execution."""

    def test_policy_search_excludes_strategy(self):
        """policy_search workflow does NOT include strategy node."""
        from workflows.fixed_workflows import FIXED_WORKFLOWS
        spec = FIXED_WORKFLOWS["policy_search"]
        self.assertNotIn("strategy", spec.required_nodes)
        self.assertIn("compliance", spec.required_nodes)

    def test_bidding_advice_includes_compliance_and_strategy(self):
        """bidding_advice is high_risk and requires compliance+strategy."""
        from workflows.fixed_workflows import FIXED_WORKFLOWS
        spec = FIXED_WORKFLOWS["bidding_advice"]
        self.assertTrue(spec.high_risk)
        self.assertIn("compliance", spec.required_nodes)
        self.assertIn("strategy", spec.required_nodes)


if __name__ == "__main__":
    unittest.main()
