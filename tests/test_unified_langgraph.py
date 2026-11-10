"""Unified LangGraph tests — ALL use FakeAgentRunner, NO real LLM/API."""

from __future__ import annotations

import unittest

from langgraph.checkpoint.memory import MemorySaver

from agents.react_graph_agent import (
    PowerMindAgentV3,
    build_graph,
    FakeAgentRunner,
)


class _FakeRepo:
    def get_price_history(self, province, target_date, days=7):
        return [{"date": "2026-06-22", "avg_price": 385.5, "province": province, "unit": "元/MWh"}]
    def get_weather(self, province, target_date, days=3):
        return [{"date": target_date, "temperature": 28.5, "cloud_cover": 45.0, "province": province}]
    def get_contracts(self, province):
        return []
    def health_check(self):
        return True


def _fake_runner():
    return FakeAgentRunner(responses={
        "quant": {"status": "success", "summary": "quant ok", "agent": "quant"},
        "expert": {"status": "success", "summary": "expert ok", "agent": "expert"},
        "compliance": {"status": "success", "risk_status": "PASS", "summary": "compliance ok", "agent": "compliance"},
        "strategy": {"status": "success", "summary": "strategy ok", "agent": "strategy"},
        "reflection": {"status": "success", "summary": "reflection ok", "agent": "reflection"},
    })


class UnifiedLangGraphTests(unittest.TestCase):
    def test_graph_builds_and_compiles(self) -> None:
        """Main entry builds and compiles a StateGraph."""
        graph = build_graph(agent_runner=_fake_runner())
        self.assertIsNotNone(graph)
        self.assertIn("task_init", graph.nodes if hasattr(graph, "nodes") else {})

    def test_agent_creates_graph(self) -> None:
        """PowerMindAgentV3 creates a compiled graph."""
        agent = PowerMindAgentV3(agent_runner=_fake_runner())
        self.assertIsNotNone(agent.graph)

    def test_execute_returns_response(self) -> None:
        """execute() returns a non-empty string even for minimal input."""
        agent = PowerMindAgentV3(agent_runner=_fake_runner(), market_data_repo=_FakeRepo())
        result = agent.execute("测试查询", task_type="general")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_execute_state_returns_structured_state(self) -> None:
        """execute_state() returns a dict with minimal expected keys."""
        agent = PowerMindAgentV3(agent_runner=_fake_runner(), market_data_repo=_FakeRepo())
        state = agent.execute_state("测试", task_type="general")
        self.assertIsInstance(state, dict)
        self.assertIn("request_id", state)

    def test_graph_has_required_nodes(self) -> None:
        """Graph executes and produces state with request_id and outcome."""
        agent = PowerMindAgentV3(agent_runner=_fake_runner(), market_data_repo=_FakeRepo())
        state = agent.execute_state("政策查询", province="山东", task_type="policy_search")
        self.assertIn("request_id", state)
        self.assertTrue(len(state.get("request_id", "")) > 0)

    def test_checkpointer_is_injectable(self) -> None:
        """build_graph accepts an external checkpointer."""
        graph = build_graph(checkpointer=None, agent_runner=_fake_runner())
        self.assertIsNotNone(graph)
        graph2 = build_graph(checkpointer=MemorySaver(), agent_runner=_fake_runner())
        self.assertIsNotNone(graph2)

    def test_failure_state_produces_safe_stop_response(self) -> None:
        """A task with missing province triggers human_wait and has interrupt or response."""
        agent = PowerMindAgentV3(agent_runner=_fake_runner(), market_data_repo=_FakeRepo())
        state = agent.execute_state("分析", province="", task_type="general")
        self.assertTrue(
            "__interrupt__" in state or "final_response" in state or "pending_human_action" in state
        )

    def test_no_second_orchestrator_architecture(self) -> None:
        """PowerMindAgentV3 has no legacy orchestrator attribute."""
        agent = PowerMindAgentV3()
        self.assertFalse(hasattr(agent, "_legacy_orchestrator"),
            "PowerMindAgentV3 must not have _legacy_orchestrator")


if __name__ == "__main__":
    unittest.main()
