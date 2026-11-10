"""Human-wait / interrupt state normalisation tests.

Verifies that ALL paths into human_wait produce readable state with
task_status, outcome, pending_human_action, and final_response.
"""

from __future__ import annotations

import unittest

from agents.react_graph_agent import (
    PowerMindAgentV3,
    FakeAgentRunner,
    _normalize_waiting_human_state,
)


class _FakeRepo:
    def get_price_history(self, p, d, days=7):
        return [{"date": "2026-06-22", "avg_price": 385.5, "province": p, "unit": "元/MWh"}]
    def get_weather(self, p, d, days=3):
        return [{"date": d, "temperature": 28.5, "cloud_cover": 45.0, "province": p}]
    def get_contracts(self, p):
        return []


class _EmptyRepo:
    def get_price_history(self, p, d, days=7): return []
    def get_weather(self, p, d, days=3): return []
    def get_contracts(self, p): return []


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Missing province → readable human-wait state
# ═══════════════════════════════════════════════════════════════════════════

class MissingProvinceHumanWaitTests(unittest.TestCase):
    """Supervisor → human_wait when province is missing."""

    def test_execute_state_returns_waiting_human_status(self):
        """Missing province → task_status=waiting_human, outcome=human_review."""
        agent = PowerMindAgentV3(agent_runner=FakeAgentRunner(), market_data_repo=_FakeRepo())
        state = agent.execute_state("analysis", province="", task_type="general")

        self.assertEqual(state.get("task_status"), "waiting_human",
            f"Expected waiting_human, got: {state.get('task_status')}")
        self.assertEqual(state.get("outcome"), "human_review",
            f"Expected human_review, got: {state.get('outcome')}")

    def test_execute_state_has_pending_human_action(self):
        """Missing province → pending_human_action is populated."""
        agent = PowerMindAgentV3(agent_runner=FakeAgentRunner(), market_data_repo=_FakeRepo())
        state = agent.execute_state("analysis", province="", task_type="general")

        pending = state.get("pending_human_action")
        self.assertIsNotNone(pending)
        self.assertIsInstance(pending, dict)
        self.assertIn("type", pending)

    def test_execute_state_has_final_response(self):
        """Missing province → final_response is non-empty."""
        agent = PowerMindAgentV3(agent_runner=FakeAgentRunner(), market_data_repo=_FakeRepo())
        state = agent.execute_state("analysis", province="", task_type="general")

        final = state.get("final_response", "")
        self.assertTrue(final, f"final_response must be non-empty")
        self.assertTrue("等待" in final or "暂停" in final,
            f"final_response should contain wait/pause marker, got: {final[:200]}")

    def test_execute_does_not_return_fallback_nonsense(self):
        """execute() must NOT return the blanket fallback text."""
        agent = PowerMindAgentV3(agent_runner=FakeAgentRunner(), market_data_repo=_FakeRepo())
        text = agent.execute("analysis", province="", task_type="general")

        self.assertTrue(text)
        self.assertNotIn("任务没有返回可用结果", text)
        self.assertTrue("等待" in text or "暂停" in text,
            f"execute() should mention wait/pause, got: {text[:200]}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: High-risk verifier failure → human_review state
# ═══════════════════════════════════════════════════════════════════════════

class HighRiskVerifierFailureTests(unittest.TestCase):
    """Verifier → human_wait for high-risk tasks with failed verification."""

    def setUp(self):
        self.runner = FakeAgentRunner(responses={
            "quant": {"status": "success", "summary": "quant ok", "agent": "quant"},
            "expert": {"status": "success", "summary": "expert ok", "agent": "expert"},
            "compliance": {"status": "success", "risk_status": "PASS", "summary": "ok", "agent": "compliance"},
            "strategy": {"status": "success", "summary": "strategy ok", "agent": "strategy"},
            "reflection": {"status": "success", "summary": "reflection ok", "agent": "reflection"},
        })

    def test_verifier_fail_sets_waiting_human(self):
        """High-risk + no evidence → task_status=waiting_human, outcome=human_review."""
        agent = PowerMindAgentV3(agent_runner=self.runner, market_data_repo=_EmptyRepo())
        state = agent.execute_state("generate bidding strategy", province="Shandong", task_type="bidding_advice")

        self.assertIn("strategy", self.runner.calls)
        vr = state.get("verification_result", {})
        self.assertFalse(vr.get("passed"),
            f"Verification should fail with no evidence, got: {vr}")

        self.assertEqual(state.get("task_status"), "waiting_human",
            f"Expected waiting_human, got: {state.get('task_status')}")
        self.assertEqual(state.get("outcome"), "human_review",
            f"Expected human_review, got: {state.get('outcome')}")

    def test_verifier_fail_has_pending_human_action(self):
        """High-risk verifier fail → pending_human_action has review context."""
        agent = PowerMindAgentV3(agent_runner=self.runner, market_data_repo=_EmptyRepo())
        state = agent.execute_state("generate bidding strategy", province="Shandong", task_type="bidding_advice")

        pending = state.get("pending_human_action")
        self.assertIsNotNone(pending, "pending_human_action must be set")
        self.assertIsInstance(pending, dict)
        self.assertIn("type", pending)
        # Should be verification_review type
        self.assertIn(pending.get("type"), ("verification_review", "review"),
            f"Expected verification_review or review, got: {pending.get('type')}")

    def test_verifier_fail_has_final_response(self):
        """High-risk verifier fail → final_response mentions manual review."""
        agent = PowerMindAgentV3(agent_runner=self.runner, market_data_repo=_EmptyRepo())
        state = agent.execute_state("generate bidding strategy", province="Shandong", task_type="bidding_advice")

        final = state.get("final_response", "")
        self.assertTrue(final, "final_response must be non-empty")
        # Should mention manual review or pause
        keywords = ["人工", "复核", "等待", "暂停"]
        has_kw = any(kw in final for kw in keywords)
        self.assertTrue(has_kw,
            f"final_response should contain manual-review marker, got: {final[:200]}")

    def test_verifier_fail_execute_does_not_return_fallback(self):
        """execute() for high-risk verifier fail must NOT return blanket fallback."""
        agent = PowerMindAgentV3(agent_runner=self.runner, market_data_repo=_EmptyRepo())
        text = agent.execute("generate bidding strategy", province="Shandong", task_type="bidding_advice")

        self.assertTrue(text)
        self.assertNotIn("任务没有返回可用结果", text)
        keywords = ["人工", "复核", "等待", "暂停"]
        has_kw = any(kw in text for kw in keywords)
        self.assertTrue(has_kw,
            f"execute() should contain manual-review marker, got: {text[:200]}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Quant persistent failure → human_wait with final_response
# ═══════════════════════════════════════════════════════════════════════════

class QuantFailureHumanWaitTests(unittest.TestCase):
    """Fault policy → human_wait when quant fails persistently."""

    def test_quant_transient_fail_has_final_response(self):
        """Quant transient failure escalates to human_wait with final_response."""
        runner = FakeAgentRunner(responses={
            "quant": {"status": "failed", "error": "quant timeout", "agent": "quant",
                       "failure_type": "transient", "retryable": True},
            "expert": {"status": "success", "summary": "expert ok", "agent": "expert"},
            "compliance": {"status": "success", "risk_status": "PASS", "summary": "ok", "agent": "compliance"},
            "strategy": {"status": "success", "summary": "strategy ok", "agent": "strategy"},
            "reflection": {"status": "success", "summary": "reflection ok", "agent": "reflection"},
        })
        agent = PowerMindAgentV3(agent_runner=runner, market_data_repo=_FakeRepo())
        state = agent.execute_state("give me strategy", province="Shandong", task_type="bidding_advice")

        # Expert should NOT be called before quant succeeds
        self.assertNotIn("expert", runner.calls,
            f"Expert must not be called after quant failure. Calls: {runner.calls}")

        # Must have terminal state
        self.assertEqual(state.get("task_status"), "waiting_human",
            f"Expected waiting_human, got: {state.get('task_status')}")
        self.assertEqual(state.get("outcome"), "human_review",
            f"Expected human_review, got: {state.get('outcome')}")
        self.assertTrue(state.get("final_response"),
            "final_response must be non-empty after quant failure escalation")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Normaliser unit tests
# ═══════════════════════════════════════════════════════════════════════════

class NormalizerUnitTests(unittest.TestCase):
    """_normalize_waiting_human_state standalone behaviour."""

    def test_already_completed_state_not_modified(self):
        """Completed/stopped/error/limited state passes through unchanged."""
        for status in ("completed", "stopped", "error", "limited"):
            state = {"task_status": status, "outcome": "something", "final_response": "ok"}
            result = _normalize_waiting_human_state(state)
            self.assertEqual(result["task_status"], status)

    def test_interrupt_without_pending_gets_default_pending(self):
        """Bare __interrupt__ key → pending_human_action is built."""
        state = {"__interrupt__": "some interrupt value", "request_id": "req_1"}
        result = _normalize_waiting_human_state(state)
        self.assertEqual(result["task_status"], "waiting_human")
        self.assertEqual(result["outcome"], "human_review")
        self.assertTrue(result.get("pending_human_action"))

    def test_pending_without_interrupt_also_normalised(self):
        """pending_human_action alone is enough to trigger normalisation."""
        state = {"pending_human_action": {"type": "clarify", "missing": ["province"]},
                 "request_id": "req_2"}
        result = _normalize_waiting_human_state(state)
        self.assertEqual(result["task_status"], "waiting_human")
        self.assertEqual(result["outcome"], "human_review")
        self.assertTrue(result.get("final_response"))

    def test_final_response_built_when_missing(self):
        """When final_response is empty, normaliser builds one."""
        state = {"__interrupt__": "interrupted", "request_id": "req_3",
                 "pending_human_action": {"type": "review", "prompt": "need human"}}
        result = _normalize_waiting_human_state(state)
        final = result.get("final_response", "")
        self.assertTrue(final)
        self.assertIn("req_3", final)
        self.assertIn("等待人工处理", final)

    def test_existing_final_response_preserved(self):
        """Normaliser does not overwrite existing final_response."""
        state = {"pending_human_action": {"type": "review"},
                 "final_response": "Custom wait message"}
        result = _normalize_waiting_human_state(state)
        self.assertEqual(result["final_response"], "Custom wait message")


if __name__ == "__main__":
    unittest.main()
