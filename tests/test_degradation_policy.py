from __future__ import annotations

import unittest

from middleware.degradation_policy import (
    DegradationAction,
    DegradationPolicy,
    FailureContext,
    FailureType,
)


class DegradationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DegradationPolicy()

    def test_transient_failure_retries_within_limit(self) -> None:
        decision = self.policy.decide(
            FailureContext(
                component="llm_gateway",
                failure_type=FailureType.TRANSIENT,
                error="timeout",
                critical=True,
                attempt=0,
                max_retries=1,
            )
        )
        self.assertEqual(decision.action, DegradationAction.RETRY)

    def test_noncritical_failure_reduces_scope(self) -> None:
        decision = self.policy.decide(
            FailureContext(
                component="agent.expert",
                failure_type=FailureType.AGENT_FAILURE,
                error="expert unavailable",
                critical=False,
            )
        )
        self.assertEqual(decision.action, DegradationAction.REDUCE_SCOPE)
        self.assertEqual(decision.allowed_use, "reference_or_review")

    def test_registered_alternative_is_selected_before_scope_reduction(self) -> None:
        decision = self.policy.decide(
            FailureContext(
                component="query_agent.analyse",
                failure_type=FailureType.TOOL_FAILURE,
                error="LLM tool routing unavailable",
                critical=False,
                alternatives=("direct_tool_call",),
            )
        )
        self.assertEqual(decision.action, DegradationAction.SWITCH_CAPABILITY)
        self.assertEqual(decision.selected_alternative, "direct_tool_call")

    def test_unknown_external_state_never_retries(self) -> None:
        decision = self.policy.decide(
            FailureContext(
                component="execution_api",
                failure_type=FailureType.EXTERNAL_STATE_UNKNOWN,
                error="request timed out after submission",
                critical=True,
                external_side_effect_possible=True,
                external_state_known=False,
            )
        )
        self.assertEqual(decision.action, DegradationAction.HUMAN_REVIEW)
        self.assertIn("不得重复提交", decision.limitations)

    def test_unregistered_critical_failure_goes_to_human(self) -> None:
        decision = self.policy.decide(
            FailureContext(
                component="agent.compliance",
                failure_type=FailureType.AGENT_FAILURE,
                error="unknown",
                critical=True,
            )
        )
        self.assertEqual(decision.action, DegradationAction.HUMAN_REVIEW)

    def test_audit_failure_blocks_formal_task(self) -> None:
        decision = self.policy.decide(
            FailureContext(
                component="audit",
                failure_type=FailureType.AUDIT_UNAVAILABLE,
                error="audit store unavailable",
                critical=True,
            )
        )
        self.assertEqual(decision.action, DegradationAction.BLOCK)


if __name__ == "__main__":
    unittest.main()
