"""Compliance Gate tests — all 6 statuses route correctly.

PROHIBITED and BLOCKED must NEVER enter Strategy.
The gate reads structured state, never LLM text.
"""

from __future__ import annotations

import unittest

from contracts.compliance import (
    ComplianceStatus,
    ComplianceResult,
    compliance_gate_route,
)


class ComplianceGateTests(unittest.TestCase):
    def test_allowed_routes_to_step_dispatcher(self) -> None:
        """ALLOWED routes to step_dispatcher — plan decides next step."""
        result = ComplianceResult(status=ComplianceStatus.ALLOWED)
        self.assertTrue(not result.is_blocking)
        self.assertEqual(compliance_gate_route(result), "step_dispatcher")

    def test_needs_modification_routes_to_step_dispatcher(self) -> None:
        """NEEDS_MODIFICATION routes to step_dispatcher (forward with modifications noted)."""
        result = ComplianceResult(status=ComplianceStatus.NEEDS_MODIFICATION)
        self.assertFalse(result.is_blocking)
        self.assertEqual(compliance_gate_route(result), "step_dispatcher")

    def test_needs_review_routes_to_human_wait(self) -> None:
        result = ComplianceResult(status=ComplianceStatus.NEEDS_REVIEW)
        self.assertFalse(result.is_blocking)
        self.assertEqual(compliance_gate_route(result), "human_wait")

    def test_prohibited_routes_to_safe_stop(self) -> None:
        result = ComplianceResult(status=ComplianceStatus.PROHIBITED)
        self.assertTrue(result.is_blocking)
        self.assertEqual(compliance_gate_route(result), "safe_stop")

    def test_blocked_routes_to_safe_stop(self) -> None:
        result = ComplianceResult(status=ComplianceStatus.BLOCKED)
        self.assertTrue(result.is_blocking)
        self.assertEqual(compliance_gate_route(result), "safe_stop")

    def test_insufficient_evidence_routes_to_context_assembly(self) -> None:
        result = ComplianceResult(status=ComplianceStatus.INSUFFICIENT_EVIDENCE)
        self.assertFalse(result.is_blocking)
        self.assertEqual(compliance_gate_route(result), "context_assembly")

    def test_prohibited_never_enters_strategy(self) -> None:
        """PROHIBITED must never route to strategy."""
        result = ComplianceResult(status=ComplianceStatus.PROHIBITED)
        route = compliance_gate_route(result)
        self.assertNotEqual(route, "strategy")

    def test_blocked_never_enters_strategy(self) -> None:
        """BLOCKED must never route to strategy."""
        result = ComplianceResult(status=ComplianceStatus.BLOCKED)
        route = compliance_gate_route(result)
        self.assertNotEqual(route, "strategy")

    def test_gate_does_not_parse_natural_language(self) -> None:
        """Gate reads structured status only — PASS/FAIL strings in summary don't matter."""
        result = ComplianceResult(
            status=ComplianceStatus.PROHIBITED,
            summary="All checks PASS. Strategy is recommended. FAIL not found.",
        )
        self.assertEqual(compliance_gate_route(result), "safe_stop")

    def test_gate_reads_status_not_summary_for_allow(self) -> None:
        """Even if summary mentions FAIL, status=ALLOWED must route to step_dispatcher."""
        result = ComplianceResult(
            status=ComplianceStatus.ALLOWED,
            summary="Some non-critical checks FAIL but overall acceptable.",
        )
        self.assertEqual(compliance_gate_route(result), "step_dispatcher")


if __name__ == "__main__":
    unittest.main()
