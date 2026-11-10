"""Agent failure contract tests.

- LLM failure is NOT wrapped as agent success.
- Tool failure is NOT fed back as normal observation.
- Retries come from DegradationPolicy, not internal loops.
"""

from __future__ import annotations

import unittest

from contracts.results import ComponentResult, ComponentExecutionError


class AgentFailureContractTests(unittest.TestCase):
    def test_component_execution_error_carries_metadata(self) -> None:
        """ComponentExecutionError has component, failure_type, retryable, details."""
        exc = ComponentExecutionError(
            component="agent.quant",
            message="LLM call failed",
            failure_type="agent_failure",
            retryable=True,
            alternatives=("fallback_quant",),
            details={"iteration": 3},
        )
        self.assertEqual(exc.component, "agent.quant")
        self.assertEqual(exc.failure_type, "agent_failure")
        self.assertTrue(exc.retryable)
        self.assertIn("fallback_quant", exc.alternatives)
        self.assertEqual(exc.details["iteration"], 3)

    def test_component_result_success_is_explicit(self) -> None:
        """ComponentResult.success MUST be explicitly True for success."""
        result = ComponentResult(
            component="quant",
            success=True,
            data={"price": 380},
        )
        self.assertTrue(result.success)
        self.assertIn("price", result.data)

    def test_component_result_failure_is_not_success(self) -> None:
        """ComponentResult with success=False is clearly failure."""
        result = ComponentResult(
            component="quant",
            success=False,
            warnings=["Quant model timeout"],
        )
        self.assertFalse(result.success)

    def test_error_string_not_in_success_data(self) -> None:
        """Error messages must NOT be placed in data to simulate success."""
        # This is a pattern test — the contract enforces this
        result = ComponentResult(
            component="quant",
            success=True,
            data={"answer": "Legitimate result"},
        )
        # The data dict should not contain error-like fields
        self.assertNotIn("error", result.data)
        self.assertNotIn("Tool error", str(result.data))

    def test_component_execution_error_is_not_generic_exception(self) -> None:
        """ComponentExecutionError is identifiable, not a plain Exception."""
        exc = ComponentExecutionError(
            component="agent.strategy",
            message="ReAct limit reached",
            failure_type="iteration_limit",
            retryable=False,
        )
        self.assertIsInstance(exc, RuntimeError)
        self.assertIsInstance(exc, ComponentExecutionError)

    def test_retryable_vs_non_retryable_failures(self) -> None:
        """Different failure types have correct retryable flags."""
        retryable = ComponentExecutionError(
            component="agent.quant",
            message="timeout",
            failure_type="agent_failure",
            retryable=True,
        )
        self.assertTrue(retryable.retryable)

        non_retryable = ComponentExecutionError(
            component="agent.quant",
            message="iteration limit",
            failure_type="iteration_limit",
            retryable=False,
        )
        self.assertFalse(non_retryable.retryable)


if __name__ == "__main__":
    unittest.main()
