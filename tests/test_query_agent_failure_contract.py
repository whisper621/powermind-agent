from __future__ import annotations

import unittest

from agents.planner_agent import PlanStep, StepType
from agents.query_agent import QueryAgent


class BrokenRetrieval:
    def retrieve(self, **kwargs):
        raise RuntimeError("retrieval connection failed")


class FakeTool:
    name = "predict_electricity_price"
    description = "fake prediction tool"

    def invoke(self, args):
        return {"province": args["province"], "price": 380}


class FakeRAG:
    def run(self, **kwargs):
        return {
            "context": "RAG policy context",
            "sources": ["policy-001"],
        }


class QueryAgentFailureContractTests(unittest.TestCase):
    def test_query_agent_reports_failure_without_fixed_answer(self) -> None:
        agent = QueryAgent(retrieval=BrokenRetrieval())
        step = PlanStep(
            step_id="retrieve",
            step_type=StepType.RETRIEVE,
            description="检索政策",
            agent="compliance",
        )

        result = agent.execute(step, province="山东")

        self.assertFalse(result.success)
        self.assertEqual(result.output, "")
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.error)
        self.assertEqual(result.component, "query_agent.retrieve")

    def test_retrieval_alternative_is_reported_not_automatically_used(self) -> None:
        agent = QueryAgent(retrieval=BrokenRetrieval(), rag_pipeline=FakeRAG())
        step = PlanStep(
            step_id="retrieve",
            step_type=StepType.RETRIEVE,
            description="检索政策",
            agent="compliance",
        )

        primary_result = agent.execute(step, province="山东")

        self.assertFalse(primary_result.success)
        self.assertEqual(primary_result.output, "")
        self.assertEqual(primary_result.available_alternatives, ["rag_retrieval"])

        alternative_result = agent.execute_alternative(
            "rag_retrieval",
            step,
            province="山东",
        )
        self.assertTrue(alternative_result.success)
        self.assertIn("RAG policy context", alternative_result.output)

    def test_registered_direct_tool_alternative_can_be_executed(self) -> None:
        agent = QueryAgent(tools=[FakeTool()])
        step = PlanStep(
            step_id="analyse",
            step_type=StepType.ANALYSE,
            description="执行价格预测",
            agent="quant",
        )

        result = agent.execute_alternative(
            "direct_tool_call",
            step,
            province="山东",
        )

        self.assertTrue(result.success)
        self.assertTrue(result.fallback_used)
        self.assertIn("predict_electricity_price", result.output)


if __name__ == "__main__":
    unittest.main()
