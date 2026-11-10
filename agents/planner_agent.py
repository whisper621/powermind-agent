"""Planner Agent — Task decomposition and orchestration planning.

The Planner is the entry point for the multi-agent workflow. It:
  1. Analyses the user query and available context
  2. Decomposes complex queries into subtasks with dependencies
  3. Produces a Plan that the Orchestrator executes
  4. Handles re-planning when subtasks fail
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(str, Enum):
    RETRIEVE = "retrieve"  # fetch data from retrieval pipeline
    ANALYSE = "analyse"  # run quantitative analysis
    VERIFY = "verify"  # fact-check a previous result
    GENERATE = "generate"  # produce final output
    ASK_USER = "ask_user"  # request clarification


@dataclass
class PlanStep:
    """A single step in the execution plan."""

    step_id: str
    step_type: StepType
    description: str
    agent: str  # which specialist agent handles this
    depends_on: list[str] = field(default_factory=list)  # step_ids this step needs
    input_context: str = ""
    output: str = ""
    status: StepStatus = StepStatus.PENDING
    retry_count: int = 0
    max_retries: int = 2
    result: dict | None = None
    error: str = ""


@dataclass
class Plan:
    """Execution plan produced by the Planner Agent."""

    plan_id: str
    query: str
    intent: str
    steps: list[PlanStep] = field(default_factory=list)
    global_context: dict = field(default_factory=dict)
    created_at: str = ""

    @property
    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    @property
    def ready_steps(self) -> list[PlanStep]:
        """Steps whose dependencies are all COMPLETED."""
        completed_ids = {
            s.step_id for s in self.steps if s.status == StepStatus.COMPLETED
        }
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed_ids for dep in s.depends_on)
        ]

    @property
    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED)
            for s in self.steps
        )

    @property
    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)


# ---------------------------------------------------------------------------
# Planner prompt (Few-Shot)
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """你是一个电力行业任务规划器。你的职责是将用户的复杂查询分解为可执行的子任务。

## 可用的 Agent 类型
- quant: 数据分析和价格预测
- compliance: 政策搜索和合规检查
- expert: 专家经验和历史案例匹配
- strategy: 策略生成和情景模拟

## 可用的步骤类型
- retrieve: 检索相关信息
- analyse: 运行定量分析
- verify: 验证前序结果
- generate: 生成最终输出
- ask_user: 需要用户澄清

## 规划原则
1. 检索步骤必须先于分析步骤
2. 分析步骤必须先于生成步骤
3. 合规检查应在策略生成之前完成
4. 每个步骤最多重试2次
5. 如果查询不明确，首先添加 ask_user 步骤

## 输出格式
仅返回 JSON:
{
  "intent": "policy_search|data_analysis|bidding_advice|client_report|general_qa",
  "steps": [
    {
      "step_id": "step_1",
      "step_type": "retrieve",
      "description": "检索山东现货电价政策",
      "agent": "compliance",
      "depends_on": []
    }
  ]
}
"""


class PlannerAgent:
    """Planner agent that decomposes queries into executable plans."""

    def __init__(self, llm=None):
        self._llm = llm

    def plan(
        self,
        query: str,
        context: dict | None = None,
        memory_context: str = "",
    ) -> Plan:
        """Produce an execution plan for the query."""
        ctx = context or {}

        if self._llm is None:
            return self.fixed_workflow_plan(query, ctx)

        prompt = self._build_planning_prompt(query, ctx, memory_context)

        try:
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            data = self._parse_json(text)
            return self._build_plan(query, data)
        except Exception as exc:
            logger.warning("Planner LLM call failed: %s", exc)
            raise RuntimeError(f"Planner生成计划失败: {exc}") from exc

    def replan(
        self,
        original_plan: Plan,
        failed_step: PlanStep,
        error_message: str,
        decision_action: str | None = None,
    ) -> Plan:
        """Apply a decision already made by DegradationPolicy.

        PlannerAgent does not choose fault actions by itself.
        """
        if decision_action is None:
            raise RuntimeError(
                "replan需要DegradationPolicy给出的decision_action，"
                "Planner不能自行决定重试或跳过"
            )

        if decision_action == "retry" and failed_step.retry_count < failed_step.max_retries:
            failed_step.retry_count += 1
            failed_step.status = StepStatus.PENDING
            logger.info(
                "Planner: retrying step %s (attempt %d/%d)",
                failed_step.step_id,
                failed_step.retry_count,
                failed_step.max_retries,
            )
            return original_plan

        if decision_action == "reduce_scope":
            failed_step.status = StepStatus.SKIPPED
        else:
            failed_step.status = StepStatus.FAILED
        failed_step.error = error_message
        return original_plan

    # ----- internal ----------------------------------------------------------

    def _build_planning_prompt(self, query: str, ctx: dict, memory: str) -> str:
        parts = [PLANNER_SYSTEM_PROMPT]
        if ctx.get("province"):
            parts.append(f"\n当前省份: {ctx['province']}")
        if ctx.get("task_type"):
            parts.append(f"任务类型提示: {ctx['task_type']}")
        if memory:
            parts.append(f"\n历史记忆: {memory}")
        parts.append(f"\n用户查询: {query}")
        parts.append("\n请输出 JSON 计划:")
        return "\n".join(parts)

    def _build_plan(self, query: str, data: dict) -> Plan:
        import uuid
        from datetime import datetime

        steps = []
        for item in data.get("steps", []):
            steps.append(
                PlanStep(
                    step_id=item.get("step_id", f"step_{len(steps)+1}"),
                    step_type=StepType(item.get("step_type", "retrieve")),
                    description=item.get("description", ""),
                    agent=item.get("agent", "expert"),
                    depends_on=item.get("depends_on", []),
                    max_retries=2,
                )
            )

        return Plan(
            plan_id=str(uuid.uuid4())[:8],
            query=query,
            intent=data.get("intent", "general_qa"),
            steps=steps,
            created_at=datetime.now().isoformat(),
        )

    def fixed_workflow_plan(self, query: str, ctx: dict) -> Plan:
        """Load an approved fixed workflow for an exact supported task type.

        This is not a universal fallback.  Unknown or long-tail tasks must be
        planned and validated; they cannot be forced into a generic fixed chain.
        """
        import uuid
        from datetime import datetime

        task_type = ctx.get("task_type", "general")
        province = ctx.get("province", "")

        intent_map = {
            "daily_pulse": "data_analysis",
            "bidding_advice": "bidding_advice",
            "policy_search": "policy_search",
            "client_report": "client_report",
            "long_term_locking": "bidding_advice",
        }

        if task_type in ("bidding_advice", "long_term_locking"):
            # ★ Full strategy pipeline: retrieve → predict → factors → compliance → strategy
            steps = [
                PlanStep(
                    step_id="step_retrieve",
                    step_type=StepType.RETRIEVE,
                    description=f"检索{province}现货电价历史和政策规则",
                    agent="compliance",
                    depends_on=[],
                ),
                PlanStep(
                    step_id="step_predict",
                    step_type=StepType.ANALYSE,
                    description=f"调用 predict_electricity_price 预测{province}电价(24点逐时)",
                    agent="quant",
                    depends_on=["step_retrieve"],
                ),
                PlanStep(
                    step_id="step_factors",
                    step_type=StepType.ANALYSE,
                    description=f"调用 compute_market_factors 分析{province}9类市场因子",
                    agent="quant",
                    depends_on=["step_predict"],
                ),
                PlanStep(
                    step_id="step_compliance",
                    step_type=StepType.ANALYSE,
                    description=f"调用 search_policy + run_risk_validation 执行{province}合规风控检查",
                    agent="compliance",
                    depends_on=["step_retrieve"],
                ),
                PlanStep(
                    step_id="step_strategy",
                    step_type=StepType.ANALYSE,
                    description=f"调用 run_multi_role_analysis + run_scenario_simulation + run_strategy_advisor 生成{province}交易策略",
                    agent="strategy",
                    depends_on=["step_predict", "step_factors", "step_compliance"],
                ),
                PlanStep(
                    step_id="step_generate",
                    step_type=StepType.GENERATE,
                    description=f"综合所有分析结果，生成{province}最终交易策略报告",
                    agent="strategy",
                    depends_on=["step_strategy"],
                ),
            ]
        elif task_type == "daily_pulse":
            steps = [
                PlanStep(
                    step_id="step_retrieve",
                    step_type=StepType.RETRIEVE,
                    description=f"检索{province}今日市场数据和政策动态",
                    agent="compliance",
                    depends_on=[],
                ),
                PlanStep(
                    step_id="step_predict",
                    step_type=StepType.ANALYSE,
                    description=f"调用 predict_electricity_price 预测{province}今日电价",
                    agent="quant",
                    depends_on=["step_retrieve"],
                ),
                PlanStep(
                    step_id="step_factors",
                    step_type=StepType.ANALYSE,
                    description=f"调用 compute_market_factors 分析{province}市场因子",
                    agent="quant",
                    depends_on=["step_predict"],
                ),
                PlanStep(
                    step_id="step_generate",
                    step_type=StepType.GENERATE,
                    description=f"生成{province}电力市场日报",
                    agent="strategy",
                    depends_on=["step_factors"],
                ),
            ]
        elif task_type == "policy_search":
            steps = [
                PlanStep(
                    step_id="step_1",
                    step_type=StepType.RETRIEVE,
                    description=f"检索{province}相关信息和政策" if province else "检索相关信息",
                    agent="compliance",
                    depends_on=[],
                ),
                PlanStep(
                    step_id="step_2",
                    step_type=StepType.ANALYSE,
                    description="分析检索结果并提取关键信息",
                    agent="quant",
                    depends_on=["step_1"],
                ),
                PlanStep(
                    step_id="step_3",
                    step_type=StepType.VERIFY,
                    description="验证分析结果的准确性和合规性",
                    agent="compliance",
                    depends_on=["step_2"],
                ),
                PlanStep(
                    step_id="step_4",
                    step_type=StepType.GENERATE,
                    description="生成最终回答",
                    agent="strategy",
                    depends_on=["step_2", "step_3"],
                ),
            ]
        else:
            raise RuntimeError(
                f"任务类型 {task_type!r} 没有已发布的固定工作流，不能使用固定流程替代动态规划"
            )

        return Plan(
            plan_id=str(uuid.uuid4())[:8],
            query=query,
            intent=intent_map.get(task_type, "general_qa"),
            steps=steps,
            created_at=datetime.now().isoformat(),
        )

    # Backward-compatible private alias for callers outside the main runtime.
    def _hardcoded_plan(self, query: str, ctx: dict) -> Plan:
        return self.fixed_workflow_plan(query, ctx)

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {}
