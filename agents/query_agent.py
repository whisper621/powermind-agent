"""Query / Tool Agent — Executes retrieval and tool-calling tasks.

This agent bridges the Planner's PlanSteps to concrete retrieval and tool
execution. It wraps the HybridRetrievalPipeline and all registered tools
under a unified execute(step) -> result interface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agents.planner_agent import PlanStep, StepStatus, StepType

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a single PlanStep."""

    step_id: str
    success: bool
    output: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0
    fallback_used: bool = False
    component: str = ""
    error_type: str = ""
    degradation_decision: dict = field(default_factory=dict)
    available_alternatives: list[str] = field(default_factory=list)


class QueryAgent:
    """Executes retrieval and tool-calling PlanSteps.

    Maintains a registry of tool functions and delegates to the
    HybridRetrievalPipeline for retrieval tasks.

    Usage:
        agent = QueryAgent(retrieval=pipeline, tools=[...])
        result = agent.execute(plan_step, province="山东")
    """

    def __init__(
        self,
        retrieval=None,  # HybridRetrievalPipeline
        tools: list | None = None,
        rag_pipeline=None,
    ):
        self._retrieval = retrieval
        self._rag = rag_pipeline
        self._tools: dict[str, Callable] = {}
        if tools:
            for tool in tools:
                name = getattr(tool, "name", None) or getattr(tool, "__name__", str(id(tool)))
                self._tools[name] = tool

    def register_tool(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def execute(
        self,
        step: PlanStep,
        province: str = "",
        context: dict | None = None,
    ) -> StepResult:
        """Execute a single plan step and return the result."""
        t0 = time.perf_counter()
        ctx = context or {}

        try:
            if step.step_type == StepType.RETRIEVE:
                output, data = self._execute_retrieve(step, province, ctx)
            elif step.step_type == StepType.ANALYSE:
                output, data = self._execute_analyse(step, province, ctx)
            elif step.step_type == StepType.VERIFY:
                output, data = self._execute_verify(step, province, ctx)
            elif step.step_type == StepType.GENERATE:
                output, data = self._execute_generate(step, province, ctx)
            else:
                output, data = f"Unknown step type: {step.step_type}", {}

            duration = round((time.perf_counter() - t0) * 1000, 2)
            return StepResult(
                step_id=step.step_id,
                success=True,
                output=output,
                data=data,
                duration_ms=duration,
            )

        except Exception as exc:
            from middleware.degradation_policy import classify_failure

            duration = round((time.perf_counter() - t0) * 1000, 2)
            logger.warning("QueryAgent: step %s failed — %s", step.step_id, exc)
            component = f"query_agent.{step.step_type.value}"
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error=str(exc),
                duration_ms=duration,
                fallback_used=False,
                component=component,
                error_type=classify_failure(component, exc).value,
                available_alternatives=self._available_alternatives(step),
            )

    # ----- step executors ----------------------------------------------------

    def _available_alternatives(self, step: PlanStep) -> list[str]:
        """Return only alternatives that are registered for this step type."""

        alternatives: list[str] = []
        if step.step_type == StepType.RETRIEVE and self._rag is not None:
            alternatives.append("rag_retrieval")
        if step.step_type == StepType.ANALYSE and self._tools:
            alternatives.append("direct_tool_call")
        return alternatives

    def _execute_retrieve(
        self, step: PlanStep, province: str, ctx: dict
    ) -> tuple[str, dict]:
        """Execute the primary retrieval capability.

        This method never switches retrieval channels on its own.  If the
        primary capability fails, ``execute`` reports the registered
        alternatives and DegradationPolicy decides whether one may be used.
        """

        if self._retrieval is None:
            raise RuntimeError("主检索服务未注册")

        results = self._retrieval.retrieve(
            query=step.description,
            province=province,
            top_k=15,
        )
        if not results:
            raise RuntimeError("主检索服务未返回可用结果")

        context_blocks = [r.to_context_block() for r in results[:10]]
        return "\n\n".join(context_blocks), {
            "num_results": len(results),
            "sources": list({r.source for r in results}),
            "total_found": len(results),
            "retrieval_capability": "primary_hybrid_retrieval",
        }

    def _rag_retrieve(
        self, step: PlanStep, province: str
    ) -> tuple[str, dict]:
        """Run the separately registered RAG retrieval alternative."""

        if self._rag is None:
            raise RuntimeError("RAG替代检索能力未注册")
        rag_result = self._rag.run(query=step.description, province=province)
        context = rag_result.get("context", "")
        sources = rag_result.get("sources", [])
        if not context and not sources:
            raise RuntimeError("RAG替代检索未返回可用结果")
        return context, {
            "num_results": len(sources),
            "sources": sources,
            "retrieval_capability": "rag_retrieval",
        }

    def _execute_analyse(
        self, step: PlanStep, province: str, ctx: dict
    ) -> tuple[str, dict]:
        """Execute an ANALYSE step via registered tools using ReAct loop.

        When tools are available, uses a proper ReAct (Thought→Action→Observation)
        loop — the same pattern as sub_agents._run_agent_loop() — instead of
        keyword-based dispatch. This ensures tools are called with correct
        parameters and the LLM can reason about which tools to use.
        """
        if not self._tools:
            raise RuntimeError(f"没有注册可处理该步骤的分析工具: {step.description}")

        # Build system prompt for tool-calling
        tool_list = "\n".join(
            f"- {name}: {getattr(tool, 'description', 'No description')[:200]}"
            for name, tool in self._tools.items()
        )

        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from langchain_deepseek import ChatDeepSeek
        import os, time

        # Get API key from same sources as sub_agents
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            from pathlib import Path
            PROJECT_ROOT = Path(__file__).resolve().parents[1]
            env_path = PROJECT_ROOT / "config" / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("DEEPSEEK_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")

        if not api_key:
            raise RuntimeError("LLM工具路由不可用")

        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1, api_key=api_key)
        llm_with_tools = llm.bind_tools(list(self._tools.values()))

        # Build messages for ReAct loop
        context_info = ctx.get("retrieved_context", "")
        date_info = ctx.get("date", ctx.get("target_date", ""))

        system_prompt = f"""你是电力市场量化分析师。你必须使用工具获取真实数据。

可用工具:
{tool_list}

当前省份: {province}
{('目标日期: ' + date_info) if date_info else ''}
{('检索上下文: ' + context_info[:1500]) if context_info else ''}

任务: {step.description}

★ 重要: 必须调用相关工具获取实际数据！不要编造任何数据。
调用工具时使用 province='{province}'{(", target_date='" + date_info + "'") if date_info else ''}。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请执行: {step.description}"),
        ]

        # ReAct loop (max 6 iterations)
        for iteration in range(6):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.warning("ReAct LLM call failed: %s", e)
                raise RuntimeError(f"LLM工具路由失败: {e}") from e

            messages.append(response)

            # Check for tool calls
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                # If first iteration and no tool calls, push harder
                if iteration == 0:
                    messages.append(HumanMessage(
                        content="你必须调用工具获取真实数据！请立即调用相关工具。"
                    ))
                    continue
                # LLM is done — return its final text
                return response.content or "", {"tool": "react_loop", "iterations": iteration + 1}

            # Execute all tool calls
            tool_outputs = []
            for tc in response.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", "")

                # Ensure province and date are set
                tool_args.setdefault("province", province)
                if date_info and "target_date" in tool_args:
                    pass  # keep LLM's date
                elif date_info:
                    tool_args.setdefault("target_date", date_info)

                tool_fn = self._tools.get(tool_name)
                if tool_fn:
                    try:
                        result = tool_fn.invoke(tool_args)
                        tool_outputs.append(f"[{tool_name}]: {str(result)[:2000]}")
                    except Exception as exc:
                        tool_outputs.append(f"[{tool_name}] ERROR: {exc}")
                else:
                    tool_outputs.append(f"[{tool_name}] 工具未注册")

                messages.append(ToolMessage(
                    content=str(tool_outputs[-1])[:3000],
                    tool_call_id=tool_id,
                ))

            # If we got tool outputs, we can stop — they provide real data
            if tool_outputs and iteration >= 0:
                # One more LLM call to synthesize
                try:
                    final = llm.invoke(messages)
                    return final.content or "\n\n".join(tool_outputs), {
                        "tool": "react_loop",
                        "iterations": iteration + 1,
                        "tools_called": [tc.get("name") for tc in response.tool_calls],
                    }
                except Exception:
                    return "\n\n".join(tool_outputs), {
                        "tool": "react_loop",
                        "tools_called": [tc.get("name") for tc in response.tool_calls],
                    }

        raise RuntimeError("ReAct循环达到最大迭代次数，未形成有效工具结果")

    def _direct_tool_call(
        self, step: PlanStep, province: str, ctx: dict
    ) -> tuple[str, dict]:
        """Invoke registered tools without LLM after policy approval."""
        description_lower = step.description.lower()
        date_info = ctx.get("date", ctx.get("target_date", ""))
        outputs = []

        # Try specific tools based on description keywords
        if "价格预测" in step.description or "predict" in description_lower:
            tool = self._tools.get("predict_electricity_price")
            if tool:
                try:
                    args = {"province": province}
                    if date_info:
                        args["target_date"] = date_info
                    result = tool.invoke(args)
                    outputs.append(f"[predict_electricity_price]: {str(result)[:2000]}")
                except Exception as exc:
                    outputs.append(f"[predict_electricity_price] 失败: {exc}")

        if "因素" in step.description or "factor" in description_lower:
            tool = self._tools.get("compute_market_factors")
            if tool:
                try:
                    result = tool.invoke({"province": province, "target_date": date_info})
                    outputs.append(f"[compute_market_factors]: {str(result)[:2000]}")
                except Exception as exc:
                    outputs.append(f"[compute_market_factors] 失败: {exc}")

        if "风险" in step.description or "risk" in description_lower:
            for tool_name in ["run_risk_validation", "search_policy"]:
                tool = self._tools.get(tool_name)
                if tool:
                    try:
                        result = tool.invoke({"province": province, "target_date": date_info})
                        outputs.append(f"[{tool_name}]: {str(result)[:1500]}")
                    except Exception:
                        pass

        if "策略" in step.description or "strategy" in description_lower:
            for tool_name in ["run_strategy_advisor", "run_multi_role_analysis", "run_scenario_simulation"]:
                tool = self._tools.get(tool_name)
                if tool:
                    try:
                        result = tool.invoke({"province": province, "target_date": date_info})
                        outputs.append(f"[{tool_name}]: {str(result)[:1500]}")
                    except Exception:
                        pass

        if not outputs:
            # Generic: run all tools with matching names
            for name, tool in self._tools.items():
                if any(kw in name for kw in ["analyse", "compute", "run_", "predict", "search"]):
                    try:
                        result = tool.invoke({"province": province, "target_date": date_info})
                        outputs.append(f"[{name}]: {str(result)[:1500]}")
                    except Exception:
                        pass

        if outputs:
            return "\n\n".join(outputs), {"tools_used": len(outputs), "method": "direct"}

        raise RuntimeError(f"没有可成功执行的分析工具: {step.description}")

    def execute_alternative(
        self,
        alternative: str,
        step: PlanStep,
        province: str = "",
        context: dict | None = None,
    ) -> StepResult:
        """Execute an alternative explicitly selected by DegradationPolicy."""

        t0 = time.perf_counter()
        try:
            if alternative == "rag_retrieval":
                output, data = self._rag_retrieve(step, province)
            elif alternative == "direct_tool_call":
                output, data = self._direct_tool_call(
                    step,
                    province,
                    context or {},
                )
            else:
                raise RuntimeError(f"未注册的替代执行方式: {alternative}")
            return StepResult(
                step_id=step.step_id,
                success=True,
                output=output,
                data=data,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                fallback_used=True,
                component=f"query_agent.{step.step_type.value}",
            )
        except Exception as exc:
            from middleware.degradation_policy import classify_failure

            component = f"query_agent.{step.step_type.value}:{alternative}"
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=str(exc),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                component=component,
                error_type=classify_failure(component, exc).value,
            )

    def _execute_verify(
        self, step: PlanStep, province: str, ctx: dict
    ) -> tuple[str, dict]:
        """Execute a VERIFY step. Delegates to VerifierAgent."""
        # The actual verification happens in the Orchestrator via VerifierAgent
        # Here we just pass through the context from prior steps
        prior_output = ctx.get("generated_answer", "")
        sources = ctx.get("retrieved_context", "")

        return (
            f"验证准备就绪，待验证内容长度: {len(prior_output)} 字符",
            {"ready": True, "answer_length": len(prior_output)},
        )

    def _execute_generate(
        self, step: PlanStep, province: str, ctx: dict
    ) -> tuple[str, dict]:
        """Execute a GENERATE step. Delegates to EvidencePipeline."""
        # Generation happens in Orchestrator; here we signal readiness
        return (
            f"生成准备就绪，上下文长度: {len(ctx.get('retrieved_context', ''))} 字符",
            {"ready": True},
        )
