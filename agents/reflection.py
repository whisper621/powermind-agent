#!/usr/bin/env python3
"""
PowerMind Reflection 自反思机制 — 借鉴 PowerTrade 的 Reflection 设计,增强版。

Reflection 节点在每个 Agent 输出策略/报告后,自动进行多维质量审查,
审查不通过 → 反馈修改意见 → Agent 重新生成 → 再次审查。
最多 1 轮反思 (避免无限循环)。

与 PowerTrade 的差异:
  - PowerTrade: 单一维度检查 (PASS/REVISE)
  - PowerMind:  5 维度检查 (与 VerifierAgent 互补,Verifier 核查事实,Reflection 查质量)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


@dataclass
class ReflectionVerdict:
    """反思审查结果"""
    passed: bool
    score: float  # 0.0 - 1.0
    issues: list[dict] = field(default_factory=list)
    revision_instruction: str = ""
    dimension_scores: dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Reflection Prompt — 5 维质检
# ═══════════════════════════════════════════════════════════════════

REFLECTION_SYSTEM_PROMPT = """你是电力交易策略的质检专家。你需要严格审查 AI Agent 生成的交易策略建议。

## 审查维度 (每项打分 0-10, 然后给出 PASS 或 REVISE)

### 1. 数据真实性 (Data Fidelity)
- 策略中引用的价格/因子/规则数据是否来自工具返回的实际数据？
- 是否有编造的数字或虚构的数据来源？
- 策略中声称的"来自XX工具"的数据是否与工具输出一致？

### 2. 合规性 (Compliance)
- 建议的策略是否在目标省份的交易规则范围内？
- 报价是否超出价格上限/下限？
- 偏差管理建议是否考虑了免考核范围？
- 分时策略是否与该省 TOU 时段定义一致？

### 3. 完整性 (Completeness)
- 是否覆盖了用户问题的所有方面？
- 如果用户问了多个问题,是否逐一回答？
- 是否遗漏了关键的风险提示？

### 4. 可操作性 (Actionability)
- 建议是具体的、可量化的步骤,还是空泛的套话？
- 是否有明确的数值指标 (价格、电量、时间)？
- 交易员能否直接据此执行？

### 5. 逻辑自洽 (Logical Consistency)
- 策略的各个部分是否相互一致？
- "看涨"的判断与"低价卖出"的建议是否矛盾？
- 风险判断与策略激进程度是否匹配？

## 审查输出格式

输出严格 JSON:

{
  "dimension_scores": {
    "data_fidelity": 8,
    "compliance": 9,
    "completeness": 7,
    "actionability": 6,
    "logical_consistency": 8
  },
  "overall_score": 7.6,
  "verdict": "PASS",
  "issues": [
    {"dimension": "completeness", "severity": "warning", "description": "未提及午间光伏大发对价格的影响"}
  ],
  "revision_instruction": ""
}

## 判定规则
- overall_score >= 7.0 且 无 severity="error" 的 issues → PASS
- 否则 → REVISE, 在 revision_instruction 中给出具体修改指令

## 铁律
- 你是质检员,不是策略师,不要给出新的策略建议
- 你只指出问题,不替代 STRATEGY Agent
- REVISE 时,问题描述必须具体到可以指导修改"""


# ═══════════════════════════════════════════════════════════════════
# Reflection 执行器
# ═══════════════════════════════════════════════════════════════════

def run_reflection(
    llm,
    strategy_output: str,
    user_query: str = "",
    province: str = "",
    quant_result: dict | None = None,
    compliance_result: dict | None = None,
) -> ReflectionVerdict:
    """
    对策略输出执行 5 维 Reflection 质检。

    Args:
        llm: ChatDeepSeek 实例
        strategy_output: StrategyAgent 的原始输出文本
        user_query: 原始用户问题
        province: 目标省份
        quant_result: QUANT Agent 的结构化输出
        compliance_result: COMPLIANCE Agent 的结构化输出

    Returns:
        ReflectionVerdict 对象
    """
    # 截断过长的策略文本
    output_for_review = strategy_output[:3000]

    # 构建上下文信息
    context_parts = [f"策略建议:\n{output_for_review}"]

    if user_query:
        context_parts.insert(0, f"用户原始需求:\n{user_query}")
    if province:
        context_parts.insert(0, f"目标省份: {province}")

    if quant_result and quant_result.get("status") == "success":
        context_parts.append(
            f"\n上游量化预测:\n{quant_result.get('summary', '')[:500]}"
        )
    if compliance_result and compliance_result.get("status") == "success":
        context_parts.append(
            f"\n上游合规风控:\n风控状态={compliance_result.get('risk_status', '?')}, "
            f"{compliance_result.get('summary', '')[:500]}"
        )

    full_context = "\n\n".join(context_parts)

    # 组装反思 prompt
    reflection_messages = [
        HumanMessage(content=f"{REFLECTION_SYSTEM_PROMPT}\n\n{full_context}")
    ]

    try:
        response = llm.invoke(reflection_messages)
        response_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"Reflection LLM 调用失败: {e}")
        return ReflectionVerdict(
            passed=True,  # LLM 不可用时放行
            score=0.7,
            issues=[{"dimension": "system", "severity": "warning",
                     "description": f"Reflection LLM 不可用, 跳过质检: {e}"}],
        )

    # 解析 JSON (处理LLM在JSON前输出中文的情况)
    try:
        import re
        cleaned = response_text.strip()
        # 提取 ```json ... ``` 代码块
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if m: cleaned = m.group(1)
        # 如果开头不是{，找到第一个{
        if not cleaned.startswith('{'):
            idx = cleaned.find('{')
            if idx > 0: cleaned = cleaned[idx:]
        # 去除残留标记
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        # 截到最后一个完整 }
        depth = 0; end = 0
        for i, ch in enumerate(cleaned):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0: end = i + 1; break
        if end > 0: cleaned = cleaned[:end]

        data = json.loads(cleaned)

        return ReflectionVerdict(
            passed=data.get("verdict", "PASS") == "PASS",
            score=float(data.get("overall_score", 7.0)),
            issues=data.get("issues", []),
            revision_instruction=data.get("revision_instruction", ""),
            dimension_scores=data.get("dimension_scores", {}),
        )

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Reflection JSON 解析失败: {e}, raw={response_text[:200]}")
        # 简单判断: 如果有 "REVISE" 关键词
        if "REVISE" in response_text.upper():
            return ReflectionVerdict(
                passed=False,
                score=5.0,
                issues=[{"dimension": "parse", "severity": "error",
                         "description": f"JSON 解析失败, 但检测到 REVISE 意图: {response_text[:300]}"}],
                revision_instruction=response_text[:500],
            )
        return ReflectionVerdict(passed=True, score=7.0)  # 默认放行


def apply_reflection_revision(
    llm,
    original_output: str,
    revision_instruction: str,
    user_query: str = "",
) -> str:
    """
    将 Reflection 的修改意见反馈给 LLM, 让 LLM 重新生成策略。

    Args:
        llm: ChatDeepSeek 实例
        original_output: 被 Reflection 判定为 REVISE 的原始策略
        revision_instruction: Reflection 给出的修改指令
        user_query: 原始用户问题

    Returns:
        修改后的策略文本
    """
    revision_prompt = f"""你之前生成的交易策略需要根据以下质检意见进行修改。

原始用户需求: {user_query}

你的原始策略:
{original_output[:2000]}

质检意见:
{revision_instruction}

请根据质检意见,重新生成一份修正后的完整策略。要求:
1. 修正所有指出的问题
2. 保持策略的完整结构
3. 未指出的部分保持原样
4. 所有数字和引用保持真实"""

    try:
        response = llm.invoke([HumanMessage(content=revision_prompt)])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"Reflection 修改失败: {e}, 返回原始输出")
        return original_output


# ═══════════════════════════════════════════════════════════════════
# 便捷函数 — 供 sub_agents.py 中的 strategy_node 使用
# ═══════════════════════════════════════════════════════════════════

def reflect_and_revise(
    llm,
    strategy_output: str,
    user_query: str,
    province: str,
    quant_result: dict | None = None,
    compliance_result: dict | None = None,
    max_reflection_passes: int = 1,
) -> tuple[str, ReflectionVerdict]:
    """
    对策略输出执行 Reflection。如果不通过则尝试修改。

    Returns:
        (final_output, verdict) — 最终策略文本和质检结果
    """
    verdict = run_reflection(
        llm=llm,
        strategy_output=strategy_output,
        user_query=user_query,
        province=province,
        quant_result=quant_result,
        compliance_result=compliance_result,
    )

    if verdict.passed:
        logger.info(f"Reflection PASSED: score={verdict.score:.1f}")
        return strategy_output, verdict

    # 不通过 — 尝试最多 max_reflection_passes 次修改
    logger.warning(f"Reflection FAILED: score={verdict.score:.1f}, issues={len(verdict.issues)}")

    for attempt in range(max_reflection_passes):
        revised = apply_reflection_revision(
            llm=llm,
            original_output=strategy_output,
            revision_instruction=verdict.revision_instruction,
            user_query=user_query,
        )

        # 再次反思
        verdict2 = run_reflection(
            llm=llm,
            strategy_output=revised,
            user_query=user_query,
            province=province,
            quant_result=quant_result,
            compliance_result=compliance_result,
        )

        if verdict2.passed:
            logger.info(f"Reflection PASSED after revision: score={verdict2.score:.1f}")
            return revised, verdict2

        logger.warning(f"Reflection still FAILED after revision {attempt+1}")

    # 所有修改尝试仍未通过 — 返回原始输出 + 标注
    annotated = (
        f"> ⚠️ 系统质检未完全通过 (得分: {verdict.score:.1f}/10)"
        f"\n> 以下内容已尽力修正，但仍可能存在以下问题:\n"
        + "\n".join(f"> - [{i['dimension']}] {i['description']}" for i in verdict.issues[:5])
        + f"\n\n---\n\n{strategy_output}"
    )
    return annotated, verdict
