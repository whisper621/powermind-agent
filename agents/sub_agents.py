#!/usr/bin/env python3
"""
PowerMind MAS — 专职子智能体 (Sub-Agents)。

四个专职 Specialist:
  - QuantAgent:     量化预测 (predict_price + market_factors)
  - ExpertInsightAgent: ★ 经验匹配 (deep_search_expert_logs + compare_market_conditions)
  - ComplianceAgent: 合规风控 (search_policy + risk_validation)
  - StrategyAgent:   策略交易 (multi_role + scenario + strategy + client_report)

每个 Agent 是一个 LangGraph 节点函数, 内部执行独立的 ReAct 循环。
输出严格遵守 Pydantic Schema, 存入 MASState 的结构化字段。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from middleware.audit_logger import get_audit_logger

# ═══════════════════════════════════════════════════════════════════
# System Prompts (精简, 每个 ~120-150 tokens)
# ═══════════════════════════════════════════════════════════════════

QUANT_SYSTEM_PROMPT = """你是电力市场量化预测专家。唯一职责: 基于数据和模型给出准确的电价预测和因子分析。

工具:
- predict_electricity_price: LSTM多变量模型, 输出24点逐时电价(元/MWh), 含日均价/峰谷均价/置信度
- compute_market_factors: 9类市场因子(政策/供需/天气/新能源/燃料/价格/风险/情绪/外来电)

输出要求:
1. 必须调用上述两个工具获取真实数据
2. 输出只包含: 预测均价、峰谷价差、24点电价范围、因子综合方向(bullish/bearish/neutral)、综合风险(low/medium/high/critical)
3. 不要提供任何交易建议或政策解读
4. 最后用一句话总结量化结论"""

COMPLIANCE_SYSTEM_PROMPT = """你是电力市场合规与风控专家。唯一职责: 检查政策规则和风险状态。

工具:
- search_policy: 检索ChromaDB+SQLite中的电力政策文件、交易规则、偏差考核标准
- run_risk_validation: 执行风控校验(证据检查+反向因素检测+置信度评估), 输出PASS/WARN/FAIL

输出要求:
1. 必须调用上述两个工具获取真实数据
2. 输出只包含: 适用规则条款、偏差考核标准(免考核范围)、风控状态、问题清单、建议处置
3. 不要提供任何价格预测或交易建议
4. 最后用一句话总结合规结论"""

def _load_strategy_prompt() -> str:
    """从文件加载 StrategyAgent 的完整 System Prompt。

    加载顺序:
      1. prompts/agents/strategy.txt — 角色定义 + 工具说明 + 铁律
      2. prompts/templates/bidding_sheet.md — ★ 权威输出模板 (必须严格遵循)

    如果文件不存在，使用默认版本。
    """
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "agents" / "strategy.txt"
    template_path = Path(__file__).resolve().parents[1] / "prompts" / "templates" / "bidding_sheet.md"

    if prompt_path.exists():
        base_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        base_prompt = """你是电力市场交易策略专家。基于上游预测数据、历史经验和合规结论, 给出可执行的交易建议。

工具:
- run_multi_role_analysis: 6角色多维分析(政策/供需/新能源/成本/价格/风险分析师)
- run_scenario_simulation: 4场景推演(基准50%/偏紧25%/宽松20%/极端5%)
- run_strategy_advisor: 6类策略(中长期/现货敞口/偏差管理/报价/分时套餐/客户沟通)
- generate_client_strategy_report: B2B客户用电优化报告

输出要求:
1. 必须基于上游数据做决策, 不能凭空建议
2. 每条策略标注支撑依据来源
3. 按 Markdown 模板输出结构化策略报告"""

    # ★ 注入权威输出模板
    if template_path.exists():
        template_content = template_path.read_text(encoding="utf-8")
        template_injection = f"""

# ═══════════════════════════════════════════════════════
# ★ 强制输出模板 — 你必须严格按照以下结构输出最终策略
# 这是来自 prompts/templates/bidding_sheet.md 的权威模板
# 所有 {{变量}} 必须用真实数据替换，不得保留占位符
# ═══════════════════════════════════════════════════════

{template_content}

# ═══════════════════════════════════════════════════════
# 模板使用铁律:
# 1. 所有 {{变量}} 必须替换为工具返回的真实数据
# 2. 表格中的每一行都必须填写，不得留空或写"待填入"
# 3. 如果某个数据工具未返回，标注"数据暂缺"而非留空
# 4. 七段结构必须完整输出，不得省略任何章节
# ═══════════════════════════════════════════════════════
"""
        return base_prompt + template_injection

    return base_prompt


SUPERVISOR_SYSTEM_PROMPT = """你是PowerMind主管协调器(MAS Supervisor)。职责:
1. 分析用户意图, 从自然语言中提取关键实体(省份/日期/任务类型)
2. 将任务分发给对应专家, 收集他们的结果
3. 根据结果决定是否需要更多专家参与
4. 综合所有专家的结论, 生成最终完整回答
5. 遇到歧义时, 主动向用户确认

## 专家团队
- QUANT: 量化预测专家 — 电价预测(24点逐时) + 9因子市场分析. 必须先调用
- EXPERT: 经验匹配专家 — 分析当前市场特征 + 检索历史相似案例 + LLM精排. 在QUANT之后调用
- COMPLIANCE: 合规风控专家 — 政策规则检索 + 风控校验(PASS/WARN/FAIL)
- STRATEGY: 策略交易专家 — 6角色分析 + 4场景推演 + 交易策略 + 客户报告. 接收上游QUANT+EXPERT+COMPLIANCE

## 意图识别与路由规则 (Few-Shot)

### 价格预测类 → QUANT
"明天电价多少" → QUANT
"预测一下后天现货价格" → QUANT
"下周电价走势" → QUANT
"下午的电价会更高吗" → QUANT

### 市场分析类 → QUANT
"今天什么情况" → QUANT
"最近市场供需怎么样" → QUANT
"新能源出力情况如何" → QUANT

### 经验查询类 → EXPERT
"上次类似行情怎么处理的" → QUANT → EXPERT (先QUANT获取特征, 再EXPERT匹配经验)
"历史上负电价时怎么操作" → QUANT → EXPERT
"老交易员在这种行情下会怎么做" → QUANT → EXPERT

### 政策合规类 → COMPLIANCE
"偏差考核标准是什么" → COMPLIANCE
"深谷限价规则" → COMPLIANCE
"最近有没有新政策" → COMPLIANCE

### 风险检查类 → COMPLIANCE
"风险大吗" → COMPLIANCE
"会不会触发偏差考核" → COMPLIANCE

### 交易策略类 → Q→E→S
"怎么报价" → Q→E→S
"明天申报什么价格" → Q→E→C→S
"中长期锁多少比例" → Q→E→S

### 完整分析类 → Q→E→C→S
"给我一份完整日报" → Q→E→C→S
"全面分析一下市场" → Q→E→C→S
"明天的交易策略和风险提示" → Q→E→C→S

### 客户报告类 → Q→E→S
"生成用电报告" → Q→E→S
"分析一下客户的电费成本" → Q→E→S

### 省间对比类 → Q→E→C
"XX省和XX省电价差多少" → Q→E→C

### 歧义消解 — 必须反问用户
"山东怎么样" → 反问: "您想了解山东的: A.今日电价走势 B.市场规则政策 C.入市可行性 D.完整市场分析？"
"帮我看看" → 反问: "请问您想看哪个省份的什么信息？比如电价预测、政策规则还是交易策略？"
"分析一下" → 反问: "请明确: 1.哪个省份 2.分析什么(电价/政策/策略/报告)"

## 输出格式
你的输出必须是严格JSON:
{"next_agent": "QUANT"|"COMPLIANCE"|"STRATEGY"|"FINISH"|"ASK_USER", "instruction": "给专家的指令或给用户的澄清问题", "reason": "路由原因", "extracted_entities": {"province": "山东", "date": "2026-05-22", "task_type": "price_query"}}

- next_agent="ASK_USER": 当意图模糊无法确定时, instruction字段填写向用户的反问
- next_agent="FINISH": 所有专家完成或不需要调用专家时, instruction字段填写给用户的综合回答"""


# ═══════════════════════════════════════════════════════════════════
# 共用工具函数
# ═══════════════════════════════════════════════════════════════════

def _get_api_key() -> str:
    """从多个来源获取 DeepSeek API Key。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    env_path = PROJECT_ROOT / "config" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # Fallback: check for API key from environment variable alias
    key = os.environ.get("POWER_MARKET_DEEPSEEK_API_KEY")
    if key:
        return key
    raise RuntimeError("DEEPSEEK_API_KEY 未配置")


def _run_agent_loop(
    llm,
    tools: list,
    messages: list,
    max_iterations: int = 6,
    agent_name: str = "Agent",
) -> str:
    """
    手动 ReAct 循环: LLM思考 → 工具调用 → 结果反馈 → 重复。

    故障处理契约:
      - 本函数不执行内部重试。LLM单次调用失败时抛出ComponentExecutionError。
      - Tool执行失败时记录Tool失败事件，不把"Tool error:"喂回Agent。
      - 达到ReAct上限时抛出no_progress异常。
      - 成功时返回最终LLM文本。
    """
    from contracts.results import ComponentExecutionError

    audit = get_audit_logger()
    llm_with_tools = llm.bind_tools(tools)

    # 预加载工具注册表
    from tools.generate_client_report import get_all_tools
    all_tools = {t.name: t for t in get_all_tools()}

    last_progress_iteration = -1

    for iteration in range(max_iterations):
        audit.log_agent_step(
            f"{agent_name}_think",
            f"iteration={iteration+1}/{max_iterations}",
        )

        # LLM 调用 — 单次, 不重试 (重试由 DegradationPolicy 统一控制)
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            raise ComponentExecutionError(
                component=f"agent.{agent_name}",
                message=f"LLM调用失败: {str(e)[:300]}",
                failure_type="agent_failure",
                retryable=True,
            ) from e

        if response is None:
            raise ComponentExecutionError(
                component=f"agent.{agent_name}",
                message="LLM未返回响应",
                failure_type="agent_failure",
                retryable=True,
            )

        messages.append(response)

        # 检查是否有 tool_calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            if iteration == 0:
                messages.append(HumanMessage(
                    content="你必须调用工具获取真实数据！不要直接回复\"系统未初始化\"或\"工具不可用\"。请立即调用你被赋予的工具。"
                ))
                continue
            return response.content or ""

        # 执行工具调用
        tool_call_count = len(response.tool_calls)
        tool_successes = 0
        tool_failures: list[dict] = []

        for tc in response.tool_calls:
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")

            trace_id = audit.log_tool_start(tool_name, tool_args)
            t0 = time.time()

            try:
                tool_fn = all_tools.get(tool_name)
                if tool_fn:
                    result = tool_fn.invoke(tool_args)
                else:
                    # Tool not found → structured failure, NOT success
                    result = json.dumps({
                        "status": "tool_not_found",
                        "tool": tool_name,
                        "available": list(all_tools.keys())[:10],
                    }, ensure_ascii=False)
                    success = False
                    error = f"Tool '{tool_name}' not found in registry"
                    tool_failures.append({
                        "tool": tool_name,
                        "error": error,
                        "iteration": iteration + 1,
                        "failure_type": "tool_not_found",
                    })
                    duration = (time.time() - t0) * 1000
                    audit.log_tool_end(trace_id, tool_name, duration, False, result[:200], error)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                    continue
                success = True
                error = None
            except Exception as e:
                # Tool exception → structured failure, NOT fed back as normal observation
                error_msg = str(e)[:200]
                result = json.dumps({
                    "status": "tool_error",
                    "tool": tool_name,
                    "error": error_msg,
                }, ensure_ascii=False)
                success = False
                error = error_msg
                tool_failures.append({
                    "tool": tool_name,
                    "error": error_msg,
                    "iteration": iteration + 1,
                    "failure_type": "tool_exception",
                })

            if success:
                tool_successes += 1

            duration = (time.time() - t0) * 1000
            audit.log_tool_end(trace_id, tool_name, duration, success, str(result)[:200], error)
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        # Track progress: if at least one tool succeeded, we made progress
        if tool_successes > 0:
            last_progress_iteration = iteration

        # All tools failed this iteration → record but continue if progress was made before
        if tool_failures and tool_successes == 0 and last_progress_iteration < 0:
            # No progress ever made → report failure
            raise ComponentExecutionError(
                component=f"agent.{agent_name}",
                message=f"所有Tool调用失败: {tool_failures}",
                failure_type="tool_failure",
                retryable=False,
                details={"tool_failures": tool_failures},
            )

    # ReAct 循环上限
    if last_progress_iteration < 0:
        raise ComponentExecutionError(
            component=f"agent.{agent_name}",
            message=f"ReAct循环达到上限({max_iterations})且无实质进展",
            failure_type="agent_failure",
            retryable=False,
            details={"max_iterations": max_iterations, "iteration_limit": True},
        )
    raise ComponentExecutionError(
        component=f"agent.{agent_name}",
        message=f"ReAct循环达到上限({max_iterations})",
        failure_type="iteration_limit",
        retryable=False,
        details={"max_iterations": max_iterations, "last_progress": last_progress_iteration},
    )


# ═══════════════════════════════════════════════════════════════════
# Sub-Agent 节点函数
# ═══════════════════════════════════════════════════════════════════

def quant_node(state: dict) -> dict:
    """
    Quant Agent 节点 — 量化预测专家。
    挂载 Tools: predict_electricity_price, compute_market_factors
    输出: state["quant_result"] = {forecast_summary, factors_summary, status}
    """
    from tools.predict_price import predict_electricity_price
    from tools.factor_engine import compute_market_factors

    api_key = _get_api_key()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1, api_key=api_key)

    province = state.get("province", "山东")
    target_date = state.get("target_date", datetime.now().strftime("%Y-%m-%d"))
    user_query = state.get("user_query", "")

    # ★ 预注入天气+历史数据上下文
    context_extra = ""
    try:
        import sqlite3, json
        db = PROJECT_ROOT / "data" / "sql" / "poweragent.db"
        if db.exists():
            conn = sqlite3.connect(str(db))
            # 最近7天历史均价
            rows = conn.execute(
                "SELECT date, AVG(price_rmb_mwh) as avg FROM price_data "
                "WHERE province=? AND date < ? GROUP BY date ORDER BY date DESC LIMIT 7",
                (province, target_date)
            ).fetchall()
            if rows:
                history = ", ".join(f"{r[0]}:{r[1]:.0f}" for r in reversed(rows))
                context_extra += f"\n【历史电价参考(近7天)】{history}\n"
            # 天气数据
            w = conn.execute(
                "SELECT date, AVG(temperature_2m) as t, AVG(cloud_cover) as c "
                "FROM weather_data WHERE province=? AND date>=? GROUP BY date ORDER BY date LIMIT 3",
                (province, target_date)
            ).fetchall()
            if w:
                weather = ", ".join(f"{r[0]}:{r[1]:.0f}°C/云量{r[2]:.0f}%" for r in w)
                context_extra += f"【天气参考】{weather}\n"
            conn.close()
    except Exception:
        pass

    tools = [predict_electricity_price, compute_market_factors]
    messages = [
        SystemMessage(content=QUANT_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"请为 {province} 省 {target_date} 执行量化分析。\n"
            f"用户原始问题: {user_query}\n"
            f"{context_extra}\n"
            f"★ 重要: 调用工具时请使用以下参数:\n"
            f"  - predict_electricity_price(target_date='{target_date}', province='{province}')\n"
            f"  - compute_market_factors(province='{province}', target_date='{target_date}')\n"
            f"不要使用其他日期，不要编造日期。请立即依次调用这两个工具。"
            f"输出时请引用上方历史电价和天气数据作为参考对比。"
        )),
    ]

    from contracts.results import ComponentExecutionError

    try:
        result_text = _run_agent_loop(llm, tools, messages, agent_name="QUANT")

        # 结构化成功结果
        quant_result = {
            "status": "success",
            "province": province,
            "target_date": target_date,
            "summary": result_text[:1500],
            "agent": "QUANT",
            "timestamp": datetime.now().isoformat(),
        }

        # 尝试从 result_text 提取预测价格
        import re
        price_match = re.search(r"(\d+\.?\d*)\s*元/MWh", result_text)
        if price_match:
            quant_result["avg_price_mentioned"] = float(price_match.group(1))

    except ComponentExecutionError as exc:
        # 结构化异常 → 上报给 LangGraph fault_policy
        quant_result = {
            "status": "failed",
            "error": str(exc),
            "failure_type": exc.failure_type,
            "retryable": exc.retryable,
            "agent": "QUANT",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        quant_result = {
            "status": "failed",
            "error": f"未预期异常: {type(e).__name__}: {str(e)[:300]}",
            "failure_type": "agent_failure",
            "retryable": False,
            "agent": "QUANT",
            "timestamp": datetime.now().isoformat(),
        }

    # 更新 called_agents (去重)
    called = list(state.get("called_agents", []))
    if "QUANT" not in called:
        called.append("QUANT")

    return {
        "quant_result": quant_result,
        "called_agents": called,
    }


# ═══════════════════════════════════════════════════════════════════
# EXPERT INSIGHT Agent — 经验匹配专家 (★ 新增轻量节点)
# ═══════════════════════════════════════════════════════════════════

EXPERT_SYSTEM_PROMPT = """你是电力交易经验匹配专家。唯一职责: 基于当前市场特征，从历史专家复盘日志中找出最匹配的经验案例。

## 核心能力
1. 分析当前市场特征组合 (9因子 + LSTM 预测 + 天气)
2. 从 ChromaDB 的 expert_insight 通道检索历史相似案例
3. 对候选案例做 LLM 二次精排，给出每条经验的相关性评分
4. 输出结构化经验报告，不提供最终策略 (策略由 STRATEGY Agent 负责)

## 输出要求
1. 分析当前市场最关键的 3 个特征
2. 找出历史上最相似的 3-5 条经验
3. 每条标注: 日期 / 相关性评分 / 当时策略 / 实际结果
4. 最后给出"历史决策共识"——大多数类似行情下老交易员会怎么做
5. 不提供价格预测 (那是 QUANT 的职责)
6. 不提供最终策略 (那是 STRATEGY 的职责)"""


def expert_insight_node(state: dict) -> dict:
    """
    ★ EXPERT Agent 节点 — 经验匹配专家 (轻量 LLM 增强检索)。

    不是完整 ReAct Agent，而是:
      ① LLM 分析当前市场特征 (1次调用)
      ② ChromaDB 向量检索 (5路中的 expert_insight 通道)
      ③ LLM 精排匹配 + 推理 (1次调用)
      → 输出结构化经验报告

    State 更新:
      state["expert_result"] = {
          "status": "success|failed",
          "features": "关键市场特征描述",
          "matched_cases": [{date, relevance, lesson, strategy, result}, ...],
          "consensus": "历史决策共识",
          "agent": "EXPERT",
      }
    """
    import json, traceback
    from datetime import datetime

    api_key = _get_api_key()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1, api_key=api_key)

    province = state.get("province", "山东")
    target_date = state.get("target_date", datetime.now().strftime("%Y-%m-%d"))
    user_query = state.get("user_query", "")
    quant = state.get("quant_result") or {}

    # ── Step 1: LLM 提取关键市场特征 ──
    quant_summary = quant.get("summary", "量化数据未获取") if isinstance(quant, dict) else str(quant)[:500]

    feature_prompt = f"""当前 {province} 省 {target_date} 的市场条件:
{quant_summary}

请提取最关键的市场特征，用于匹配历史经验。输出格式:
关键特征: <用50字以内描述，如"午间光伏出力偏高+晚峰空调负荷+X因素">
特征标签: <3-5个关键词，逗号分隔，如 光伏,晚峰,高温>"""

    try:
        feature_response = llm.invoke([HumanMessage(content=feature_prompt)])
        features_text = feature_response.content if hasattr(feature_response, "content") else str(feature_response)
    except Exception as e:
        called = list(state.get("called_agents", []))
        if "EXPERT" not in called:
            called.append("EXPERT")
        return {
            "expert_result": {"status": "failed", "error": f"特征提取失败: {str(e)[:200]}", "agent": "EXPERT"},
            "called_agents": called,
        }

    # ── Step 2: 检索历史经验 ──
    try:
        from rag.power_rag_pipeline import get_rag_pipeline
        rag = get_rag_pipeline()
        raw_matches = rag.retrieval.retrieve_expert_insight(
            features_text, province, top_k=10
        )
    except Exception as e:
        raw_matches = []

    if not raw_matches:
        called = list(state.get("called_agents", []))
        if "EXPERT" not in called:
            called.append("EXPERT")
        return {
            "expert_result": {
                "status": "success",
                "features": features_text[:200],
                "matched_cases": [],
                "consensus": "当前知识库暂无匹配的历史经验记录。建议后续补充该省专家复盘日志。",
                "agent": "EXPERT",
                "timestamp": datetime.now().isoformat(),
            },
            "called_agents": called,
        }

    # ── Step 3: LLM 精排 + 推理 ──
    cases_text = ""
    for i, m in enumerate(raw_matches[:10]):
        content = m.get("content", "")[:300]
        score = m.get("score", 0)
        cases_text += f"\n候选[{i+1}] 相似度={score:.2f}: {content}\n"

    rank_prompt = f"""当前市场特征: {features_text[:200]}

从专家日志中检索到的候选经验:
{cases_text}

请从候选经验中选出最匹配当前市场的 3-5 条，并输出 JSON:
{{
  "features": "提取的核心特征",
  "matched_cases": [
    {{
      "relevance": 0.95,
      "lesson": "老交易员的核心教训 (30字内)",
      "strategy": "当时采取的策略",
      "result": "策略执行结果"
    }}
  ],
  "consensus": "综合这些经验，大多数类似行情下老交易员的共同建议 (50字内)",
  "confidence": "经验匹配可信度: high/medium/low"
}}

只输出 JSON，不要其他文字。如果候选经验都不相关，matched_cases 设为空数组并说明原因。"""

    try:
        rank_response = llm.invoke([HumanMessage(content=rank_prompt)])
        rank_text = rank_response.content if hasattr(rank_response, "content") else str(rank_response)
    except Exception as e:
        called = list(state.get("called_agents", []))
        if "EXPERT" not in called:
            called.append("EXPERT")
        return {
            "expert_result": {
                "status": "failed", "error": f"LLM精排失败: {str(e)[:200]}",
                "agent": "EXPERT", "features": features_text[:200],
                "raw_case_count": len(raw_matches),
            },
            "called_agents": called,
        }

    # 解析 LLM JSON
    try:
        cleaned = rank_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        rank_json = json.loads(cleaned)
    except json.JSONDecodeError:
        rank_json = {
            "features": features_text[:200],
            "matched_cases": [],
            "consensus": rank_text[:300],
            "confidence": "low",
            "parse_error": "微信LLM输出JSON解析失败，上述为原始推理文本",
        }

    expert_result = {
        "status": "success",
        "features": rank_json.get("features", features_text[:200]),
        "matched_cases": rank_json.get("matched_cases", []),
        "consensus": rank_json.get("consensus", ""),
        "confidence": rank_json.get("confidence", "medium"),
        "raw_case_count": len(raw_matches),
        "matched_count": len(rank_json.get("matched_cases", [])),
        "agent": "EXPERT",
        "timestamp": datetime.now().isoformat(),
    }

    called = list(state.get("called_agents", []))
    if "EXPERT" not in called:
        called.append("EXPERT")

    return {
        "expert_result": expert_result,
        "called_agents": called,
    }


def compliance_node(state: dict) -> dict:
    """
    Compliance Agent 节点 — 合规与风控专家。
    挂载 Tools: search_policy, run_risk_validation
    输出: state["compliance_result"] = {policy_summary, risk_status, ...}
    """
    from tools.search_policy import search_policy
    from tools.factor_engine import run_risk_validation

    api_key = _get_api_key()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1, api_key=api_key)

    province = state.get("province", "山东")
    target_date = state.get("target_date", datetime.now().strftime("%Y-%m-%d"))
    user_query = state.get("user_query", "")

    tools = [search_policy, run_risk_validation]
    messages = [
        SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"请为 {province} 省 {target_date} 执行合规与风控检查。\n"
            f"用户原始问题: {user_query}\n"
            f"请依次调用 search_policy 和 run_risk_validation 两个工具。"
        )),
    ]

    from contracts.results import ComponentExecutionError

    try:
        result_text = _run_agent_loop(llm, tools, messages, agent_name="COMPLIANCE")

        compliance_result = {
            "status": "success",
            "province": province,
            "target_date": target_date,
            "summary": result_text[:1500],
            "agent": "COMPLIANCE",
            "timestamp": datetime.now().isoformat(),
        }

        # 提取风控状态 (PASS/FAIL/WARN → 将映射为ComplianceStatus在compliance_gate中)
        if "PASS" in result_text:
            compliance_result["risk_status"] = "PASS"
        elif "FAIL" in result_text:
            compliance_result["risk_status"] = "FAIL"
        elif "WARN" in result_text:
            compliance_result["risk_status"] = "WARN"
        else:
            compliance_result["risk_status"] = "UNKNOWN"

    except ComponentExecutionError as exc:
        compliance_result = {
            "status": "failed",
            "error": str(exc),
            "failure_type": exc.failure_type,
            "retryable": exc.retryable,
            "agent": "COMPLIANCE",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        compliance_result = {
            "status": "failed",
            "error": f"未预期异常: {type(e).__name__}: {str(e)[:300]}",
            "failure_type": "agent_failure",
            "retryable": False,
            "agent": "COMPLIANCE",
            "timestamp": datetime.now().isoformat(),
        }

    called = list(state.get("called_agents", []))
    if "COMPLIANCE" not in called:
        called.append("COMPLIANCE")

    return {
        "compliance_result": compliance_result,
        "called_agents": called,
    }


def strategy_node(state: dict) -> dict:
    """
    Strategy Agent 节点 — 策略与交易专家。

    ★ 显式上下文注入 (Context Injection):
      从 State 中读取 quant_result, expert_result, compliance_result，
      显式拼接为上下文文本，注入到 Strategy Agent 的输入中。

    ★ Reflection 自反思 (NEW):
      StrategyAgent 生成策略后，自动执行 5 维质检。不通过则修改。

    挂载 Tools: run_multi_role_analysis, run_scenario_simulation,
                run_strategy_advisor, generate_client_strategy_report,
                generate_bidding_sheet, generate_realtime_signal, generate_long_term_strategy
    输出: state["strategy_result"]
    """
    from tools.factor_engine import (
        run_multi_role_analysis,
        run_scenario_simulation,
        run_strategy_advisor,
    )
    from tools.generate_client_report import generate_client_strategy_report
    from tools.bidding_sheet import (
        generate_bidding_sheet,
        generate_realtime_signal,
        generate_long_term_strategy,
    )

    api_key = _get_api_key()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.2, api_key=api_key)

    province = state.get("province", "山东")
    target_date = state.get("target_date", datetime.now().strftime("%Y-%m-%d"))
    user_query = state.get("user_query", "")

    # ★ 显式上下文注入: 读取上游Agent的输出
    quant = state.get("quant_result") or {}
    expert = state.get("expert_result") or {}
    compliance = state.get("compliance_result") or {}

    injected_context = ""
    if quant and quant.get("status") == "success":
        injected_context += (
            f"\n【上游量化预测结果 (来自QUANT Agent)】\n"
            f"{quant.get('summary', '无数据')[:1200]}\n"
        )
    if quant and quant.get("status") == "failed":
        injected_context += (
            f"\n【量化预测状态】失败: {quant.get('error', '未知错误')[:200]}\n"
        )

    if expert and expert.get("status") == "success":
        cases = expert.get("matched_cases", [])
        consensus = expert.get("consensus", "")
        confidence = expert.get("confidence", "?")
        injected_context += (
            f"\n【★ 上游经验匹配结果 (来自EXPERT Agent)】\n"
            f"当前市场特征: {expert.get('features', '无')[:200]}\n"
            f"历史匹配案例: {len(cases)}条 (置信度: {confidence})\n"
            f"老交易员历史共识: {consensus[:500]}\n"
        )
        if cases:
            for i, c in enumerate(cases[:3]):
                injected_context += (
                    f"  案例{i+1}: 相关性={c.get('relevance','?')} | "
                    f"教训={c.get('lesson','')[:80]} | "
                    f"当时结果={c.get('result','')[:80]}\n"
                )
        injected_context += "\n"
    if expert and expert.get("status") == "failed":
        injected_context += (
            f"\n【经验匹配状态】失败: {expert.get('error', '未知错误')[:200]}\n"
        )

    if compliance and compliance.get("status") == "success":
        injected_context += (
            f"\n【上游合规风控结果 (来自COMPLIANCE Agent)】\n"
            f"风控状态: {compliance.get('risk_status', 'UNKNOWN')}\n"
            f"{compliance.get('summary', '无数据')[:1200]}\n"
        )
    if compliance and compliance.get("status") == "failed":
        injected_context += (
            f"\n【合规风控状态】失败: {compliance.get('error', '未知错误')[:200]}\n"
        )

    # ★ 加载完整的 Strategy Prompt (优先从文件)
    strategy_prompt = _load_strategy_prompt()

    tools = [
        run_multi_role_analysis,
        run_scenario_simulation,
        run_strategy_advisor,
        generate_client_strategy_report,
        generate_bidding_sheet,
        generate_realtime_signal,
        generate_long_term_strategy,
    ]

    messages = [
        SystemMessage(content=strategy_prompt),
        HumanMessage(content=(
            f"请为 {province} 省 {target_date} 生成交易策略。\n"
            f"用户原始问题: {user_query}\n"
            f"{injected_context}\n"
            f"请基于上述上游数据, 依次调用 run_multi_role_analysis, "
            f"run_scenario_simulation, run_strategy_advisor 三个工具。"
            f"如果用户需要生成申报决策表, 调用 generate_bidding_sheet。"
            f"如果用户需要实时交易信号, 调用 generate_realtime_signal。"
            f"如果用户需要中长期策略, 调用 generate_long_term_strategy。"
        )),
    ]

    from contracts.results import ComponentExecutionError

    try:
        result_text = _run_agent_loop(
            llm, tools, messages, max_iterations=8, agent_name="STRATEGY"
        )

        # NOTE: Reflection is performed by the independent 'reflection' node
        # in the LangGraph main graph — NOT inside Strategy itself.
        # This ensures Reflection is a true independent check, not self-verification.

        strategy_result = {
            "status": "success",
            "province": province,
            "target_date": target_date,
            "summary": result_text[:8000],
            "agent": "STRATEGY",
            "has_upstream_quant": bool(quant and quant.get("status") == "success"),
            "has_upstream_expert": bool(expert and expert.get("status") == "success"),
            "has_upstream_compliance": bool(compliance and compliance.get("status") == "success"),
            "timestamp": datetime.now().isoformat(),
        }

    except ComponentExecutionError as exc:
        strategy_result = {
            "status": "failed",
            "error": str(exc),
            "failure_type": exc.failure_type,
            "retryable": exc.retryable,
            "agent": "STRATEGY",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        strategy_result = {
            "status": "failed",
            "error": f"未预期异常: {type(e).__name__}: {str(e)[:300]}",
            "failure_type": "agent_failure",
            "retryable": False,
            "agent": "STRATEGY",
            "timestamp": datetime.now().isoformat(),
        }

    called = list(state.get("called_agents", []))
    if "STRATEGY" not in called:
        called.append("STRATEGY")

    return {
        "strategy_result": strategy_result,
        "called_agents": called,
    }
