#!/usr/bin/env python3
"""
PowerMind Chainlit — Power trading assistant frontend
Start: chainlit run app_chainlit.py --port 8000

Intent routing reads from intents/task_intents.yaml — no hardcoded keyword tables.
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import date, datetime, timedelta
import re

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from dotenv import load_dotenv
from agents.react_graph_agent import PowerMindAgentV3


# ── Load intent library ───────────────────────────────────────────────────
def _load_intents():
    intents_path = PROJECT_ROOT / "intents" / "task_intents.yaml"
    if intents_path.exists():
        with open(intents_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("intents", [])
    return []


INTENTS = _load_intents()


def _match_intent(query_lower: str) -> str:
    """Choose the active intent with the strongest keyword match."""
    candidates = []
    for intent in INTENTS:
        if intent.get("status") != "active":
            continue
        matches = [kw for kw in intent.get("keywords", []) if kw.lower() in query_lower]
        workflow = intent.get("workflow_id") or "general"
        if matches:
            candidates.append((len(matches), max(len(item) for item in matches), workflow))
    return max(candidates, default=(0, 0, "daily_pulse"))[2]


def _parse_target_date(query: str) -> str:
    match = re.search(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", query)
    if match:
        return datetime.strptime(match.group().replace("/", "-"), "%Y-%m-%d").date().isoformat()
    if "明天" in query or "明日" in query or "tomorrow" in query.lower():
        return (date.today() + timedelta(days=1)).isoformat()
    return date.today().isoformat()


def _parse_province(query: str, current: str = "山东") -> str:
    supported = ("山东",)
    for province in supported:
        if province in query:
            return province
    return current


def _trace_evidence(state: dict) -> str:
    lines = []
    for item in state.get("evidence_records", []):
        claim = str(item.get("claim", ""))[:180]
        lines.append(
            f"- `{item.get('evidence_id', '?')}`：{claim}\n"
            f"  - 定位：`{item.get('source_locator', '?')}`"
        )
    return "\n".join(lines) or "未收集到证据。"


def _split_public_analysis(text: str) -> tuple[str, str]:
    start_marker = "<!-- PUBLIC_ANALYSIS_START -->"
    end_marker = "<!-- PUBLIC_ANALYSIS_END -->"
    if start_marker in text and end_marker in text:
        before, rest = text.split(start_marker, 1)
        analysis, after = rest.split(end_marker, 1)
        return (before + after).strip(), analysis.strip()

    match = re.search(
        r"(?ms)^###\s*(?:\d+\.\s*)?公开分析依据\s*\n"
        r"(.*?)(?=^###\s*(?:\d+\.\s*)?(?:交易员操作清单|执行清单|总结)\s*$|\Z)",
        text,
    )
    if not match:
        return text.strip(), ""
    return (text[:match.start()] + text[match.end():]).strip(), match.group(1).strip()


def _render_business_result(state: dict) -> str:
    """Render only the deliverable; execution details live in Chainlit Steps."""
    task_type = state.get("task_type", "general")
    result = (
        state.get("compliance_result", {})
        if task_type == "policy_search"
        else state.get("strategy_result", {})
    )
    body, _ = _split_public_analysis(result.get("summary", "任务未形成可用业务结果。"))
    verification = state.get("verification_result", {})
    reflection = state.get("reflection_result", {})
    quality = (
        f"数据库证据覆盖 {verification.get('verified_claims', 0)}/"
        f"{verification.get('total_claims', 0)}"
    )
    if task_type != "policy_search":
        quality += f"｜规则型 Reflection：{'通过' if reflection.get('passed') else '未通过'}"
    llm_used = bool(result.get("llm_used"))
    runtime_note = (
        f"DeepSeek {result.get('llm_model')} 已参与分析表达"
        if llm_used
        else "本次未使用LLM或已透明降级"
    )
    return (
        f"# {state.get('province', '')} · {state.get('target_date', '')}\n\n"
        f"{body}\n\n"
        f"---\n"
        f"**演示校验**：{quality}｜{runtime_note}\n\n"
        f"> 市场与政策内容为数据库演示假数据；DeepSeek只基于这些数据分析，未进行真实预测、申报、成交或锁价。"
    )


async def _send_execution_steps(state: dict) -> None:
    """Expose observable execution records without fabricating chain-of-thought."""
    plan_steps = [item.get("step_id") for item in state.get("plan", {}).get("steps", [])]
    async with cl.Step(name="路由与固定工作流", type="run", default_open=False) as step:
        step.input = {
            "task_type": state.get("task_type"),
            "province": state.get("province"),
            "target_date": state.get("target_date"),
        }
        step.output = {"plan": plan_steps, "completed": state.get("completed_steps", [])}

    async with cl.Step(name="Context Assembly", type="retrieval", default_open=False) as step:
        step.input = "从 powermind_demo.db 读取历史电价与天气"
        step.output = state.get("structured_data", [])

    for result_key, label, step_type in (
        ("quant_result", "Quant Agent · 存储预测与因子", "tool"),
        ("expert_result", "Expert Agent · 案例RAG", "retrieval"),
        ("compliance_result", "Compliance Agent · 政策RAG与门禁", "retrieval"),
    ):
        result = state.get(result_key)
        if not result:
            continue
        async with cl.Step(name=label, type=step_type, default_open=False) as step:
            step.input = {
                "prompt": f"{result.get('prompt_id', '?')}@{result.get('prompt_version', '?')}",
                "runtime": result.get("runtime_mode"),
                "llm_used": result.get("llm_used", False),
                "llm_model": result.get("llm_model"),
                "llm_status": result.get("llm_status"),
            }
            step.output = result.get("summary", result)

    _, public_analysis = _split_public_analysis(
        (state.get("strategy_result") or {}).get("summary", "")
    )
    strategy = state.get("strategy_result") or {}
    if strategy.get("llm_status") == "degraded":
        async with cl.Step(name="DeepSeek 降级详情", type="tool", default_open=False) as step:
            step.input = {
                "model": strategy.get("llm_model"),
                "status": strategy.get("llm_status"),
            }
            step.output = {
                "error": str(strategy.get("llm_error", "未知错误"))[:500],
                "fallback": "已使用SQLite确定性策略结果，未中断闭环",
            }
    if public_analysis:
        async with cl.Step(name="公开分析依据", type="tool", default_open=False) as step:
            step.input = "数据覆盖 → 指标口径 → 驱动因素 → 方案映射 → 反向情景 → 数值自检"
            step.output = public_analysis

    reflection = state.get("reflection_result")
    if reflection:
        async with cl.Step(name="规则型 Reflection", type="tool", default_open=False) as step:
            step.input = "检查完整性、证据、占位符、人工确认和执行边界"
            step.output = {
                "passed": reflection.get("passed"),
                "score": reflection.get("score"),
                "checks": reflection.get("checks"),
                "issues": reflection.get("issues"),
                "public_review": reflection.get("summary"),
                "llm_used": reflection.get("llm_used", False),
                "llm_model": reflection.get("llm_model"),
            }

    async with cl.Step(name="EvidencePipeline · 数据库证据", type="retrieval", default_open=False) as step:
        step.input = "汇总并按 evidence_id 去重"
        step.output = _trace_evidence(state)

    verification = state.get("verification_result", {})
    async with cl.Step(name="Verifier · 交付检查", type="tool", default_open=False) as step:
        step.input = "校验数据库来源定位、合规门禁和规则型 Reflection"
        step.output = {
            "passed": verification.get("passed"),
            "verification_scope": "演示数据库记录完整性，不代表现实真实性",
            "database_evidence_coverage": (
                f"{verification.get('verified_claims', 0)}/"
                f"{verification.get('total_claims', 0)}"
            ),
            "issues": verification.get("issues", []),
        }


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="生成今日市场日报", message="生成山东今天的市场日报"),
        cl.Starter(label="制定日前申报策略", message="山东明天日前申报怎么做？"),
        cl.Starter(label="检查实时交易窗口", message="山东当前实时交易怎么调整？"),
        cl.Starter(label="查询偏差考核规则", message="山东偏差考核标准是什么？"),
        cl.Starter(label="分析中长期锁价", message="山东中长期应该锁多少电量？"),
    ]


@cl.on_chat_start
async def start():
    load_dotenv(PROJECT_ROOT / "config" / ".env")
    use_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))
    cl.user_session.set("agent", PowerMindAgentV3(use_deepseek=use_deepseek))
    cl.user_session.set("province", "山东")
    cl.user_session.set("runtime_mode", "sqlite_rag_deepseek" if use_deepseek else "sqlite_demo_no_api")

    await cl.Message(content=f"""# ⚡ PowerMind V1 · 电力交易智能协作台

**SQLite + {'DeepSeek V4' if use_deepseek else '无LLM'} 闭环演示已就绪**｜{datetime.now().strftime('%Y-%m-%d')}｜山东省

系统会依次展示：任务路由 → 数据上下文 → Quant → Expert → Compliance → Strategy → Reflection → 证据链 → Verifier → 详细输出。

### 直接试问
- `生成山东今天的市场日报`
- `山东明天日前申报怎么做？`
- `山东偏差考核标准是什么？`
- `山东中长期应该锁多少电量？`

> 市场、政策和案例均为SQLite假数据；DeepSeek仅用于Agent分析表达，不连接预测算法或正式交易系统。
""").send()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    province = _parse_province(message.content, cl.user_session.get("province", "山东"))
    cl.user_session.set("province", province)

    task = _match_intent(message.content.lower())
    target_date = _parse_target_date(message.content)

    import asyncio
    loop = asyncio.get_running_loop()
    labels = {
        "daily_pulse": "市场日报",
        "bidding_advice": "日前申报",
        "real_time_alert": "实时交易",
        "policy_search": "政策RAG",
        "long_term_locking": "中长期锁价",
    }
    async with cl.Step(
        name=f"{labels.get(task, task)} · 固定工作流",
        type="run",
        default_open=True,
        auto_collapse=True,
    ) as workflow_step:
        workflow_step.input = {
            "query": message.content,
            "province": province,
            "target_date": target_date,
            "runtime": cl.user_session.get("runtime_mode", "sqlite_demo_no_api"),
        }
        state = await loop.run_in_executor(
            None,
            lambda: agent.execute_state(
                query=message.content,
                province=province,
                task_type=task,
                target_date=target_date,
            ),
        )
        await _send_execution_steps(state)
        workflow_step.output = {
            "status": state.get("task_status"),
            "verification_passed": state.get("verification_result", {}).get("passed"),
            "llm_agents": [
                key.replace("_result", "")
                for key in ("quant_result", "expert_result", "compliance_result", "strategy_result", "reflection_result")
                if (state.get(key) or {}).get("llm_used")
            ],
            "note": "以上是可审计执行记录，不是模型隐式思维链。",
        }

    await cl.Message(content=_render_business_result(state)).send()
