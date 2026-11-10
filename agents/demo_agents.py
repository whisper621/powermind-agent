"""Deterministic database-backed Agents for the PowerMind V1 demo."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, timedelta
from typing import Any

from middleware.dynamic_prompt import PromptRegistry, get_prompt_registry
from retrieval.demo_rag import DemoRAGRetriever
from services.repository import DemoSQLiteMarketDataRepository


PROMPT_IDS = {
    "quant": "quant_system",
    "expert": "expert_system",
    "compliance": "compliance_system",
    "strategy": "strategy_system",
    "reflection": "reflection_system",
}


def _has_estimated_cost_claim(text: str) -> bool:
    for match in re.finditer(r"预计购电成本\s*(?:为|约|：|:|=)?\s*[-+]?\d", text):
        if not any(term in text[max(0, match.start() - 16):match.start()] for term in ("不", "未", "无法", "不能", "不得")):
            return True
    return False


def _has_unsupported_probability_claim(text: str) -> bool:
    phrases = (
        "平衡成交概率",
        "提高成交概率",
        "提升成交概率",
        "降低成交概率",
        "成交概率较高",
        "成交概率较低",
        "成交概率为",
        "成交概率约",
    )
    for phrase in phrases:
        start = 0
        while (index := text.find(phrase, start)) >= 0:
            prefix = text[max(0, index - 16):index]
            if not any(term in prefix for term in ("不", "未", "无法", "不能", "不得", "禁止")):
                return True
            start = index + len(phrase)
    return False


def _normalize_qualitative_probability_language(text: str) -> str:
    replacements = {
        "平衡成交概率": "平衡报价意愿与未成交风险",
        "提高成交概率": "提高报价意愿",
        "提升成交概率": "提高报价意愿",
        "降低成交概率": "提高未成交风险",
        "成交概率较高": "报价意愿较强",
        "成交概率较低": "未成交风险较高",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _evidence(
    evidence_id: str,
    *,
    evidence_type: str,
    source_name: str,
    source_locator: str,
    province: str,
    data_time: str,
    claim: str,
    source_type: str = "structured_data",
) -> dict:
    fingerprint = hashlib.sha256(claim.encode("utf-8")).hexdigest()[:16]
    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "source_type": source_type,
        "source_name": source_name,
        "source_locator": source_locator,
        "province": province,
        "data_time": data_time,
        "claim": claim,
        "content_fingerprint": fingerprint,
        "verification_status": "verified",
        "data_notice": "数据库演示假数据",
    }


def _base_result(agent_name: str, registry: PromptRegistry) -> dict:
    prompt_id = PROMPT_IDS[agent_name]
    asset = registry.get_asset(prompt_id)
    return {
        "status": "success",
        "agent": agent_name,
        "runtime_mode": "sqlite_demo_no_api",
        "prompt_id": prompt_id,
        "prompt_version": asset["version"],
        "prompt_file": asset["file"],
    }


def _quant(state: dict, repo: DemoSQLiteMarketDataRepository, registry: PromptRegistry) -> dict:
    province, target_date = state.get("province", "山东"), state.get("target_date", "")
    forecast = repo.get_forecast(province, target_date)
    factors = repo.get_market_factors(province, target_date)
    if len(forecast) != 24 or not factors:
        return {"status": "failed", "agent": "quant", "error": "数据库缺少24点存储预测或市场因子", "retryable": False}

    prices = [row["predicted_price"] for row in forecast]
    avg = round(sum(prices) / len(prices), 1)
    peak = round(sum(prices[18:21]) / 3, 1)
    valley = round(sum(prices[0:6]) / 6, 1)
    maximum = round(max(prices), 1)
    minimum = round(min(prices), 1)
    max_hour = prices.index(max(prices))
    min_hour = prices.index(min(prices))
    top = factors[:3]
    result = _base_result("quant", registry)
    result.update({
        "summary": (
            f"### Quant 数据结论\n"
            f"- 数据方式：SQLite已存储预测（未运行算法）\n"
            f"- {target_date}日均价：**{avg} 元/MWh**，18-20时均价：**{peak}**，00-05时均价：**{valley}**\n"
            f"- 日内最高/最低：**{maximum}（{max_hour:02d}:00）/ {minimum}（{min_hour:02d}:00）元/MWh**\n"
            f"- 预测区间：{min(row['lower_bound'] for row in forecast):.1f}～{max(row['upper_bound'] for row in forecast):.1f} 元/MWh\n"
            f"- 主导因子：" + "；".join(f"{item['factor_name']}({item['direction']}，{item['observation']})" for item in top)
        ),
        "metrics": {
            "avg_price": avg,
            "peak_price": peak,
            "valley_price": valley,
            "peak_avg_price": peak,
            "valley_avg_price": valley,
            "max_price": maximum,
            "max_price_hour": max_hour,
            "min_price": minimum,
            "min_price_hour": min_hour,
            "time_block_avg_spread": round(peak - valley, 1),
            "intraday_range": round(maximum - minimum, 1),
            "confidence": forecast[0]["confidence"],
        },
        "hourly_forecast": forecast,
        "market_factors": factors,
        "evidence": [
            _evidence(f"forecast:{province}:{target_date}", evidence_type="stored_forecast", source_name="forecasts", source_locator=f"powermind_demo.db/forecasts/{province}/{target_date}", province=province, data_time=target_date, claim=f"24点存储预测日均{avg}元/MWh、18-20时均价{peak}元/MWh、00-05时均价{valley}元/MWh、日内最高{maximum}元/MWh、最低{minimum}元/MWh"),
            _evidence(f"factors:{province}:{target_date}", evidence_type="market_factors", source_name="market_factors", source_locator=f"powermind_demo.db/market_factors/{province}/{target_date}", province=province, data_time=target_date, claim="；".join(item["observation"] for item in top)),
        ],
    })
    return result


def _expert(state: dict, rag: DemoRAGRetriever, registry: PromptRegistry) -> dict:
    province = state.get("province", "山东")
    task_type = state.get("task_type", "daily_pulse")
    boosters = {
        "daily_pulse": "高温 光伏 负荷",
        "bidding_advice": "申报 高温 午间 光伏 偏差",
        "long_term_locking": "中长期 基差 锁价 分批",
        "real_time_alert": "实时 日内 价差 偏差 晚峰",
    }
    docs = rag.search(f"{state.get('user_query', '')} {boosters.get(task_type, '')}", province=province, doc_type="expert_case", limit=3)
    if not docs:
        return {"status": "failed", "agent": "expert", "error": "数据库RAG未召回专家案例", "retryable": False}
    result = _base_result("expert", registry)
    result.update({
        "summary": "### Expert 相似案例\n" + "\n".join(
            f"- [{doc['doc_id']}] **{doc['title']}**：{doc['content']}" for doc in docs
        ),
        "retrieval": {"method": "sqlite_keyword_rag", "count": len(docs), "documents": docs},
        "evidence": [
            _evidence(doc["doc_id"], evidence_type="expert_case", source_name=doc["source_name"], source_locator=doc["source_locator"], province=province, data_time=doc["effective_date"], claim=doc["content"], source_type="rag_document")
            for doc in docs
        ],
    })
    return result


def _compliance(state: dict, rag: DemoRAGRetriever, registry: PromptRegistry) -> dict:
    province = state.get("province", "山东")
    task_type = state.get("task_type", "daily_pulse")
    boosters = {
        "policy_search": "政策 规则 偏差 考核",
        "bidding_advice": "日前 申报 限价 偏差 人工确认",
        "long_term_locking": "中长期 锁价 敞口 人工确认",
        "real_time_alert": "实时 日内 申报 限价 偏差 人工确认",
        "daily_pulse": "披露 证据 留痕",
    }
    docs = rag.search(f"{state.get('user_query', '')} {boosters.get(task_type, '')}", province=province, doc_type="policy", limit=4)
    if not docs:
        return {"status": "failed", "agent": "compliance", "error": "数据库RAG未召回演示政策", "retryable": False}
    policy_lines = []
    for doc in docs:
        policy_lines.append(f"- [{doc['doc_id']}] **{doc['title']}**（{doc['document_no']}）：{doc['content']}")
    result = _base_result("compliance", registry)
    result.update({
        "risk_status": "PASS",
        "summary": (
            "### Compliance 政策RAG与合规结论\n"
            "- 门禁结论：**PASS（仅限演示流程）**\n"
            "- 检索方式：SQLite关键词RAG，无网络调用\n"
            + "\n".join(policy_lines)
            + "\n- 合规建议：保留任务编号与证据链；申报和锁价建议必须由交易员人工确认，系统不得自动执行。"
        ),
        "retrieval": {"method": "sqlite_keyword_rag", "count": len(docs), "documents": docs},
        "rules": [doc["content"] for doc in docs],
        "evidence": [
            _evidence(doc["doc_id"], evidence_type="policy", source_name=doc["source_name"], source_locator=doc["source_locator"], province=province, data_time=doc["effective_date"], claim=doc["content"], source_type="rag_document")
            for doc in docs
        ],
    })
    return result


def _strategy(state: dict, repo: DemoSQLiteMarketDataRepository, registry: PromptRegistry) -> dict:
    province, target_date = state.get("province", "山东"), state.get("target_date", "")
    task_type = state.get("task_type", "daily_pulse")
    scenario = repo.get_scenario(task_type, province, target_date)
    if not scenario:
        return {"status": "failed", "agent": "strategy", "error": "数据库缺少目标日期的预制场景结果", "retryable": False}
    payload = scenario["payload"]
    quant = state.get("quant_result", {})
    metrics = quant.get("metrics", {})
    analysis_data = {}
    extra_evidence = []

    if task_type == "bidding_advice":
        history = repo.get_price_history(province, target_date, days=7)
        weather = repo.get_weather(province, target_date, days=3)
        bidding_history = repo.get_bidding_history(province, limit=7, before_date=target_date)
        analysis_data = {
            "price_history_7d": history,
            "weather_3d": weather,
            "bidding_history_7d": bidding_history,
            "hourly_forecast_24h": quant.get("hourly_forecast", []),
            "market_factors": quant.get("market_factors", []),
        }
        if history:
            extra_evidence.append(_evidence(
                f"history:{province}:{target_date}:7d",
                evidence_type="price_history",
                source_name="price_data",
                source_locator=f"powermind_demo.db/price_data/{province}/before/{target_date}",
                province=province,
                data_time=target_date,
                claim=json.dumps(history, ensure_ascii=False),
            ))
        if weather:
            extra_evidence.append(_evidence(
                f"weather-summary:{province}:{target_date}:3d",
                evidence_type="weather",
                source_name="weather_data",
                source_locator=f"powermind_demo.db/weather_data/{province}/{target_date}/3d",
                province=province,
                data_time=target_date,
                claim=json.dumps(weather, ensure_ascii=False),
            ))
        if bidding_history:
            extra_evidence.append(_evidence(
                f"bidding-history:{province}:{target_date}:7d",
                evidence_type="bidding_history",
                source_name="bidding_records",
                source_locator=f"powermind_demo.db/bidding_records/{province}/latest/7",
                province=province,
                data_time=target_date,
                claim=json.dumps(bidding_history, ensure_ascii=False),
            ))

    if task_type == "long_term_locking":
        history = repo.get_price_history(province, target_date, days=7)
        contracts = repo.get_contracts(province)
        basis = repo.get_basis_history(province, limit=6)
        horizon = []
        try:
            start = date.fromisoformat(target_date)
            for offset in range(10):
                day = start + timedelta(days=offset)
                forecast = repo.get_forecast(province, day.isoformat())
                if len(forecast) == 24:
                    prices = [row["predicted_price"] for row in forecast]
                    horizon.append({
                        "date": day.isoformat(),
                        "coverage": "24/24",
                        "avg_price": round(sum(prices) / 24, 1),
                        "peak_price": round(sum(prices[18:21]) / 3, 1),
                        "valley_price": round(sum(prices[:6]) / 6, 1),
                        "data_type": "存储预测",
                    })
                else:
                    horizon.append({
                        "date": day.isoformat(),
                        "coverage": f"{len(forecast)}/24",
                        "data_type": "数据缺失",
                    })
        except ValueError:
            horizon = []

        analysis_data = {
            "price_history_7d": history,
            "contracts": contracts,
            "basis_history": basis,
            "forecast_horizon_10d": horizon,
            "position_basis": payload.get("position_basis"),
            "forecast_demand_mwh": payload.get("forecast_demand_mwh"),
            "locked_volume_mwh": payload.get("locked_volume_mwh"),
            "remaining_exposure_mwh": payload.get("remaining_exposure_mwh"),
        }
        if history:
            extra_evidence.append(_evidence(
                f"history:{province}:{target_date}:7d",
                evidence_type="price_history",
                source_name="price_data",
                source_locator=f"powermind_demo.db/price_data/{province}/before/{target_date}",
                province=province,
                data_time=target_date,
                claim=json.dumps(history, ensure_ascii=False),
            ))
        if contracts:
            extra_evidence.append(_evidence(
                f"contracts:{province}:{target_date}",
                evidence_type="contract_position",
                source_name="contracts",
                source_locator=f"powermind_demo.db/contracts/{province}",
                province=province,
                data_time=target_date,
                claim=json.dumps(contracts, ensure_ascii=False),
            ))
        if basis:
            extra_evidence.append(_evidence(
                f"basis:{province}:{target_date}",
                evidence_type="basis_history",
                source_name="basis_history",
                source_locator=f"powermind_demo.db/basis_history/{province}",
                province=province,
                data_time=target_date,
                claim=json.dumps(basis, ensure_ascii=False),
            ))
        if horizon:
            extra_evidence.append(_evidence(
                f"forecast-horizon:{province}:{target_date}:10d",
                evidence_type="stored_forecast",
                source_name="forecasts",
                source_locator=f"powermind_demo.db/forecasts/{province}/{target_date}/10d",
                province=province,
                data_time=target_date,
                claim=json.dumps(horizon, ensure_ascii=False),
            ))

    if task_type == "bidding_advice":
        rows = "\n".join(
            f"| {item['period']} | {item['volume_mwh']} | {item['forecast_price']} | "
            f"{item['bid_price']} | {item['adjustment']:+} | {item['action']} |"
            for item in payload["segments"]
        )
        body = (
            f"### 决策摘要\n{payload['headline']}\n\n"
            f"### 分时量价建议\n| 时段 | 电量(MWh) | 时段预测均价 | 建议申报价 | 调整值 | 动作 |\n"
            f"|---|---:|---:|---:|---:|---|\n{rows}\n\n"
            f"### 主方案\n申报总量 **{payload['total_volume_mwh']} MWh**，加权均价约 **{payload['recommended_bid_avg']} 元/MWh**。\n\n"
            "### 备选方案\n" + "\n".join(f"- {item}" for item in payload["alternatives"]) + "\n\n"
            "### 停止条件\n" + "\n".join(f"- {item}" for item in payload["stop_conditions"])
        )
    elif task_type == "long_term_locking":
        rows = "\n".join(f"| {item['window']} | {item['volume_mwh']} | ≤{item['max_price']} |" for item in payload["batches"])
        body = (
            f"### 决策摘要\n{payload['headline']}\n\n"
            f"### 持仓与目标\n当前锁定 **{payload['current_lock_ratio']}%**，目标 **{payload['target_lock_ratio']}%**，建议新增 **{payload['recommended_add_volume_mwh']} MWh**。\n\n"
            f"### 分批方案\n| 批次窗口 | 电量(MWh) | 价格上限(元/MWh) |\n|---|---:|---:|\n{rows}\n\n"
            "### 备选方案\n" + "\n".join(f"- {item}" for item in payload["alternatives"]) + "\n\n"
            "### 停止条件\n" + "\n".join(f"- {item}" for item in payload["stop_conditions"])
        )
    elif task_type == "real_time_alert":
        rows = "\n".join(
            f"| {item['window']} | {item['action']} | {item['volume_mwh']} | {item['price_boundary']} | {item['trigger']} |"
            for item in payload["windows"]
        )
        body = (
            f"### 快照摘要\n"
            f"- 数据类型：**{payload['snapshot_type']}**（不是实时行情）\n"
            f"- 截止时间：**{payload['as_of']}**\n"
            f"- 日前基准价：**{payload['day_ahead_price']} 元/MWh**\n"
            f"- 快照价：**{payload['snapshot_price']} 元/MWh**\n"
            f"- 价差（快照价-日前价）：**{payload['spread']} 元/MWh**\n\n"
            f"### 分时调整建议\n"
            f"| 窗口 | 动作 | 电量(MWh) | 价格边界(元/MWh) | 触发条件 |\n"
            f"|---|---|---:|---:|---|\n{rows}\n\n"
            "### 风险与停止条件\n"
            + "\n".join(f"- {item}" for item in payload["stop_conditions"])
        )
    else:
        body = (
            f"### 今日一句话判断\n{payload['headline']}\n\n"
            f"### 核心数字\n日均 **{metrics.get('avg_price', payload['forecast_avg'])} 元/MWh**；晚峰 **{metrics.get('peak_price', payload['forecast_peak'])}**；谷段 **{metrics.get('valley_price', payload['forecast_valley'])}**。\n\n"
            "### 今日动作\n" + "\n".join(f"- {item}" for item in payload["actions"]) + "\n\n"
            "### 风险雷达\n" + "\n".join(f"- {item}" for item in payload["risks"])
        )

    body += "\n\n### 执行边界\n以上为数据库演示建议；未执行任何申报、成交或锁价，必须由交易员人工复核并在正式系统中手工操作。"
    scenario_prompt_id = f"{task_type}_scenario"
    scenario_prompt = registry.get_asset(scenario_prompt_id)
    result = _base_result("strategy", registry)
    result.update({
        "summary": body,
        "scenario": payload,
        "analysis_data": analysis_data,
        "scenario_prompt_id": scenario_prompt_id,
        "scenario_prompt_version": scenario_prompt["version"],
        "evidence": [
            _evidence(f"scenario:{task_type}:{province}:{target_date}", evidence_type="scenario_output", source_name="demo_scenarios", source_locator=f"powermind_demo.db/demo_scenarios/{task_type}/{province}/{target_date}", province=province, data_time=target_date, claim=json.dumps(payload, ensure_ascii=False)),
        ] + extra_evidence,
    })
    return result


def _reflection(state: dict, registry: PromptRegistry) -> dict:
    strategy = state.get("strategy_result", {})
    text = strategy.get("summary", "")
    checks = {
        "has_strategy": bool(text),
        "has_manual_confirmation": "人工" in text and "正式系统" in text,
        "has_evidence": bool(strategy.get("evidence")),
        "no_placeholders": not bool(re.search(r"\{[a-zA-Z_][^}]*\}", text)),
        "no_execution_claim": not any(term in text for term in ("已自动申报", "已自动成交", "已自动锁价")),
        "no_bid_as_expected_cost": not _has_estimated_cost_claim(text),
        "no_unsupported_probability": not _has_unsupported_probability_claim(text),
    }
    scenario = strategy.get("scenario", {})
    task_type = state.get("task_type")
    if task_type == "bidding_advice" and scenario.get("segments"):
        segments = scenario["segments"]
        total = sum(item["volume_mwh"] for item in segments)
        weighted = sum(item["volume_mwh"] * item["bid_price"] for item in segments) / total
        checks["segment_volume_total_matches"] = total == scenario.get("total_volume_mwh")
        checks["weighted_bid_avg_matches"] = abs(weighted - scenario.get("recommended_bid_avg", 0)) <= 0.1
    elif task_type == "long_term_locking" and scenario.get("forecast_demand_mwh"):
        demand = scenario["forecast_demand_mwh"]
        current_ratio = scenario["locked_volume_mwh"] / demand * 100
        add_volume = demand * (scenario["target_lock_ratio"] - scenario["current_lock_ratio"]) / 100
        checks["current_lock_ratio_matches"] = abs(current_ratio - scenario["current_lock_ratio"]) <= 0.1
        checks["recommended_add_volume_matches"] = abs(add_volume - scenario["recommended_add_volume_mwh"]) <= 0.1
    elif task_type == "real_time_alert" and scenario:
        checks["spread_matches"] = abs(
            scenario.get("snapshot_price", 0)
            - scenario.get("day_ahead_price", 0)
            - scenario.get("spread", 0)
        ) <= 0.1
    issues = [name for name, passed in checks.items() if not passed]
    result = _base_result("reflection", registry)
    result.update({
        "passed": not issues,
        "score": round(sum(checks.values()) / len(checks) * 10, 1),
        "checks": checks,
        "issues": issues,
        "revision_instruction": "" if not issues else "补齐人工确认、证据或清除占位符后再交付",
        "summary": "Reflection通过：完整性、证据引用、占位符、人工确认和执行边界均符合要求。" if not issues else f"Reflection未通过：{', '.join(issues)}",
        "evidence": [],
    })
    if issues:
        result["status"] = "failed"
        result["error"] = result["summary"]
    return result


def run_demo_agent(
    state: dict,
    agent_name: str,
    *,
    repository: DemoSQLiteMarketDataRepository | None = None,
    retriever: DemoRAGRetriever | None = None,
    prompt_registry: PromptRegistry | None = None,
) -> dict[str, Any]:
    """Execute one Agent using only seeded SQLite data."""
    repo = repository or DemoSQLiteMarketDataRepository()
    rag = retriever or DemoRAGRetriever(repo.db_path)
    registry = prompt_registry or get_prompt_registry()
    if agent_name == "quant":
        return _quant(state, repo, registry)
    if agent_name == "expert":
        return _expert(state, rag, registry)
    if agent_name == "compliance":
        return _compliance(state, rag, registry)
    if agent_name == "strategy":
        return _strategy(state, repo, registry)
    if agent_name == "reflection":
        return _reflection(state, registry)
    return {"status": "failed", "agent": agent_name, "error": f"未知演示Agent: {agent_name}", "retryable": False}


class DeepSeekAgentRunner:
    """Use DeepSeek for Agent analysis while keeping facts and gates local."""

    def __init__(
        self,
        repository: DemoSQLiteMarketDataRepository,
        retriever: DemoRAGRetriever,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        from langchain_deepseek import ChatDeepSeek

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY未配置")
        self.repository = repository
        self.retriever = retriever
        self.registry = prompt_registry or get_prompt_registry()
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.llm = ChatDeepSeek(
            model=self.model,
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0.1,
            max_tokens=3500,
            timeout=60,
            max_retries=2,
            extra_body={"thinking": {"type": "disabled"}},
        )

    def run(self, state: dict, agent_name: str) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        base = run_demo_agent(
            state,
            agent_name,
            repository=self.repository,
            retriever=self.retriever,
            prompt_registry=self.registry,
        )
        if base.get("status") != "success":
            return base
        task_type = state.get("task_type", "daily_pulse")
        if agent_name != "strategy" and not (
            agent_name == "compliance" and task_type == "policy_search"
        ):
            return base

        prompt = self.registry.get_asset(PROMPT_IDS[agent_name])
        runtime_rules = self.registry.get_asset("component_runtime_rules")["content"]
        evidence_by_id = {
            item.get("evidence_id"): item
            for result in [base] + [
                state.get(key) or {}
                for key in ("quant_result", "expert_result", "compliance_result")
            ]
            for item in result.get("evidence", [])
            if item.get("evidence_id")
        }
        evidence = [
            {
                "evidence_id": item.get("evidence_id"),
                "claim": str(item.get("claim", ""))[:300],
                "source_locator": item.get("source_locator"),
            }
            for item in evidence_by_id.values()
        ]
        upstream_limit = 6000 if agent_name == "reflection" else 1800
        upstream = {
            key: str((state.get(key) or {}).get("summary", ""))[:upstream_limit]
            for key in ("quant_result", "expert_result", "compliance_result", "strategy_result")
            if state.get(key)
        }
        request = {
            "user_query": state.get("user_query"),
            "province": state.get("province"),
            "target_date": state.get("target_date"),
            "task_type": state.get("task_type"),
            "database_result": str(base.get("summary", ""))[:6000],
            "database_payload": base.get("scenario", {}),
            "database_analysis": base.get("analysis_data", {}),
            "evidence": evidence,
            "upstream_agent_outputs": upstream,
            "upstream_structured": {
                "quant": {
                    "metrics": (state.get("quant_result") or {}).get("metrics", {}),
                    "hourly_forecast": (state.get("quant_result") or {}).get("hourly_forecast", []),
                    "market_factors": (state.get("quant_result") or {}).get("market_factors", []),
                },
                "expert": (state.get("expert_result") or {}).get("retrieval", {}),
                "compliance": {
                    "risk_status": (state.get("compliance_result") or {}).get("risk_status"),
                    "retrieval": (state.get("compliance_result") or {}).get("retrieval", {}),
                    "rules": (state.get("compliance_result") or {}).get("rules", []),
                },
            },
        }
        scenario_text = ""
        if agent_name == "strategy" or (agent_name == "compliance" and task_type == "policy_search"):
            scenario_id = self.registry.SCENARIO_PROMPTS.get(task_type)
            if scenario_id:
                scenario_text = self.registry.get_asset(scenario_id)["content"]
                replacements = {
                    "{province}": str(state.get("province", "")),
                    "{target_date}": str(state.get("target_date", "")),
                    "{date}": str(state.get("target_date", "")),
                    "{risk_preference}": "未提供（采用平衡型演示）",
                    "{upstream_data}": "见数据库结果、database_payload 与上游 Agent 输出",
                }
                for placeholder, value in replacements.items():
                    scenario_text = scenario_text.replace(placeholder, value)
        system_text = (
            f"{prompt['content']}\n\n{scenario_text}\n\n{runtime_rules}\n\n"
            "当前是SQLite+DeepSeek演示模式。数据库结果和证据是事实边界，"
            "不得发明任何数字、政策、来源或执行状态。不要输出隐藏思维链；"
            "只输出可公开审计的分析摘要，说明结论、依据、不确定性和人工确认边界。"
        )
        user_text = (
            "请根据以下数据库结果完成本Agent的公开分析输出。"
            "保留业务所需的表格、主方案、备选方案和停止条件；"
            "正文不要显示evidence_id；证据映射只放在PUBLIC_ANALYSIS标记块中。\n\n"
            + json.dumps(request, ensure_ascii=False)
        )

        retry_mode = "full_prompt"
        first_error = ""
        required_terms = {
            "bidding_advice": ("数据覆盖", "近7日", "目标日", "日内极差", "申报", "总结", "PUBLIC_ANALYSIS_START"),
            "long_term_locking": ("数据覆盖", "当前持仓", "未来10日", "分批", "总结", "PUBLIC_ANALYSIS_START"),
            "real_time_alert": ("快照摘要", "当前敞口", "分时调整", "确认清单", "PUBLIC_ANALYSIS_START"),
        }.get(task_type, ())
        forbidden_terms = {
            "bidding_advice": ("峰谷价差",),
        }.get(task_type, ())

        def missing_contract_terms(output: str) -> list[str]:
            if agent_name != "strategy":
                return []
            issues = (
                [f"缺少:{term}" for term in required_terms if term not in output]
                + [f"禁用:{term}" for term in forbidden_terms if term in output]
            )
            if _has_estimated_cost_claim(output):
                issues.append("禁用:无出清依据的预计购电成本")
            if _has_unsupported_probability_claim(output):
                issues.append("禁用:无模型依据的成交概率")
            return issues

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ])
            content = response.content if isinstance(response.content, str) else str(response.content)
            content = _normalize_qualitative_probability_language(content)
            if not content.strip():
                raise RuntimeError("DeepSeek返回空内容")
            missing = missing_contract_terms(content)
            if missing:
                raise RuntimeError(f"DeepSeek输出契约缺失: {', '.join(missing)}")
        except Exception as first_exc:
            retry_mode = "compact_retry"
            first_error = f"{type(first_exc).__name__}: {str(first_exc)[:240]}"
            compact_request = {
                "user_query": state.get("user_query"),
                "province": state.get("province"),
                "target_date": state.get("target_date"),
                "task_type": task_type,
                "database_payload": base.get("scenario", {}),
                "database_analysis": base.get("analysis_data", {}),
                "quant_metrics": (state.get("quant_result") or {}).get("metrics", {}),
                "evidence": evidence,
            }
            try:
                response = self.llm.invoke([
                    SystemMessage(content=(
                        f"{prompt['content']}\n\n{scenario_text}\n\n{runtime_rules}\n\n"
                        "这是一次结构纠正生成。严格完整输出场景要求的全部章节和"
                        "PUBLIC_ANALYSIS标记块；只使用给定数据，不得新增数字、政策或执行状态，"
                        "不输出隐藏思维链。"
                    )),
                    HumanMessage(content=(
                        f"请重写为完整交付稿。上次未通过原因：{first_error}\n"
                        "数据库输入如下：\n"
                        + json.dumps(compact_request, ensure_ascii=False)
                    )),
                ])
                content = response.content if isinstance(response.content, str) else str(response.content)
                content = _normalize_qualitative_probability_language(content)
                if not content.strip():
                    raise RuntimeError("DeepSeek紧凑重试返回空内容")
                missing = missing_contract_terms(content)
                if missing:
                    raise RuntimeError(f"DeepSeek紧凑重试仍缺失: {', '.join(missing)}")
            except Exception as retry_exc:
                return {
                    **base,
                    "llm_used": False,
                    "llm_status": "degraded",
                    "llm_model": self.model,
                    "llm_error": (
                        f"首次:{type(first_exc).__name__}; "
                        f"重试:{type(retry_exc).__name__}: {str(retry_exc)[:180]}"
                    ),
                    "summary": (
                        "⚠️ DeepSeek调用失败，本节点已透明降级为纯数据库规则结果。\n\n"
                        + base.get("summary", "")
                    ),
                }

        content = content.strip()
        if agent_name == "strategy" and not ("人工" in content and "正式系统" in content):
            content += (
                "\n\n### 执行边界\n"
                "以上为数据库演示建议；未执行任何申报、成交或锁价，"
                "必须由交易员人工复核并在正式系统中手工操作。"
            )

        return {
            **base,
            "deterministic_summary": base.get("summary", ""),
            "summary": content,
            "runtime_mode": "sqlite_rag_deepseek",
            "llm_used": True,
            "llm_status": "success",
            "llm_model": self.model,
            "llm_retry_mode": retry_mode,
            "llm_first_error": first_error or None,
            "usage_metadata": getattr(response, "usage_metadata", None) or {},
            "analysis_disclosure": "公开分析摘要，不含模型隐藏思维链",
        }
