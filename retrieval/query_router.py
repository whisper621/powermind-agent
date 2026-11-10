"""Query Router — Intent classification for multi-channel routing.

Classifies user queries into one of 8 intents:
  - POLICY_SEARCH    -> route to policy docs + GraphRAG
  - DATA_ANALYSIS    -> route to SQL + quantitative tools
  - REAL_TIME_ALERT  -> route to market snapshots + monitoring
  - GENERAL_QA       -> route to full hybrid retrieval
  - BIDDING_ADVICE   -> route to strategy + quant
  - CLIENT_REPORT    -> route to report generation pipeline
  - EXPERT_INSIGHT   -> route to expert channel + historical cases
  - RISK_ASSESSMENT  -> route to policy + risk validation

Uses a keyword + heuristic classifier with LLM fallback for ambiguous queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    POLICY_SEARCH = "policy_search"
    DATA_ANALYSIS = "data_analysis"
    REAL_TIME_ALERT = "real_time_alert"
    GENERAL_QA = "general_qa"
    BIDDING_ADVICE = "bidding_advice"
    CLIENT_REPORT = "client_report"
    EXPERT_INSIGHT = "expert_insight"  # 经验查询：历史案例匹配、专家复盘检索
    RISK_ASSESSMENT = "risk_assessment"  # 风险检查：偏差考核、红线触发、风险评估


# Mapping from Intent to retrieval channels
INTENT_CHANNELS: dict[Intent, list[str]] = {
    Intent.POLICY_SEARCH: ["policy", "graph", "bm25"],
    Intent.DATA_ANALYSIS: ["sql", "market", "dense", "price_history"],
    Intent.REAL_TIME_ALERT: ["market", "sql", "dense"],
    Intent.GENERAL_QA: ["bm25", "dense", "policy", "graph", "market"],
    Intent.BIDDING_ADVICE: ["policy", "graph", "sql", "dense", "price_history"],
    Intent.CLIENT_REPORT: ["sql", "market", "dense", "price_history"],
    Intent.EXPERT_INSIGHT: ["expert", "market", "dense", "price_history"],
    Intent.RISK_ASSESSMENT: ["policy", "graph", "sql"],
}

# Intent -> required specialist agents
INTENT_AGENTS: dict[Intent, list[str]] = {
    Intent.POLICY_SEARCH: ["compliance"],
    Intent.DATA_ANALYSIS: ["quant"],
    Intent.REAL_TIME_ALERT: ["quant", "compliance"],
    Intent.GENERAL_QA: ["expert"],
    Intent.BIDDING_ADVICE: ["quant", "compliance", "strategy"],
    Intent.CLIENT_REPORT: ["quant", "strategy"],
    Intent.EXPERT_INSIGHT: ["expert"],
    Intent.RISK_ASSESSMENT: ["compliance"],
}


@dataclass
class QueryRoute:
    """Routing decision produced by QueryRouter."""

    intent: Intent
    confidence: float  # 0.0 - 1.0
    channels: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    extracted_entities: dict[str, str] = field(default_factory=dict)
    needs_clarification: bool = False
    clarification_question: str = ""


# ---------------------------------------------------------------------------
# Keyword-intent mapping (no LLM needed for unambiguous cases)
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS: dict[Intent, list[list[str]]] = {
    Intent.POLICY_SEARCH: [
        ["政策", "文件", "规定", "通知", "办法", "细则"],
        ["偏差", "考核", "免责", "免考"],
        ["准入", "资质", "备案"],
        ["电价", "上限", "下限", "浮动", "目录电价"],
    ],
    Intent.DATA_ANALYSIS: [
        ["分析", "趋势", "统计", "对比", "同比", "环比"],
        ["因素", "因子", "指标", "信号"],
        ["预测", "走势", "预判"],
        ["负荷", "发电", "用电量", "装机"],
    ],
    Intent.REAL_TIME_ALERT: [
        ["实时", "预警", "报警", "告警"],
        ["当前", "今日", "现在", "最新"],
        ["异常", "突变", "跳变"],
        ["供需", "缺口", "紧张"],
    ],
    Intent.BIDDING_ADVICE: [
        ["报价", "投标", "竞价", "交易策略"],
        ["日前", "日内", "现货", "中长期"],
        ["套利", "收益", "成本", "盈亏"],
        ["仓位", "持仓", "头寸"],
    ],
    Intent.CLIENT_REPORT: [
        ["报告", "报表", "汇报"],
        ["客户", "用电户", "售电公司"],
        ["月报", "周报", "日报", "总结"],
        ["方案", "建议书"],
    ],
    Intent.EXPERT_INSIGHT: [
        ["经验", "案例", "历史", "类似行情", "之前"],
        ["复盘", "总结", "教训"],
        ["老交易员", "专家", "心得"],
        ["上次", "以前", "上回", "过去"],
    ],
    Intent.RISK_ASSESSMENT: [
        ["风险", "会不会", "可能", "触发"],
        ["偏差考核", "罚款", "惩罚"],
        ["红线", "警戒", "超标"],
        ["安全", "合规性", "违规"],
    ],
}


class QueryRouter:
    """Fast intent classifier using keyword heuristics with LLM fallback."""

    def __init__(self, llm=None):
        self._llm = llm  # optional LLM for ambiguous cases

    # ----- public API --------------------------------------------------------

    def classify(self, query: str, context: dict | None = None) -> QueryRoute:
        """Classify query intent and produce a routing decision."""
        ctx = context or {}

        # Step 1: keyword scoring
        scores = self._keyword_score(query)
        top_intent, top_score = max(scores.items(), key=lambda x: x[1])

        # Step 2: entity extraction
        entities = self._extract_entities(query, ctx)

        # Step 3: decide — LLM fallback for low-confidence cases
        if top_score < 2:
            route = self._llm_classify(query, ctx)
        else:
            route = QueryRoute(
                intent=top_intent,
                confidence=min(top_score / 5.0, 1.0),
                channels=list(INTENT_CHANNELS.get(top_intent, ["bm25", "dense"])),
                agents=list(INTENT_AGENTS.get(top_intent, ["expert"])),
                extracted_entities=entities,
            )

        # Override province from context
        if "province" in ctx and "province" not in route.extracted_entities:
            route.extracted_entities["province"] = ctx["province"]

        logger.info(
            "QueryRouter: intent=%s confidence=%.2f channels=%s",
            route.intent.value,
            route.confidence,
            route.channels,
        )
        return route

    # ----- keyword scoring ---------------------------------------------------

    def _keyword_score(self, query: str) -> dict[Intent, int]:
        scores: dict[Intent, int] = {intent: 0 for intent in Intent}
        for intent, keyword_groups in _INTENT_KEYWORDS.items():
            for group in keyword_groups:
                if any(kw in query for kw in group):
                    scores[intent] += 1
        # Boost GENERAL_QA as fallback
        if max(scores.values()) == 0:
            scores[Intent.GENERAL_QA] = 1
        return scores

    # ----- entity extraction -------------------------------------------------

    @staticmethod
    def _extract_entities(query: str, ctx: dict) -> dict[str, str]:
        entities: dict[str, str] = {}

        # Province detection
        PROVINCES = [
            "山东", "江苏", "浙江", "广东", "河南", "山西",
            "河北", "安徽", "福建", "湖南", "湖北", "四川",
            "辽宁", "吉林", "黑龙江", "陕西", "甘肃", "青海",
            "云南", "贵州", "海南", "江西", "内蒙古", "新疆",
            "西藏", "广西", "宁夏",
        ]
        for p in PROVINCES:
            if p in query:
                entities["province"] = p
                break

        # Date detection (simple: YYYY-MM-DD or YYYY年M月D日)
        import re

        date_patterns = [
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{4}年\d{1,2}月\d{1,2}日)",
        ]
        for pat in date_patterns:
            m = re.search(pat, query)
            if m:
                entities["date"] = m.group(1)
                break

        # Client ID detection
        client_match = re.search(r"客户[:\s]*(\w+)", query)
        if client_match:
            entities["client_id"] = client_match.group(1)

        return entities

    # ----- LLM fallback ------------------------------------------------------

    def _llm_classify(self, query: str, ctx: dict) -> QueryRoute:
        """Use LLM for ambiguous intent classification."""
        if self._llm is None:
            # Hard fallback to GENERAL_QA
            return QueryRoute(
                intent=Intent.GENERAL_QA,
                confidence=0.3,
                channels=["bm25", "dense", "policy"],
                agents=["expert"],
                extracted_entities=self._extract_entities(query, ctx),
            )

        prompt = (
            "你是一个电力行业查询分类器。请将以下用户查询分类为以下之一：\n"
            "- policy_search: 政策法规、文件、规定查询\n"
            "- data_analysis: 数据分析、趋势、预测\n"
            "- real_time_alert: 实时预警、当前状态\n"
            "- bidding_advice: 交易策略、报价建议\n"
            "- client_report: 客户报告、方案\n"
            "- expert_insight: 经验查询、历史案例、复盘心得\n"
            "- risk_assessment: 风险检查、偏差考核、红线评估\n"
            "- general_qa: 通用问答\n\n"
            f"查询: {query}\n\n"
            "请仅返回 JSON: {{\"intent\": \"<类别>\", \"confidence\": <0.0-1.0>}}"
        )
        try:
            resp = self._llm.invoke(prompt)
            import json

            data = json.loads(resp.content if hasattr(resp, "content") else str(resp))
            intent = Intent(data.get("intent", "general_qa"))
            return QueryRoute(
                intent=intent,
                confidence=float(data.get("confidence", 0.5)),
                channels=list(INTENT_CHANNELS.get(intent, ["bm25", "dense"])),
                agents=list(INTENT_AGENTS.get(intent, ["expert"])),
                extracted_entities=self._extract_entities(query, ctx),
            )
        except Exception:
            return QueryRoute(
                intent=Intent.GENERAL_QA,
                confidence=0.3,
                channels=["bm25", "dense", "policy"],
                agents=["expert"],
                extracted_entities=self._extract_entities(query, ctx),
            )
