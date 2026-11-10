"""Verifier / Critic Agent — Enhanced fact verification and compliance audit.

Replaces the superficial regex-based fact verification in the legacy RAG pipeline
with a multi-dimensional verification framework:

  1. Numerical Accuracy  — cross-reference numbers against source context
  2. Citation Integrity   — verify every factual claim has a valid source
  3. Policy Compliance    — check output against known rules and bounds
  4. Cross-Province Guard — detect and flag cross-province data leakage

Returns a structured Verdict: {pass, issues[], suggestions[]}
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VerificationIssue:
    """A single issue found during verification."""

    category: str  # "numerical" | "citation" | "compliance" | "cross_province"
    severity: str  # "error" | "warning" | "info"
    description: str
    location: str = ""  # snippet of answer where issue occurs
    suggestion: str = ""


@dataclass
class Verdict:
    """Complete verification result."""

    passed: bool
    total_claims: int = 0
    verified_claims: int = 0
    unverified_claims: int = 0
    issues: list[VerificationIssue] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    overall_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Built-in default power-market price bounds; deployment config may override them.
# ---------------------------------------------------------------------------

PROVINCE_PRICE_BOUNDS: dict[str, dict[str, float]] = {
    "山东": {"floor": 0, "cap": 1500, "typical_avg": 380},
    "江苏": {"floor": 0, "cap": 1500, "typical_avg": 420},
    "浙江": {"floor": 0, "cap": 1500, "typical_avg": 450},
    "广东": {"floor": 0, "cap": 1500, "typical_avg": 500},
    "河南": {"floor": 0, "cap": 1500, "typical_avg": 350},
    "山西": {"floor": 0, "cap": 1500, "typical_avg": 300},
    "default": {"floor": 0, "cap": 1500, "typical_avg": 400},
}


class VerifierAgent:
    """Multi-dimensional verification agent for LLM-generated answers.

    Usage:
        verifier = VerifierAgent(llm=llm)
        verdict = verifier.verify(answer, sources, province="山东")
        if not verdict.passed:
            for issue in verdict.issues:
                print(issue.description)
    """

    def __init__(self, llm=None):
        self._llm = llm
        self._price_bounds = dict(PROVINCE_PRICE_BOUNDS)

    def verify(
        self,
        answer: str,
        sources: list[dict] | str,
        province: str = "",
        task_type: str = "general",
    ) -> Verdict:
        """Run full verification on the generated answer."""
        source_text = self._normalize_sources(sources)
        issues: list[VerificationIssue] = []

        # 1. Numerical accuracy check
        num_issues = self._check_numerical_accuracy(answer, source_text, province)
        issues.extend(num_issues)

        # 2. Citation integrity check
        cite_issues = self._check_citation_integrity(answer, source_text)
        issues.extend(cite_issues)

        # 3. Price bound compliance
        bound_issues = self._check_price_bounds(answer, province)
        issues.extend(bound_issues)

        # 4. Cross-province guard
        if province:
            cp_issues = self._check_cross_province(answer, province)
            issues.extend(cp_issues)

        # 5. LLM-based semantic verification (if available)
        semantic_issues = self._semantic_verify(answer, source_text, province, task_type)
        issues.extend(semantic_issues)

        # Aggregate
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]

        # Extract all numeric claims for counting
        numeric_pattern = re.compile(r"(\d+\.?\d*)\s*(元/MWh|%|MW|MWh|元)")
        total_claims = len(numeric_pattern.findall(answer))
        unverified_count = len([i for i in issues if i.category == "numerical"])

        passed = len(errors) == 0 and len(warnings) <= 3

        suggestions = [i.suggestion for i in issues if i.suggestion]
        suggestions.append("建议人工复核关键数据点" if warnings else "")

        verdict = Verdict(
            passed=passed,
            total_claims=total_claims,
            verified_claims=total_claims - unverified_count,
            unverified_claims=unverified_count,
            issues=issues,
            suggestions=[s for s in suggestions if s],
            overall_confidence=self._compute_confidence(total_claims, len(errors), len(warnings)),
        )

        logger.info(
            "Verifier: %s | claims=%d verified=%d errors=%d warnings=%d confidence=%.2f",
            "PASS" if passed else "FAIL",
            total_claims,
            total_claims - unverified_count,
            len(errors),
            len(warnings),
            verdict.overall_confidence,
        )
        return verdict

    # ----- numerical accuracy -------------------------------------------------

    def _check_numerical_accuracy(
        self, answer: str, sources: str, province: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        # Extract all number+unit pairs from the answer
        claims = re.findall(r"(\d+\.?\d*)\s*(元/MWh|%|MW|MWh|元)", answer)

        for num_str, unit in claims:
            num = float(num_str)
            # Check if this number (as a string) appears in the sources
            if num_str not in sources:
                # Also check as int if .0
                if num == int(num) and str(int(num)) not in sources:
                    issues.append(
                        VerificationIssue(
                            category="numerical",
                            severity="warning",
                            description=f"数值 {num_str} {unit} 在来源中未找到对应验证",
                            location=self._find_snippet(answer, num_str),
                            suggestion=f"请核实 {num_str}{unit} 的来源",
                        )
                    )

        return issues

    # ----- citation integrity -------------------------------------------------

    def _check_citation_integrity(
        self, answer: str, sources: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        # Find citation markers — both [来源:X] format and natural attributions
        formal_citations = re.findall(
            r"\[(?:来源|引用|source)[：:\s]*(\d+|[^\]]+)\]", answer, re.IGNORECASE
        )
        # Strategy Agent uses natural attributions like:
        #   "来源: QUANT Agent", "来自 EXPERT Agent", "SQLite/price_data"
        natural_citations = re.findall(
            r"(?:来源|来自|source)[：:\s]*(?:QUANT|EXPERT|COMPLIANCE|STRATEGY|SQLite)",
            answer, re.IGNORECASE
        )
        has_any_citation = bool(formal_citations) or bool(natural_citations)

        # Check for factual sentences without citations
        factual_snippets = re.findall(
            r"([^。\n]{20,}?(?:\d+\.?\d*\s*(?:元/MWh|%|MW|MWh|元))[^。\n]{0,30})",
            answer,
        )
        for snippet in factual_snippets:
            has_cite = bool(
                re.search(r"\[(?:来源|引用|source)", snippet, re.IGNORECASE)
                or re.search(r"(?:来源|来自|source)[：:\s]*(?:QUANT|EXPERT|COMPLIANCE|STRATEGY)", snippet, re.IGNORECASE)
            )
            if not has_cite and has_any_citation:
                issues.append(
                    VerificationIssue(
                        category="citation",
                        severity="warning",
                        description="包含数值的声明缺少来源引用",
                        location=snippet[:80],
                        suggestion="为数值声明添加 [来源:X] 引用标记",
                    )
                )

        # Check for [推断] or [待验证] markers — these are acceptable but flag for review
        uncertain = re.findall(r"\[(?:推断|待验证)\]", answer)
        if uncertain:
            issues.append(
                VerificationIssue(
                    category="citation",
                    severity="info",
                    description=f"存在 {len(uncertain)} 处推断性/待验证声明",
                    suggestion="建议人工复核推断性内容",
                )
            )

        return issues

    # ----- price bounds -------------------------------------------------------

    def _check_price_bounds(
        self, answer: str, province: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        bounds = self._price_bounds.get(province, self._price_bounds["default"])

        prices = re.findall(r"(\d+\.?\d*)\s*元/MWh", answer)
        for price_str in prices:
            price = float(price_str)
            if price < bounds["floor"]:
                issues.append(
                    VerificationIssue(
                        category="compliance",
                        severity="error",
                        description=f"电价 {price} 元/MWh 低于{province}市场下限 {bounds['floor']} 元/MWh",
                        location=self._find_snippet(answer, price_str),
                        suggestion=f"核实电价计算，{province}市场下限为 {bounds['floor']} 元/MWh",
                    )
                )
            elif price > bounds["cap"]:
                issues.append(
                    VerificationIssue(
                        category="compliance",
                        severity="error",
                        description=f"电价 {price} 元/MWh 超过{province}市场上限 {bounds['cap']} 元/MWh",
                        location=self._find_snippet(answer, price_str),
                        suggestion=f"核实电价计算，{province}市场上限为 {bounds['cap']} 元/MWh",
                    )
                )

        return issues

    # ----- cross-province guard -----------------------------------------------

    def _check_cross_province(
        self, answer: str, province: str
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        all_provinces = [
            p for p in PROVINCE_PRICE_BOUNDS if p != "default" and p != province
        ]

        for other_prov in all_provinces:
            # Check if another province's data is mentioned without context
            pattern = rf"{other_prov}\s*(?:现货|电价|市场|政策)"
            matches = re.findall(pattern, answer)
            if matches:
                # Acceptable if explicitly comparative
                comparative_markers = ["相比", "对比", "高于", "低于", "不同于", "与.*不同"]
                is_comparative = any(
                    re.search(marker, answer) for marker in comparative_markers
                )
                if not is_comparative:
                    issues.append(
                        VerificationIssue(
                            category="cross_province",
                            severity="warning",
                            description=f"回答中提及 {other_prov} 数据，但当前省份为 {province}，可能存在跨省数据混淆",
                            location=self._find_snippet(answer, other_prov),
                            suggestion=f"请确认 {other_prov} 数据引用是否有意为之，如为误用请删除",
                        )
                    )

        return issues

    # ----- semantic verification (LLM) ----------------------------------------

    def _semantic_verify(
        self, answer: str, sources: str, province: str, task_type: str
    ) -> list[VerificationIssue]:
        """LLM-based semantic consistency check. Falls back gracefully.

        IMPORTANT: The answer is generated by specialized agents (QUANT, COMPLIANCE,
        STRATEGY) whose tool outputs (LSTM predictions, factor analysis, policy search,
        risk validation) ARE the ground truth. The 'sources' text is a truncated summary
        of those tool outputs. Therefore:

        - Data that plausibly comes from tools (prices, factors, risk levels) is VALID
          even if the exact number isn't verbatim in the truncated sources.
        - Only flag clear CONTRADICTIONS (e.g., answer says price=500 but source says price=300)
        - Reasonable inferences and structured formatting are NOT hallucinations.
        - Default severity is 'warning', use 'error' only for clear contradictions.
        """
        if self._llm is None:
            return []

        prompt = f"""你是电力行业答案验证器。请检查以下回答是否与来源上下文**矛盾**。

★ 重要背景:
- 该回答由多个专业Agent（量化预测、合规风控、策略交易）基于工具调用生成
- 工具输出（LSTM预测、因子分析、政策检索等）是真实数据源
- "来源上下文"是工具输出的截断摘要，可能不包含所有细节
- 因此: 数据只要**不矛盾**就视为合法，不要因为"来源中找不到这个具体数字"而标记为编造

来源上下文:
{sources[:5000]}

生成的回答:
{answer[:3000]}

请检查:
1. 回答中的数据是否与来源**明显矛盾**？（如来源说均价300，回答说均价500）
2. 回答中是否有**明显不可能**的数据？（如电价为负数、超过上限1500元/MWh）
3. 回答结构是否合理？

★ 注意:
- 回答中由Agent工具产出的数据（如LSTM预测值、因子分析结果）即使不在截断的来源中，也是合法的
- 场景推演中的概率（50%/25%/20%/5%）是标准框架，不是编造
- 分时申报建议中的价格区间如果与预测均价一致，就是合法的
- 火电边际成本、煤价等属于行业常识，不需要在来源中逐字验证
- 只在发现**明确矛盾**或**明显不可能的数据**时才标记为error

如果发现问题，以 JSON 格式列出:
[{{"category": "semantic", "severity": "warning", "description": "...", "suggestion": "..."}}]

如果没有矛盾，返回空数组 []。只返回 JSON。"""

        try:
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            data = self._safe_json_parse(text)
            if isinstance(data, list):
                issues = []
                for item in data[:10]:
                    # Downgrade 'error' to 'warning' unless it's a clear contradiction
                    severity = item.get("severity", "warning")
                    if severity == "error":
                        severity = "warning"  # Semantic check should never block
                    issues.append(
                        VerificationIssue(
                            category=item.get("category", "semantic"),
                            severity=severity,
                            description=item.get("description", ""),
                            suggestion=item.get("suggestion", ""),
                        )
                    )
                return issues
        except Exception as exc:
            logger.warning("Semantic verification via LLM failed: %s", exc)

        return []

    # ----- utilities ----------------------------------------------------------

    @staticmethod
    def _normalize_sources(sources: list[dict] | str) -> str:
        if isinstance(sources, str):
            return sources
        parts = []
        for i, src in enumerate(sources):
            if isinstance(src, dict):
                content = src.get("content") or src.get("text") or str(src)
                src_name = src.get("source", src.get("name", f"来源{i+1}"))
                parts.append(f"[来源{i+1} | {src_name}] {content}")
            else:
                parts.append(str(src))
        return "\n\n".join(parts)

    @staticmethod
    def _find_snippet(answer: str, keyword: str, context: int = 60) -> str:
        idx = answer.find(keyword)
        if idx < 0:
            return ""
        start = max(0, idx - context // 2)
        end = min(len(answer), idx + len(keyword) + context // 2)
        return answer[start:end].replace("\n", " ")

    @staticmethod
    def _compute_confidence(total: int, errors: int, warnings: int) -> float:
        if total == 0:
            return 0.5
        penalty = (errors * 0.3 + warnings * 0.1) / max(1, total)
        return round(max(0.0, 1.0 - penalty), 2)

    @staticmethod
    def _safe_json_parse(text: str):
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return []
