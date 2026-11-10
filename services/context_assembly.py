"""Context Assembly — builds structured per-agent context packages.

SQL precise data is kept separate from RRF-fused documents.
Facts, predictions, rules, experiences are never flattened into undifferentiated text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextPackage:
    """Structured context targeted at a specific component.

    Each professional agent receives a different ContextPackage.
    The final rendering to prompt text happens in one place.
    """

    target_component: str
    task_scope: dict = field(default_factory=dict)
    facts: list[dict] = field(default_factory=list)       # SQL data rows
    predictions: list[dict] = field(default_factory=list)  # Algorithm outputs
    rules: list[dict] = field(default_factory=list)        # Applicable regulations
    experiences: list[dict] = field(default_factory=list)  # Historical cases
    assumptions: list[dict] = field(default_factory=list)  # Explicit assumptions
    constraints: list[dict] = field(default_factory=list)  # Hard constraints
    evidence_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_prompt_text(self, max_facts: int = 20, max_rules: int = 10) -> str:
        """Convert to a prompt-ready text block at the LAST step."""
        parts: list[str] = []

        if self.facts:
            parts.append("## 数据事实 (来自SQL精确查询)")
            for f in self.facts[:max_facts]:
                parts.append(f"- {self._format_item(f)}")
            parts.append("")

        if self.predictions:
            parts.append("## 预测结果 (来自算法)")
            for p in self.predictions[:10]:
                parts.append(f"- {self._format_item(p)}")
            parts.append("")

        if self.rules:
            parts.append("## 适用规则 (来自政策文件)")
            for r in self.rules[:max_rules]:
                parts.append(f"- {self._format_item(r)}")
            parts.append("")

        if self.experiences:
            parts.append("## 历史经验 (来自已审核案例)")
            for e in self.experiences[:5]:
                parts.append(f"- {self._format_item(e)}")
            parts.append("")

        if self.assumptions:
            parts.append("## 明确假设")
            for a in self.assumptions:
                parts.append(f"- ⚠️ {self._format_item(a)}")
            parts.append("")

        if self.constraints:
            parts.append("## 硬约束")
            for c in self.constraints:
                parts.append(f"- 🚫 {self._format_item(c)}")
            parts.append("")

        if self.warnings:
            parts.append("## 注意事项")
            for w in self.warnings:
                parts.append(f"- ⚠️ {w}")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _format_item(item: dict) -> str:
        source = item.get("source", "")
        province = item.get("province", "")
        time_str = item.get("time", item.get("data_time", ""))
        value = item.get("value", item.get("content", str(item)))
        scope = item.get("scope", "")
        parts = [str(value)]
        if source:
            parts.append(f"[来源: {source}]")
        if province:
            parts.append(f"[省份: {province}]")
        if time_str:
            parts.append(f"[时间: {time_str}]")
        if scope:
            parts.append(f"[范围: {scope}]")
        return " ".join(parts)


class ContextAssembly:
    """Builds ContextPackages from retrieval results and agent outputs."""

    def build_for_quant(
        self,
        structured_data: list[dict],
        province: str,
        target_date: str,
    ) -> ContextPackage:
        return ContextPackage(
            target_component="quant",
            task_scope={"province": province, "target_date": target_date},
            facts=structured_data,
        )

    def build_for_compliance(
        self,
        rules: list[dict],
        province: str,
    ) -> ContextPackage:
        return ContextPackage(
            target_component="compliance",
            task_scope={"province": province},
            rules=rules,
        )

    def build_for_strategy(
        self,
        facts: list[dict],
        predictions: list[dict],
        rules: list[dict],
        experiences: list[dict],
        quant_result: dict | None = None,
        compliance_result: dict | None = None,
        province: str = "",
    ) -> ContextPackage:
        return ContextPackage(
            target_component="strategy",
            task_scope={"province": province},
            facts=facts,
            predictions=predictions,
            rules=rules,
            experiences=experiences,
            constraints=(
                compliance_result.get("required_changes", [])
                if compliance_result
                else []
            ),
            evidence_ids=(
                compliance_result.get("evidence_ids", [])
                if compliance_result
                else []
            ),
        )
