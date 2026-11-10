#!/usr/bin/env python3
"""
经验自动归档器 —— StrategyAgent 决策完成后自动生成经验文档。

流程:
  StrategyAgent 输出策略 → 交易员审核确认 → 系统自动归档决策过程
  → 生成 Markdown 经验文档 → 写入 data/source_docs/experiences/{省份}/
  → AutoImporter (下次启动时) 自动导入 ChromaDB

经验文档包含:
  - 决策日期、策略类型、市场特征
  - QUANT/EXPERT/COMPLIANCE 上游数据摘要
  - 最终策略建议
  - Reflection 质检结果
  - 交易员确认/修改记录 (可选)

设计原则:
  - 不阻塞 Agent 主流程 (静默失败)
  - 去重: 同一天同一类型的经验不重复写入
  - 人工可编辑: Markdown 格式，交易员可事后修改
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

logger = logging.getLogger(__name__)


class ExperienceArchiver:
    """StrategyAgent 决策完成后自动归档经验文档"""

    def __init__(self):
        self._target_dir = DATA_ROOT / "source_docs" / "experiences"

    def archive_decision(
        self,
        province: str,
        strategy_type: str,
        user_query: str,
        strategy_output: str,
        quant_summary: str = "",
        expert_summary: str = "",
        compliance_summary: str = "",
        reflection_score: Optional[float] = None,
        trader_feedback: str = "",
    ) -> Optional[str]:
        """
        归档一次完整的策略决策为经验文档。

        Args:
            province: 省份
            strategy_type: 策略类型 (日前申报/实时交易/中长期锁价)
            user_query: 原始用户问题
            strategy_output: 最终策略输出
            quant_summary: QUANT Agent 摘要
            expert_summary: EXPERT Agent 摘要
            compliance_summary: COMPLIANCE Agent 摘要
            reflection_score: Reflection 评分 (0-10)
            trader_feedback: 交易员审核反馈

        Returns:
            生成的文档路径，失败返回 None
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            output_dir = self._target_dir / province
            output_dir.mkdir(parents=True, exist_ok=True)

            # 去重: 同一天同类型不重复生成
            filename = f"{today}_{strategy_type}_决策复盘.md"
            filepath = output_dir / filename
            if filepath.exists():
                logger.info(f"[ExperienceArchiver] 跳过: {filepath} 已存在")
                return str(filepath)

            # 生成 Markdown 经验文档
            content = self._build_document(
                today=today,
                province=province,
                strategy_type=strategy_type,
                user_query=user_query,
                strategy_output=strategy_output,
                quant_summary=quant_summary,
                expert_summary=expert_summary,
                compliance_summary=compliance_summary,
                reflection_score=reflection_score,
                trader_feedback=trader_feedback,
            )

            filepath.write_text(content, encoding="utf-8")
            logger.info(f"[ExperienceArchiver] 经验已归档: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.warning(f"[ExperienceArchiver] 归档失败 (非致命): {e}")
            return None

    def _build_document(
        self,
        today: str,
        province: str,
        strategy_type: str,
        user_query: str,
        strategy_output: str,
        quant_summary: str,
        expert_summary: str,
        compliance_summary: str,
        reflection_score: Optional[float],
        trader_feedback: str,
    ) -> str:
        """构建经验文档 Markdown"""
        parts = [
            f"# {province} {strategy_type} 决策复盘",
            f"",
            f"- **日期**: {today}",
            f"- **省份**: {province}",
            f"- **策略类型**: {strategy_type}",
            f"- **Reflection评分**: {reflection_score if reflection_score is not None else 'N/A'}",
            f"",
            f"## 用户需求",
            f"{user_query[:500]}",
            f"",
        ]

        if quant_summary:
            parts.extend([
                f"## 上游量化预测 (QUANT Agent)",
                f"{quant_summary[:800]}",
                f"",
            ])

        if expert_summary:
            parts.extend([
                f"## 历史经验匹配 (EXPERT Agent)",
                f"{expert_summary[:500]}",
                f"",
            ])

        if compliance_summary:
            parts.extend([
                f"## 合规风控 (COMPLIANCE Agent)",
                f"{compliance_summary[:500]}",
                f"",
            ])

        parts.extend([
            f"## 策略决策",
            f"{strategy_output[:3000]}",
            f"",
        ])

        if trader_feedback:
            parts.extend([
                f"## 交易员审核反馈",
                f"{trader_feedback[:500]}",
                f"",
            ])

        parts.extend([
            f"---",
            f"*本文档由 PowerMind ExperienceArchiver 自动生成，",
            f"可作为交易员事后复盘参考。人工可编辑修改。*",
        ])

        return "\n".join(parts)

    def list_archived(self, province: str | None = None) -> list[dict]:
        """列出已归档的经验文档"""
        results = []
        search_dir = self._target_dir / province if province else self._target_dir
        if not search_dir.exists():
            return results

        for f in search_dir.rglob("*.md"):
            if f.is_file():
                results.append({
                    "filename": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
        return sorted(results, key=lambda x: x["modified"], reverse=True)


# ══ 单例 ══

_archiver: Optional[ExperienceArchiver] = None


def get_experience_archiver() -> ExperienceArchiver:
    global _archiver
    if _archiver is None:
        _archiver = ExperienceArchiver()
    return _archiver
