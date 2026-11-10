#!/usr/bin/env python3
"""
售电交易决策表生成器 — 聚合所有上游 Agent 输出，生成结构化申报决策表。

这不是"自动申报"——这是给交易员的决策辅助表。
交易员审核后，手动在交易中心系统完成申报。

输出:
  1. 日前申报决策表 (Markdown)
  2. 实时交易建议
  3. 中长期锁价策略
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool


# ═══════════════════════════════════════════════════════════════════
# 场景一: 日前申报决策表
# ═══════════════════════════════════════════════════════════════════

@tool
def generate_bidding_sheet(
    province: str,
    target_date: str,
    customer_id: str = "",
    risk_preference: str = "balanced"
) -> str:
    """
    生成日前申报决策表。聚合电价预测 + 因子分析 + 合规边界 + 历史经验 + 风险矩阵，
    输出结构化的24点申报建议。

    ★ 模板来源: prompts/templates/bidding_sheet.md (权威输出格式)

    这不是自动申报——这是给交易员的决策辅助表。

    Args:
        province: 省份名称
        target_date: 目标日期 YYYY-MM-DD
        customer_id: 客户ID (可选)
        risk_preference: 风险偏好 (aggressive/balanced/conservative)

    Returns:
        完整的 Markdown 格式申报决策表 (从模板文件加载)
    """
    # ★ 从权威模板文件加载输出格式
    template_path = PROJECT_ROOT / "prompts" / "templates" / "bidding_sheet.md"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        # 回退: 简化版模板
        template = """# 📝 日前申报决策表 · {province} · {target_date}

> 生成时间: {timestamp} | 风险偏好: {risk_preference}
> ⚠️ **本决策表为AI辅助建议，请交易员审核确认后在交易中心系统手动申报。**

---

## 一、决策总览
| 项目 | 建议值 | 置信度 | 关键依据 |
|------|--------|:---:|------|
| **申报均价** | **{bid_avg} 元/MWh** | {confidence} | {adjustment_reason} |
| **峰段策略** | {peak_strategy} | {peak_confidence} | {peak_reason} |
| **谷段策略** | {valley_strategy} | {valley_confidence} | {valley_reason} |

## 二、24点分时申报建议
| 时段 | 小时 | 日前出清预测 (元/MWh) | 建议申报价 (元/MWh) | 策略 | 理由 |
|------|------|:------:|:------:|:----:|------|
| 深谷 | 00:00-06:00 | {v_price} | {v_bid} | {v_strategy} | {v_reason} |
| 平段 | 06:00-10:00 | {f_price_1} | {f_bid_1} | {f_strategy_1} | {f_reason_1} |
| 峰段 | 10:00-15:00 | {p_price_1} | {p_bid_1} | {p_strategy_1} | {p_reason_1} |
| 平段 | 15:00-18:00 | {f_price_2} | {f_bid_2} | {f_strategy_2} | {f_reason_2} |
| 晚峰 | 18:00-21:00 | {p_price_2} | {p_bid_2} | {p_strategy_2} | {p_reason_2} |
| 平段 | 21:00-24:00 | {f_price_3} | {f_bid_3} | {f_strategy_3} | {f_reason_3} |

---
> ⚠️ **免责声明**: 本决策表仅供辅助决策参考，不构成交易指令。实际申报由交易员负责。"""

    # 填充已知变量
    filled = template.replace("{province}", province)
    filled = filled.replace("{target_date}", target_date)
    filled = filled.replace("{timestamp}", datetime.now().strftime('%Y-%m-%d %H:%M'))
    filled = filled.replace("{risk_preference}", risk_preference)

    # 添加引导说明: StrategyAgent 需要填充剩余变量
    header = filled + f"""

---
> 📌 **模板使用说明**: 以上模板中的 {{变量}} 需要由 STRATEGY Agent 基于上游 QUANT/EXPERT/COMPLIANCE 输出填充真实数据。
> Strategy Agent 会调用 run_multi_role_analysis / run_scenario_simulation / run_strategy_advisor 获取数据后填入。
"""
    return header


# ═══════════════════════════════════════════════════════════════════
# 场景二: 实时交易信号
# ═══════════════════════════════════════════════════════════════════

@tool
def generate_realtime_signal(
    province: str,
    current_price: float,
    day_ahead_price: float,
    position_mwh: float = 0,
    deviation_limit_pct: float = 5.0,
) -> str:
    """
    生成实时交易信号和建议。

    Args:
        province: 省份
        current_price: 当前实时电价 (元/MWh)
        day_ahead_price: 日前出清价 (元/MWh)
        position_mwh: 当前持仓 (MWh)
        deviation_limit_pct: 偏差免考核范围 (%)

    Returns:
        实时交易建议
    """
    deviation_pct = ((current_price - day_ahead_price) / day_ahead_price) * 100 if day_ahead_price else 0

    # 判断状态
    if abs(deviation_pct) <= deviation_limit_pct:
        status = "✅ 正常"
        action = "hold"
        advice = f"实时电价在正常偏差范围内 (±{deviation_limit_pct}%)，无需操作。"
    elif deviation_pct > deviation_limit_pct:
        status = "🟡 偏高"
        action = "sell"
        advice = f"实时电价高于日前 {deviation_pct:.1f}%，超出免考核范围。建议: 如有正偏差风险，考虑在实时市场卖出多余电量。"
    else:
        status = "🟢 偏低"
        action = "buy"
        advice = f"实时电价低于日前 {abs(deviation_pct):.1f}%，超出免考核范围。建议: 如有负偏差风险，考虑在实时市场补仓。"

    return f"""【实时交易信号 · {province} · {datetime.now().strftime('%H:%M')}】

当前实时电价: {current_price} 元/MWh
日前出清价: {day_ahead_price} 元/MWh
偏离幅度: {deviation_pct:+.1f}%
状态: {status}

建议操作: {advice}

持仓: {position_mwh} MWh
偏差免考核范围: ±{deviation_limit_pct}%

---
⚠️ 实时交易风险较高，请交易员根据实际持仓和风险承受能力决策。"""


# ═══════════════════════════════════════════════════════════════════
# 场景三: 中长期锁价策略
# ═══════════════════════════════════════════════════════════════════

@tool
def generate_long_term_strategy(
    province: str,
    current_holdings: str = "",
    forward_price_forecast: str = "",
    monthly_consumption_mwh: float = 5000,
) -> str:
    """
    生成中长期合约锁价策略建议。

    Args:
        province: 省份
        current_holdings: 当前持仓信息 (JSON 格式)
        forward_price_forecast: 远期价格预测
        monthly_consumption_mwh: 月用电量 (MWh)

    Returns:
        中长期锁价策略
    """
    return f"""【中长期锁价策略 · {province}】

## 当前持仓
{current_holdings or "待填入"}

## 远期预测
{forward_price_forecast or "待填入"}

## 锁价建议
- 月用电量: {monthly_consumption_mwh} MWh
- 建议锁价比例: 待策略分析
- 建议锁价价格: 待策略分析
- 滚动撮合策略: 待分析

## 操作日历
| 日期 | 撮合类型 | 建议操作 | 价格区间 |
|------|---------|---------|---------|
| 待填入 | - | - | - |

---
⚠️ 中长期策略基于远期价格预测，存在不确定性。请结合市场实际行情决策。"""


# ═══════════════════════════════════════════════════════════════════
# CLI 测试
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 日前申报决策表模板 ===")
    print(generate_bidding_sheet.invoke({"province": "山东", "target_date": "2026-06-16"}))

    print("\n=== 实时交易信号 ===")
    print(generate_realtime_signal.invoke({
        "province": "山东",
        "current_price": 420,
        "day_ahead_price": 385,
        "position_mwh": 1000,
    }))
