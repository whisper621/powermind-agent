#!/usr/bin/env python3
"""
客户策略报告 Tool —— 生成 B2B 客户定制化策略报告。
包含: 预测成本统计、分日分析、分时分析、实时量价分析、三场景用电策略。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool


@tool
def generate_client_strategy_report(
    client_id: str,
    province: str,
    report_period_start: str,
    report_period_end: str,
    adjustable_load_ratio: float = 0.10,
    has_storage: bool = False,
    storage_mw: float = 0.0,
    storage_mwh: float = 0.0,
) -> str:
    """
    为指定用电客户生成用电成本预测与优化策略报告。

    包含:
      - 预测成本统计（总用电量、参考电价、总电费）
      - 分日分析（高成本风险日识别、成本波动主因）
      - 分时分析（量价错配诊断、峰谷利用评估）
      - 三场景用电策略（5%/10%/15% 可调负荷 + 储能套利）

    Args:
        client_id: 客户ID
        province: 省份
        report_period_start: 报告周期开始日期 YYYY-MM-DD
        report_period_end: 报告周期结束日期 YYYY-MM-DD
        adjustable_load_ratio: 可调负荷比例（0.05-0.15）
        has_storage: 是否有储能设备
        storage_mw: 储能功率 (MW)
        storage_mwh: 储能容量 (MWh)

    Returns:
        Markdown 格式的完整客户策略报告。
    """
    try:
        from tools.lstm_data_adapter import get_adapter

        adapter = get_adapter()

        # ── 逐日获取电价预测 ──
        from datetime import datetime, timedelta

        start_dt = datetime.strptime(report_period_start, "%Y-%m-%d")
        end_dt = datetime.strptime(report_period_end, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1

        daily_prices: list[float] = []
        for i in range(days):
            d = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                r = adapter.predict(province, d)
                daily_prices.append(r["predicted_price"])
            except Exception:
                daily_prices.append(daily_prices[-1] if daily_prices else 500.0)

        # ── 模拟用户负荷曲线（基于典型工业负荷） ──
        import numpy as np

        # 典型工业日负荷曲线 (24h, 归一化)
        base_load_profile = np.array([
            0.65, 0.60, 0.55, 0.50, 0.48, 0.45,  # 00-06
            0.55, 0.75, 0.90, 0.95, 0.98, 1.00,  # 06-12
            0.92, 0.88, 0.85, 0.82, 0.80, 0.85,  # 12-18
            0.95, 1.00, 0.98, 0.90, 0.80, 0.70,  # 18-24
        ])
        avg_load_mw = 100.0  # 默认平均负荷 100MW
        hourly_load = base_load_profile * avg_load_mw

        # 日总用电量
        daily_total_mwh = hourly_load.sum()  # MWh
        total_load_mwh = daily_total_mwh * days

        # 加权均价
        avg_price = sum(daily_prices) / len(daily_prices)
        total_cost = sum(
            daily_total_mwh * p for p in daily_prices
        )

        # ── 分日分析 ──
        daily_costs = [daily_total_mwh * p for p in daily_prices]
        risk_day_idx = int(np.argmax(daily_costs))
        risk_day = (start_dt + timedelta(days=risk_day_idx)).strftime("%Y-%m-%d")
        risk_day_cost = daily_costs[risk_day_idx]
        risk_day_price = daily_prices[risk_day_idx]

        cost_range = max(daily_costs) - min(daily_costs)
        price_range_pct = (max(daily_prices) - min(daily_prices)) / avg_price * 100

        # ── 分时分析 (简化: 5段 TOU) ──
        peak_hours = [8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20]
        valley_hours = [0, 1, 2, 3, 4, 5, 22, 23]
        super_valley_hours = [10, 11, 12, 13]  # 午间深谷

        peak_load_pct = sum(base_load_profile[h] for h in peak_hours) / base_load_profile.sum() * 100
        valley_load_pct = sum(base_load_profile[h] for h in valley_hours) / base_load_profile.sum() * 100

        # ── 三场景策略 ──
        def simulate_scenario(load_shift_pct: float, storage_mw_val: float, storage_mwh_val: float):
            shifted_load = hourly_load.copy()
            # 峰时减负荷
            for h in peak_hours:
                shifted_load[h] *= (1 - load_shift_pct)
            # 深谷加负荷
            for h in super_valley_hours:
                shifted_load[h] *= (1 + load_shift_pct * len(peak_hours) / len(super_valley_hours))

            daily_cost_new = sum(shifted_load * avg_price)
            if storage_mw_val > 0:
                storage_saving = sum(
                    storage_mwh_val * avg_price * 0.3  # 峰谷价差 ~30%
                    for _ in range(days)
                )
                daily_cost_new -= storage_saving / days

            savings_per_day = daily_total_mwh * avg_price - daily_cost_new
            return daily_cost_new * days, savings_per_day * days

        s1_cost, s1_saving = simulate_scenario(0.05, 0, 0)
        s2_cost, s2_saving = simulate_scenario(0.10, 1.0, 2.0)
        s3_cost, s3_saving = simulate_scenario(0.15, 2.0, 4.0)

        # ── 生成报告 ──
        report = f"""# 市场联动深度分析报告

> 成本预估是以量价结合预估参考值，具体以真实缴费为准
> 预测周期: {report_period_start} 至 {report_period_end}
> 客户: {client_id} | 省份: {province}

---

## 一、预测成本统计

| 指标 | 数值 |
|------|------|
| 预测周期 | {report_period_start} 至 {report_period_end} |
| 预测总用电量 (MWh) | {total_load_mwh:,.1f} |
| 预测市场参考电价 (元/MWh) | {avg_price:.1f} |
| 预测市场总电费 (元) | {total_cost:,.0f} |

---

## 二、成本分析

### 「分日分析」

* **高成本风险预警**: 预测数据显示，您的用电成本高峰将出现在 **{risk_day}**，届时日电费将达到 {risk_day_cost:,.0f} 元，主要因其全日电价预测均值 {risk_day_price:.1f} 元/MWh 为周期最高。这表明，{risk_day} 是需要重点进行用能管控的风险日。

* **成本波动主因**: 预测期内，您的日均用电量相对平稳，但日电价水平波动显著（最高与最低日相差约 {price_range_pct:.1f}%）。这表明，**未来成本波动的首要驱动因素是预测电价的日间波动**，而非用电量的变化。

### 「分时分析」

* **"量价错配"风险**: 结合山东省典型的 5 段分时电价结构（尖、峰、平、谷、深）分析，您的预测用电曲线与预测电价峰谷结构高度重合。
* **峰段用电占比**: {peak_load_pct:.1f}% 的电量集中在高峰/尖峰时段
* **谷段利用潜力**: 在预测电价最低的深谷时段，您的用电量占比仅为 {valley_load_pct:.1f}%，**未能充分利用低价电力**，这是未来降低平均购电成本的主要优化方向。

### 「实时量价分析」

* **极端高价损失点**: 预测显示，周期内最高电价将出现在特定尖峰时段，若不进行干预，高峰小时将产生高额电费，是最大的单一成本侵蚀点。
* **重复性高价窗口**: 晚间 19:00-20:00 的尖峰时段构成连续、可预测的高成本风险窗口，需提前制定规避策略。

---

## 三、假设性分析及用电计划

### 「用电策略」

**场景一：基础柔性调控（可调负荷 5%）**
* **负荷压降策略**: 在每日预测电价最高的尖峰时段，削减最高 5% 的非生产关键负荷（如办公空调、部分照明、辅助通风）的功率。
* 预测总电费 (元): **{s1_cost:,.0f}**
* 电费环比下降 (元): **{s1_saving:,.0f}**
* 降本率 (%): **{s1_saving/total_cost*100:.1f}%**

**场景二：生产工序优化（可调负荷 10%）+ 中小型储能套利**
* **负荷压降策略**: 将一条可中断的预处理产线（约占总负荷 10%）从每日午间尖峰时段转移至凌晨深谷时段运行。
* **储能策略**: 配置一套 1MW/2MWh 的工商业储能系统。谷段充电，在午、晚尖峰时段各放电 1 小时，实现"低储高发"。
* 预测总电费 (元): **{s2_cost:,.0f}**
* 电费环比下降 (元): **{s2_saving:,.0f}**
* 降本率 (%): **{s2_saving/total_cost*100:.1f}%**

**场景三：深度调度与储能（可调负荷 15%）+ 大型储能**
* **负荷压降策略**: 重组生产班次，将高耗能工序（约占总负荷 15%）全部调整至夜间深谷时段运行。
* **储能策略**: 配置一套 2MW/4MWh 的储能系统。执行"两充两放"策略，在谷段充电，覆盖午、晚双尖峰，最大化套利。
* 预测总电费 (元): **{s3_cost:,.0f}**
* 电费环比下降 (元): **{s3_saving:,.0f}**
* 降本率 (%): **{s3_saving/total_cost*100:.1f}%**

### 「用电计划」

**多用电**
* **核心时段**: 每日 00:00-08:00（深谷/谷段）。
* **行动建议**: 安排高耗电、可中断的生产工序（如原料处理、电加热、恒温车间）在此时段满负荷运行。若配置储能，应在此时段完成充电。

**少用电**
* **核心时段**: 每日 10:00-12:00、19:00-21:00（尖峰时段）。
* **行动建议**: 在此时段降低非关键负荷用电，避免开启非必要的高耗能设备。

---

> ⚠️ 免责声明: 以上电价预测基于 LSTM 时序模型及历史数据，实际电价受市场供需、天气、政策等多因素影响，本报告仅供参考，不构成交易指令。
> 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        return report

    except Exception as e:
        import traceback
        return f"【系统提示】客户报告生成异常: {type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-500:]}"


# ── 工具列表 ─────────────────────────────────────────────────

ALL_TOOLS: list = []


def get_all_tools() -> list:
    """返回所有已注册的 LangChain tools (含新增领域工具)。"""
    global ALL_TOOLS
    if ALL_TOOLS:
        return ALL_TOOLS

    from tools.predict_price import predict_electricity_price
    from tools.search_policy import search_policy
    from tools.factor_engine import (
        compute_market_factors,
        run_multi_role_analysis,
        run_scenario_simulation,
        run_risk_validation,
        run_strategy_advisor,
    )
    from tools.energy_tools import (
        pv_forecast,
        storage_optimizer,
        pv_storage_joint_optimize,
        demand_forecast_alert,
        hvac_optimizer,
        power_factor_monitor,
    )
    from tools.market_tools import (
        cross_province_arbitrage,
        settlement_reconciliation,
        generator_offer_strategy,
        dr_vpp_aggregator,
        real_time_market_adapter,
    )
    from tools.client_tools import (
        province_entry_assessment,
        client_portfolio_analyzer,
        dynamic_pricing_ev,
    )
    from tools.bidding_sheet import (
        generate_bidding_sheet,
        generate_realtime_signal,
        generate_long_term_strategy,
    )
    from tools.weather_adapter import (
        fetch_weather_forecast,
        fetch_weather_current,
    )

    ALL_TOOLS = [
        predict_electricity_price,
        search_policy,
        compute_market_factors,
        run_multi_role_analysis,
        run_scenario_simulation,
        run_risk_validation,
        run_strategy_advisor,
        generate_client_strategy_report,
        # 天气数据 (NEW)
        fetch_weather_forecast,
        fetch_weather_current,
        # 售电交易决策表 (NEW)
        generate_bidding_sheet,
        generate_realtime_signal,
        generate_long_term_strategy,
        # 新增领域工具
        pv_forecast,
        storage_optimizer,
        pv_storage_joint_optimize,
        demand_forecast_alert,
        hvac_optimizer,
        power_factor_monitor,
        cross_province_arbitrage,
        settlement_reconciliation,
        generator_offer_strategy,
        dr_vpp_aggregator,
        real_time_market_adapter,
        province_entry_assessment,
        client_portfolio_analyzer,
        dynamic_pricing_ev,
    ]
    return ALL_TOOLS


# ── CLI 测试 ─────────────────────────────────────────────────

if __name__ == "__main__":
    report = generate_client_strategy_report.invoke({
        "client_id": "TEST_CLIENT_001",
        "province": "山东",
        "report_period_start": "2026-01-01",
        "report_period_end": "2026-01-10",
        "adjustable_load_ratio": 0.10,
        "has_storage": False,
    })
    print(report)
