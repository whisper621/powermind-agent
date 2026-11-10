#!/usr/bin/env python3
"""
Market-domain tools — cross-province arbitrage, settlement reconciliation,
generator offer strategy, DR/VPP aggregation.

Rule-based and optimization-driven, not physical simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool


# ═══════════════════════════════════════════════════════════════════
# S14 省间套利分析
# ═══════════════════════════════════════════════════════════════════

# Approximate transmission costs between provinces (元/MWh)
_TRANSMISSION_COSTS = {
    ("山东", "江苏"): 40, ("山东", "安徽"): 45, ("山东", "浙江"): 55,
    ("山东", "河南"): 30, ("山东", "河北"): 25, ("山西", "山东"): 35,
    ("山西", "江苏"): 60, ("山西", "河北"): 30, ("内蒙古", "山东"): 50,
    ("内蒙古", "江苏"): 70, ("内蒙古", "河北"): 40,
    ("江苏", "安徽"): 28, ("江苏", "浙江"): 35, ("江苏", "上海"): 25,
    ("广东", "广西"): 35, ("广东", "湖南"): 55, ("广东", "福建"): 45,
    ("四川", "重庆"): 20, ("四川", "湖北"): 50,
}

# Default average prices by province when no live data (元/MWh)
_DEFAULT_PROVINCE_PRICES = {
    "山东": 380, "江苏": 420, "山西": 320, "内蒙古": 280,
    "广东": 450, "浙江": 440, "河北": 360, "安徽": 390,
    "河南": 350, "湖北": 370, "湖南": 380, "四川": 300,
    "福建": 400, "广西": 370, "上海": 430, "重庆": 340,
}

# Major inter-province corridors
_MAJOR_CORRIDORS = [
    ("山西", "山东", "银东直流", 4000),
    ("内蒙古", "山东", "锡盟-山东交流", 6000),
    ("内蒙古", "江苏", "锡盟-泰州直流", 8000),
    ("四川", "江苏", "锦苏直流", 7200),
    ("四川", "浙江", "宾金直流", 8000),
    ("云南", "广东", "云广直流", 5000),
    ("山西", "江苏", "雁淮直流", 8000),
    ("宁夏", "山东", "宁东-山东直流", 4000),
    ("甘肃", "湖南", "酒湖直流", 8000),
    ("新疆", "安徽", "准皖直流", 10000),
]


@tool
def cross_province_arbitrage(
    source_province: str,
    target_province: str,
    target_date: str,
    source_avg_price: float = 0.0,
    target_avg_price: float = 0.0,
    trade_volume_mw: float = 100.0,
) -> str:
    """
    省间电力套利分析: 计算两省间价差、扣除输电费和线损后的净套利空间。

    当用户询问"从XX省买电卖到YY省能赚钱吗"、"省间价差"、"跨省交易机会"时调用。

    Args:
        source_province: 送端省份
        target_province: 受端省份
        target_date: 日期 YYYY-MM-DD
        source_avg_price: 送端日均价 (元/MWh)，为0则使用默认值
        target_avg_price: 受端日均价 (元/MWh)，为0则使用默认值
        trade_volume_mw: 交易电量 (MW)

    Returns:
        价差分析、输电费、线损、净价差、是否可行、预估日利润。
    """
    try:
        # Use provided prices or defaults
        src_price = source_avg_price if source_avg_price > 0 else _DEFAULT_PROVINCE_PRICES.get(source_province, 350)
        tgt_price = target_avg_price if target_avg_price > 0 else _DEFAULT_PROVINCE_PRICES.get(target_province, 400)

        price_spread = tgt_price - src_price

        # Transmission cost
        tx_cost = _TRANSMISSION_COSTS.get(
            (source_province, target_province),
            _TRANSMISSION_COSTS.get((target_province, source_province), 50),
        )

        # Line loss (longer distance = higher loss)
        # Simplified: 3% base + 0.5% per 500km
        loss_rate = 0.03

        # Net spread
        loss_cost = tgt_price * loss_rate
        net_spread = price_spread - tx_cost - loss_cost

        # Find corridor info
        corridor_info = ""
        corridor_capacity = 0
        for src, tgt, name, cap in _MAJOR_CORRIDORS:
            if (src == source_province and tgt in target_province) or \
               (tgt == target_province and src in source_province):
                corridor_info = f"{name} ({cap}MW)"
                corridor_capacity = cap
                break

        # Viability
        viable = net_spread > 10  # need at least 10 元/MWh margin
        daily_volume_mwh = trade_volume_mw * 24
        daily_profit = net_spread * daily_volume_mwh if viable else 0.0

        lines = [
            f"【省间套利分析】{source_province} → {target_province}  {target_date}",
            f"",
            f"  {source_province} 均价: {src_price:.0f} 元/MWh",
            f"  {target_province} 均价: {tgt_price:.0f} 元/MWh",
            f"  价差: {price_spread:+.0f} 元/MWh",
            f"  输电费: {tx_cost:.0f} 元/MWh",
            f"  线损率: {loss_rate:.1%} (损耗成本: {loss_cost:.0f} 元/MWh)",
            f"  净价差: {net_spread:+.0f} 元/MWh",
        ]

        if corridor_info:
            lines.append(f"  可用通道: {corridor_info}")

        lines.append(f"")
        lines.append(f"  套利可行: {'是 ✅' if viable else '否 ❌'}")
        lines.append(f"  交易电量: {trade_volume_mw}MW × 24h = {daily_volume_mwh:.0f} MWh/日")
        lines.append(f"  预估日利润: ¥{daily_profit:,.0f}")

        if not viable:
            reasons = []
            if price_spread <= 0:
                reasons.append(f"送端电价高于受端 ({src_price} > {tgt_price})")
            if tx_cost > price_spread:
                reasons.append(f"输电费 ({tx_cost}) 侵蚀全部价差")
            if loss_cost > max(0, price_spread - tx_cost):
                reasons.append(f"线损成本 ({loss_cost:.0f}) 过高")
            if reasons:
                lines.append(f"  原因: {'; '.join(reasons)}")

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】省间套利分析异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S18 结算复盘
# ═══════════════════════════════════════════════════════════════════


@tool
def settlement_reconciliation(
    province: str,
    client_id: str,
    period_start: str,
    period_end: str,
    total_energy_mwh: float = 0.0,
    predicted_avg_price: float = 0.0,
    actual_avg_price: float = 0.0,
    deviation_energy_mwh: float = 0.0,
    contracted_energy_mwh: float = 0.0,
) -> str:
    """
    电力市场结算复盘与偏差考核分析。

    对比预测电价 vs 实际电价，计算MAPE、偏差考核费用，
    评估策略有效性并给出改进建议。

    当用户询问"这个月结算怎么样"、"偏差考核多少"、"预测准不准"时调用。

    Args:
        province: 省份
        client_id: 客户ID
        period_start: 结算开始日期 YYYY-MM-DD
        period_end: 结算结束日期 YYYY-MM-DD
        total_energy_mwh: 总用电量 (MWh)
        predicted_avg_price: 预测均价 (元/MWh)
        actual_avg_price: 实际均价 (元/MWh)
        deviation_energy_mwh: 偏差电量 (MWh)
        contracted_energy_mwh: 合同电量 (MWh)

    Returns:
        结算总结、价格预测准确率、偏差考核费用、策略评级、改进建议。
    """
    try:
        # Price accuracy
        if predicted_avg_price > 0 and actual_avg_price > 0:
            price_mape = abs(predicted_avg_price - actual_avg_price) / actual_avg_price * 100
        else:
            predicted_avg_price = 380
            actual_avg_price = 395
            price_mape = abs(predicted_avg_price - actual_avg_price) / actual_avg_price * 100

        if total_energy_mwh <= 0:
            total_energy_mwh = 72000  # 100MW * 24h * 30d
        if contracted_energy_mwh <= 0:
            contracted_energy_mwh = total_energy_mwh * 0.85
        if deviation_energy_mwh <= 0:
            deviation_energy_mwh = abs(total_energy_mwh - contracted_energy_mwh)

        # Deviation rate
        deviation_rate = deviation_energy_mwh / max(1, contracted_energy_mwh) * 100

        # Deviation penalty (province-specific rules)
        province_dev_rules = {
            "山东": {"free_range": 5, "penalty_mult": 1.1},
            "江苏": {"free_range": 3, "penalty_mult": 1.2},
            "山西": {"free_range": 5, "penalty_mult": 1.0},
            "广东": {"free_range": 4, "penalty_mult": 1.15},
        }
        rules = province_dev_rules.get(province, {"free_range": 5, "penalty_mult": 1.1})
        free_range = rules["free_range"]
        penalty_mult = rules["penalty_mult"]

        penalty_energy = max(0, deviation_rate - free_range) / 100 * contracted_energy_mwh
        deviation_fee = penalty_energy * actual_avg_price * (penalty_mult - 1)

        # Total settlement
        total_settlement = total_energy_mwh * actual_avg_price + deviation_fee

        # Strategy effectiveness
        if price_mape < 5 and deviation_rate < free_range:
            effectiveness = "优秀 — 预测准确，偏差受控"
        elif price_mape < 10 and deviation_rate < free_range + 3:
            effectiveness = "良好 — 预测基本准确，偏差在可接受范围"
        elif price_mape < 15:
            effectiveness = "一般 — 预测/偏差有一项需改进"
        else:
            effectiveness = "待改进 — 预测偏差大，偏差考核风险高"

        lessons = []
        if price_mape > 10:
            lessons.append(f"价格预测 MAPE {price_mape:.1f}% 偏高，建议优化预测模型或增加多模型ensemble")
        if deviation_rate > free_range:
            lessons.append(f"偏差率 {deviation_rate:.1f}% 超出免考范围 ±{free_range}%，建议加强日前负荷预测精度")
        if deviation_fee > total_settlement * 0.02:
            lessons.append("偏差考核费用占比过高，建议调整中长期签约策略，适当增加现货敞口")

        if not lessons:
            lessons.append("本期结算表现良好，继续保持现有策略。")

        lines = [
            f"【结算复盘】{client_id} @ {province}",
            f"  结算周期: {period_start} ~ {period_end}",
            f"",
            f"  ── 电量结算 ──",
            f"  总用电量: {total_energy_mwh:,.0f} MWh",
            f"  合同电量: {contracted_energy_mwh:,.0f} MWh",
            f"  偏差电量: {deviation_energy_mwh:,.0f} MWh ({deviation_rate:.1f}%)",
            f"  免考范围: ±{free_range}%",
            f"",
            f"  ── 电价预测 ──",
            f"  预测均价: {predicted_avg_price:.0f} 元/MWh",
            f"  实际均价: {actual_avg_price:.0f} 元/MWh",
            f"  MAPE: {price_mape:.1f}%",
            f"",
            f"  ── 费用汇总 ──",
            f"  电能量费: ¥{total_energy_mwh * actual_avg_price:,.0f}",
            f"  偏差考核费: ¥{deviation_fee:,.0f}",
            f"  总结算: ¥{total_settlement:,.0f}",
            f"",
            f"  ── 策略评级 ──",
            f"  {effectiveness}",
            f"",
            f"  💡 改进建议:",
        ]
        for i, lesson in enumerate(lessons, 1):
            lines.append(f"  {i}. {lesson}")

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】结算复盘异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S21 发电侧报价策略
# ═══════════════════════════════════════════════════════════════════


@tool
def generator_offer_strategy(
    province: str,
    target_date: str,
    unit_capacity_mw: float = 300.0,
    marginal_cost: float = 280.0,
    forecast_mcp: float = 0.0,
    min_output_mw: float = 120.0,
    ramp_rate_mw_per_h: float = 60.0,
) -> str:
    """
    火电/燃气机组日前市场报价策略。

    基于边际成本 + 预测出清价，计算最优报价和申报量，
    平衡中标概率与发电利润。

    当用户询问"发电厂怎么报价"、"机组报价策略"、"日前申报"时调用。

    Args:
        province: 省份
        target_date: 日期 YYYY-MM-DD
        unit_capacity_mw: 机组容量 (MW)
        marginal_cost: 边际成本 (元/MWh)，含燃料+运维
        forecast_mcp: 预测出清价 (元/MWh)，为0则从默认价格估算
        min_output_mw: 最小技术出力 (MW)
        ramp_rate_mw_per_h: 爬坡率 (MW/h)

    Returns:
        建议报价、建议申报量、预期收益、弃电风险评估。
    """
    try:
        # Use forecast or default
        if forecast_mcp <= 0:
            mcp = _DEFAULT_PROVINCE_PRICES.get(province, 380) + 20
        else:
            mcp = forecast_mcp

        # Strategy logic
        if mcp < marginal_cost:
            # MCP below cost: bid at cost floor, may not clear
            strategy = "防御策略"
            offer_price = marginal_cost  # bid at cost, accept if cleared
            offer_volume = min_output_mw  # minimum output
            expected_revenue = offer_volume * mcp * 24
            curtailment_risk = "high"
            note = f"预测 MCP ({mcp:.0f}) 低于边际成本 ({marginal_cost:.0f})。建议报最低出力，或考虑停机备用的机会成本。"
        elif mcp > marginal_cost * 1.3:
            # Strong profit margin: bid competitively
            strategy = "进攻策略"
            # Bid slightly below forecast MCP to ensure clearing
            offer_price = mcp * 0.97
            offer_volume = unit_capacity_mw * 0.95  # 95% of capacity
            expected_revenue = offer_volume * (mcp - marginal_cost) * 24
            curtailment_risk = "low"
            note = f"利润空间充裕 (价差 {mcp - marginal_cost:.0f} 元/MWh)。建议接近满发报价，略低于预测 MCP 确保中标。"
        else:
            # Moderate: split capacity
            strategy = "稳健策略"
            # Bid part at marginal cost (base load), part at market price
            base_share = 0.6
            offer_price = marginal_cost + (mcp - marginal_cost) * 0.5
            offer_volume = unit_capacity_mw * 0.85
            expected_revenue = (
                offer_volume * base_share * (mcp - marginal_cost) * 24
                + offer_volume * (1 - base_share) * (mcp - offer_price) * 24
            )
            curtailment_risk = "medium"
            note = f"中等利润。建议分两段报价: {offer_volume * base_share:.0f}MW 报边际成本确保基础负荷，{(offer_volume * (1 - base_share)):.0f}MW 报更高价获取超额利润。"

        lines = [
            f"【发电报价策略】{province} {target_date}",
            f"  机组容量: {unit_capacity_mw:.0f} MW  |  最低出力: {min_output_mw:.0f} MW",
            f"  边际成本: {marginal_cost:.0f} 元/MWh  |  预测 MCP: {mcp:.0f} 元/MWh",
            f"  策略: {strategy}",
            f"",
            f"  建议申报价: {offer_price:.0f} 元/MWh",
            f"  建议申报量: {offer_volume:.0f} MW",
            f"  预期收益: ¥{expected_revenue:,.0f}/日",
            f"  弃电风险: {curtailment_risk}",
            f"",
            f"  📋 {note}",
        ]

        # Sensitivity
        lines.append(f"")
        lines.append(f"  敏感性分析 (MCP ±10%):")
        for pct in [-10, 0, 10]:
            mcp_sens = mcp * (1 + pct / 100)
            rev = offer_volume * max(0, mcp_sens - marginal_cost) * 24
            label = "悲观" if pct < 0 else ("乐观" if pct > 0 else "基准")
            lines.append(f"    {label} (MCP={mcp_sens:.0f}): ¥{rev:,.0f}")

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】发电报价异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S22 需求响应/VPP 聚合
# ═══════════════════════════════════════════════════════════════════


@tool
def dr_vpp_aggregator(
    province: str,
    event_date: str,
    assets: str = "",
    dr_incentive: float = 2.0,
) -> str:
    """
    虚拟电厂 (VPP) 需求响应聚合调度。

    汇总多元分布式资源 (储能、充电桩、空调、工业可中断负荷)，
    评估 DR 事件响应能力、承诺削减量和预期补偿收益。

    当用户询问"需求响应能赚多少"、"VPP怎么调度"、"DR事件响应"时调用。

    Args:
        province: 省份
        event_date: DR 事件日期 YYYY-MM-DD
        assets: 资产列表 JSON，格式 [{"name":"储能1","type":"storage","capacity_mw":2,"adjustable":true},...]
                为空则使用默认示例组合
        dr_incentive: DR 补偿标准 (元/kWh)，默认 2.0

    Returns:
        聚合可调负荷、基线负荷、承诺削减量、预期补偿、各资产调度分解。
    """
    try:
        import json

        # Parse or default assets
        if assets:
            asset_list = json.loads(assets)
        else:
            asset_list = [
                {"name": "工商业储能A", "type": "storage", "capacity_mw": 2.0, "adjustable": True},
                {"name": "工商业储能B", "type": "storage", "capacity_mw": 1.5, "adjustable": True},
                {"name": "充电桩集群", "type": "ev_charging", "capacity_mw": 3.0, "adjustable": True},
                {"name": "中央空调", "type": "hvac", "capacity_mw": 1.2, "adjustable": True},
                {"name": "可中断产线1", "type": "industrial", "capacity_mw": 5.0, "adjustable": True},
                {"name": "屋顶光伏", "type": "solar", "capacity_mw": 0.8, "adjustable": False},
                {"name": "办公楼宇", "type": "commercial", "capacity_mw": 2.5, "adjustable": True},
            ]

        total_capacity = sum(a["capacity_mw"] for a in asset_list)
        adjustable_capacity = sum(a["capacity_mw"] for a in asset_list if a.get("adjustable", False))
        non_adjustable = total_capacity - adjustable_capacity

        # Baseline: typical load at DR event hours (peak 10-12, 14-16, 18-20)
        dr_hours = [10, 11, 12, 14, 15, 16, 18, 19, 20]
        baseline_mw = total_capacity * 0.72  # average load factor across DR hours

        # Commitment: adjustable assets can reduce
        reduction_ratios = {
            "storage": 0.90,     # can discharge 90%
            "ev_charging": 0.70,  # can pause 70% of charging
            "hvac": 0.50,         # can reduce 50% of HVAC load
            "industrial": 0.60,   # can interrupt 60%
            "commercial": 0.40,   # can reduce 40% of commercial
        }

        asset_decomposition = []
        total_committed = 0.0

        for asset in asset_list:
            if not asset.get("adjustable", False):
                asset_decomposition.append({
                    "name": asset["name"],
                    "type": asset["type"],
                    "capacity_mw": asset["capacity_mw"],
                    "reduction_mw": 0,
                    "note": "不可调资源",
                })
                continue

            ratio = reduction_ratios.get(asset["type"], 0.50)
            reduction = asset["capacity_mw"] * ratio * 0.75  # 75% availability factor
            total_committed += reduction

            notes = {
                "storage": "放电支撑，SOC允许连续2h满放",
                "ev_charging": "暂停充电服务，优先保障V2G车辆",
                "hvac": "上调设定温度2°C，利用建筑热惰性",
                "industrial": "执行需求响应协议，中断非关键产线",
                "commercial": "降低公共区域照明与空调负荷",
            }

            asset_decomposition.append({
                "name": asset["name"],
                "type": asset["type"],
                "capacity_mw": asset["capacity_mw"],
                "reduction_mw": round(reduction, 2),
                "note": notes.get(asset["type"], "通用负荷削减"),
            })

        # DR compensation
        dr_duration_h = 3  # typical DR event 3 hours
        committed_energy_mwh = total_committed * dr_duration_h
        expected_compensation = committed_energy_mwh * 1000 * dr_incentive  # MWh->kWh * 元/kWh

        # Feasibility check
        feasible = total_committed > 1.0  # at least 1MW reduction

        lines = [
            f"【VPP需求响应调度】{province}  DR事件: {event_date}",
            f"  聚合总容量: {total_capacity:.1f} MW  |  可调容量: {adjustable_capacity:.1f} MW",
            f"  基线负荷 (DR时段): {baseline_mw:.1f} MW",
            f"  承诺削减: {total_committed:.2f} MW ({total_committed/baseline_mw*100:.1f}% 基线)",
            f"  DR补偿标准: ¥{dr_incentive}/kWh",
            f"  预期补偿: ¥{expected_compensation:,.0f} (约{dr_duration_h}h DR事件)",
            f"  参与可行: {'是 ✅' if feasible else '否 ❌ (削减量不足)'}",
            f"",
            f"  ── 资产调度分解 ──",
        ]

        for a in asset_decomposition:
            a_type = a["type"]
            a_type_label = {
                "storage": "🔋储能", "ev_charging": "🚗充电桩", "hvac": "❄️空调",
                "industrial": "🏭工业", "commercial": "🏢商业", "solar": "☀️光伏",
            }.get(a_type, a_type)
            lines.append(
                f"  {a_type_label} {a['name']}: {a['capacity_mw']}MW → "
                f"削减 {a['reduction_mw']}MW ({a['note']})"
            )

        lines.append(f"")
        lines.append(f"  ⚠️ 风险提示: 实际响应量受设备可用率、基线认定方式影响。建议预留10%安全裕度。")

        return "\n".join(lines)

    except json.JSONDecodeError:
        return "【参数错误】assets 格式不正确，请提供合法 JSON 数组。"
    except Exception as e:
        return f"【系统提示】VPP聚合异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S20 实时数据适配 (简化版)
# ═══════════════════════════════════════════════════════════════════


@tool
def real_time_market_adapter(
    province: str,
    data_type: str = "spot_price",
) -> str:
    """
    实时/准实时市场数据适配器。

    对接到电力交易中心 API 或数据库，返回最新市场数据快照。
    当前为框架实现，实际部署时替换为真实 API 调用。

    当用户询问"当前电价"、"实时价格"、"最新市场数据"时调用。

    Args:
        province: 省份
        data_type: 数据类型 (spot_price/load/RE_output/trade_volume/all)

    Returns:
        最新数据快照。
    """
    try:
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        base_price = _DEFAULT_PROVINCE_PRICES.get(province, 380)

        # Simulated real-time data
        data = {
            "timestamp": now,
            "province": province,
            "spot_price": round(base_price * (0.85 + 0.3 * (hash(province + now[:10]) % 100) / 100), 1),
            "load_mw": round(45000 * (0.9 + 0.2 * (hash(province + "load") % 100) / 100), 0),
            "re_output_mw": round(8000 * (0.5 + 1.0 * (hash(province + "re") % 100) / 100), 0),
            "trade_volume_mwh": round(12000 * (0.8 + 0.4 * (hash(province + "trade") % 100) / 100), 0),
        }

        lines = [f"【实时市场数据】{province}  {now}"]
        if data_type in ("spot_price", "all"):
            lines.append(f"  现货价格: {data['spot_price']:.1f} 元/MWh")
        if data_type in ("load", "all"):
            lines.append(f"  系统负荷: {data['load_mw']:.0f} MW")
        if data_type in ("RE_output", "all"):
            lines.append(f"  新能源出力: {data['re_output_mw']:.0f} MW")
        if data_type in ("trade_volume", "all"):
            lines.append(f"  成交量: {data['trade_volume_mwh']:.0f} MWh")

        lines.append(f"")
        lines.append(f"  ⚠️ 当前为模拟数据。接入真实交易中心API后替换。")

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】实时数据适配异常: {type(e).__name__}: {str(e)[:300]}"
