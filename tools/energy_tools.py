#!/usr/bin/env python3
"""
Energy-domain tools — PV forecast, storage optimization, demand alerts, HVAC, power factor.

All tools are self-contained with rule-based/greedy optimization logic.
No heavy physics simulation — practical operational models suitable for
B2B advisory and daily dispatch decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool


# ═══════════════════════════════════════════════════════════════════
# S08 光伏出力预测
# ═══════════════════════════════════════════════════════════════════

# Typical solar radiation profiles by season (W/m², hourly)
_SOLAR_PROFILES = {
    "summer": [0, 0, 0, 0, 0, 10, 80, 200, 380, 550, 680, 780, 820, 800, 720, 580, 420, 250, 100, 30, 0, 0, 0, 0],
    "winter": [0, 0, 0, 0, 0, 0, 10, 80, 200, 350, 480, 560, 600, 580, 500, 380, 220, 80, 10, 0, 0, 0, 0, 0],
    "spring": [0, 0, 0, 0, 0, 20, 120, 280, 450, 600, 720, 780, 800, 760, 650, 500, 320, 150, 40, 0, 0, 0, 0, 0],
    "autumn": [0, 0, 0, 0, 0, 15, 100, 240, 400, 560, 680, 740, 760, 720, 620, 460, 290, 120, 30, 0, 0, 0, 0, 0],
}

# PV efficiency parameters
_DEFAULT_INVERTER_EFF = 0.96
_DEFAULT_TEMP_COEFF = -0.0035  # per °C above 25°C
_DEFAULT_TEMP_BASELINE = 25.0


@tool
def pv_forecast(
    province: str,
    target_date: str,
    capacity_kw: float = 1000.0,
    tilt_angle: float = 30.0,
    season: str = "auto",
) -> str:
    """
    预测光伏电站日出力曲线与发电量。

    基于典型太阳辐射曲线 + 装机容量 + 逆变器效率 + 温度修正。
    当用户询问"光伏能发多少电"、"光伏出力预测"、"分布式光伏收益"时调用。

    Args:
        province: 省份名称
        target_date: 日期 YYYY-MM-DD
        capacity_kw: 光伏装机容量 (kW)
        tilt_angle: 组件倾角 (度)，默认30°
        season: 季节 (summer/winter/spring/autumn/auto)，auto根据月份自动判断

    Returns:
        24点逐时光伏出力预测、日总发电量、峰值功率与时段。
    """
    try:
        # Auto-detect season from month
        if season == "auto":
            month = int(target_date[5:7])
            if month in (6, 7, 8):
                season = "summer"
            elif month in (12, 1, 2):
                season = "winter"
            elif month in (3, 4, 5):
                season = "spring"
            else:
                season = "autumn"

        base_profile = _SOLAR_PROFILES.get(season, _SOLAR_PROFILES["spring"])

        # Tilt adjustment (cosine factor, simplified)
        import math
        tilt_factor = math.cos(math.radians(max(0, abs(tilt_angle - 30)))) * 0.1 + 0.9

        # Temperature derating (simplified: summer +15°C above baseline)
        temp_offset = {"summer": 15, "spring": 5, "autumn": 5, "winter": -5}[season]
        temp_derate = 1 + _DEFAULT_TEMP_COEFF * temp_offset

        hourly_power = []
        for hour, rad in enumerate(base_profile):
            dc_power = capacity_kw * (rad / 1000.0) * tilt_factor * temp_derate
            ac_power = dc_power * _DEFAULT_INVERTER_EFF
            hourly_power.append(round(ac_power, 1))

        total_kwh = round(sum(hourly_power), 1)
        peak_power = max(hourly_power)
        peak_hour = hourly_power.index(peak_power)
        capacity_factor = total_kwh / (capacity_kw * 24) * 100

        lines = [
            f"【光伏出力预测】{province} {target_date}",
            f"  装机容量: {capacity_kw:.0f} kW  |  季节: {season}",
            f"  日总发电量: {total_kwh:,.0f} kWh",
            f"  峰值功率: {peak_power:.0f} kW (@ {peak_hour}:00)",
            f"  容量因子: {capacity_factor:.1f}%",
            f"  置信度: 75% (基于典型气象年辐射数据)",
            f"",
            f"  逐时出力 (kW):",
        ]
        for i in range(0, 24, 4):
            chunk = hourly_power[i:i + 4]
            labels = [f"{j:02d}:00" for j in range(i, min(i + 4, 24))]
            lines.append("  " + " | ".join(f"{l} {v:7.0f}" for l, v in zip(labels, chunk)))

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】光伏预测异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S07 储能优化调度
# ═══════════════════════════════════════════════════════════════════


@tool
def storage_optimizer(
    province: str,
    target_date: str,
    battery_capacity_mwh: float = 2.0,
    battery_power_mw: float = 1.0,
    roundtrip_efficiency: float = 0.90,
    price_forecast: str = "",
) -> str:
    """
    工商业储能日前优化调度（峰谷套利）。

    使用贪心算法: 在低价时段充电，高价时段放电，
    受 SOC 上下限 (10%-90%) 和充放电功率约束。

    当用户询问"储能怎么充放电"、"储能套利策略"、"电池调度"时调用。

    Args:
        province: 省份名称
        target_date: 日期 YYYY-MM-DD
        battery_capacity_mwh: 电池容量 (MWh)
        battery_power_mw: 额定功率 (MW)
        roundtrip_efficiency: 充放电效率 (0-1)
        price_forecast: 24点电价预测，逗号分隔 (元/MWh)，为空则使用默认分时电价

    Returns:
        24点充放电计划、SOC 轨迹、日总套利收益、等效循环次数。
    """
    try:
        # Parse or generate price forecast
        if price_forecast:
            prices = [float(x.strip()) for x in price_forecast.split(",")]
            if len(prices) != 24:
                return "【参数错误】price_forecast 需提供 24 个小时的电价值 (逗号分隔)"
        else:
            # Default TOU price profile (peak/valley/flat pattern)
            prices = [
                280, 260, 240, 220, 210, 200,  # 00-06 valley
                250, 350, 500, 600, 620, 580,  # 06-12 flat->peak
                450, 400, 380, 420, 500, 620,  # 12-18 flat->peak
                700, 800, 750, 650, 500, 380,  # 18-24 peak->valley
            ]

        # Sort hours by price (ascending for charge, descending for discharge)
        sorted_cheap = sorted(range(24), key=lambda h: prices[h])
        sorted_expensive = sorted(range(24), key=lambda h: prices[h], reverse=True)

        soc = 0.3 * battery_capacity_mwh  # start at 30%
        soc_min = 0.10 * battery_capacity_mwh
        soc_max = 0.90 * battery_capacity_mwh

        schedule = []
        hourly_soc = []
        total_revenue = 0.0

        for hour in range(24):
            action = "idle"
            power_mw = 0.0
            revenue = 0.0

            price_rank_cheap = sorted_cheap.index(hour)
            price_rank_expensive = sorted_expensive.index(hour)

            # Charge if price is in bottom 8 hours and SOC below max
            if price_rank_cheap < 8 and soc < soc_max:
                action = "charge"
                max_charge = min(battery_power_mw, (soc_max - soc) / roundtrip_efficiency)
                power_mw = min(max_charge, battery_power_mw)
                energy_in = power_mw * roundtrip_efficiency
                soc += energy_in
                cost = power_mw * prices[hour]
                revenue = -cost

            # Discharge if price is in top 6 hours and SOC above min
            elif price_rank_expensive < 6 and soc > soc_min:
                action = "discharge"
                max_discharge = min(battery_power_mw, (soc - soc_min) * roundtrip_efficiency)
                power_mw = min(max_discharge, battery_power_mw)
                energy_out = power_mw
                soc -= energy_out
                revenue = power_mw * prices[hour]

            total_revenue += revenue
            hourly_soc.append(round(soc, 2))
            schedule.append({
                "hour": hour,
                "action": action,
                "power_mw": round(power_mw, 2),
                "soc_after": round(soc / battery_capacity_mwh * 100, 1),
                "price": prices[hour],
                "revenue": round(revenue, 0),
            })

        # Calculate cycle count (sum of discharge energy / capacity)
        total_discharge = sum(s["power_mw"] for s in schedule if s["action"] == "discharge")
        cycle_count = round(total_discharge / battery_capacity_mwh, 2)

        avg_soc = sum(hourly_soc) / 24 / battery_capacity_mwh * 100

        lines = [
            f"【储能优化调度】{province} {target_date}",
            f"  电池配置: {battery_power_mw}MW / {battery_capacity_mwh}MWh (效率 {roundtrip_efficiency:.0%})",
            f"  日总套利收益: ¥{total_revenue:,.0f}",
            f"  等效循环: {cycle_count} 次  |  平均 SOC: {avg_soc:.1f}%",
            f"",
            "  时段  | 电价(元/MWh) | 动作     | 功率(MW) | SOC%  | 收益(元)",
            "  ------|-------------|----------|---------|-------|--------",
        ]
        for s in schedule:
            action_label = {"charge": "充电", "discharge": "放电", "idle": "待机"}[s["action"]]
            lines.append(
                f"  {s['hour']:02d}:00 | {s['price']:9.0f}   | {action_label:6s} | "
                f"{s['power_mw']:7.2f} | {s['soc_after']:4.0f}% | {s['revenue']:8.0f}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】储能优化异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S07 光储联合优化
# ═══════════════════════════════════════════════════════════════════


@tool
def pv_storage_joint_optimize(
    province: str,
    target_date: str,
    pv_capacity_kw: float = 1000.0,
    battery_capacity_mwh: float = 2.0,
    battery_power_mw: float = 1.0,
    load_profile_kw: str = "",
) -> str:
    """
    光储联合优化: 光伏出力优先自用，余电储能或上网，最大化收益。

    策略: PV出力 → 先供本地负荷 → 余电充储能 → 再余电上网
    储能优先在电价高峰放电供本地负荷，减少电网购电。

    当用户询问"光储一体怎么赚钱"、"光伏配储能收益"、"微电网调度"时调用。

    Args:
        province: 省份名称
        target_date: 日期 YYYY-MM-DD
        pv_capacity_kw: 光伏装机 (kW)
        battery_capacity_mwh: 储能容量 (MWh)
        battery_power_mw: 储能功率 (MW)
        load_profile_kw: 24点本地负荷 (kW)，逗号分隔，为空则使用典型工业负荷

    Returns:
        24点联合调度表、光伏自用率、余电上网量、总收益。
    """
    try:
        # Get PV forecast
        pv_result = pv_forecast.invoke({
            "province": province, "target_date": target_date,
            "capacity_kw": pv_capacity_kw,
        })
        # Extract hourly PV from the result (parse the table)
        pv_hourly = _extract_pv_hourly(pv_result, pv_capacity_kw)

        # Parse load profile
        if load_profile_kw:
            load = [float(x.strip()) for x in load_profile_kw.split(",")]
            if len(load) != 24:
                return "【参数错误】load_profile_kw 需提供 24 个小时的值 (逗号分隔)"
        else:
            # Typical industrial load (kW) scaled to ~50% of PV capacity
            base = [0.55, 0.50, 0.45, 0.42, 0.40, 0.38,
                    0.55, 0.70, 0.85, 0.90, 0.92, 0.95,
                    0.85, 0.80, 0.78, 0.75, 0.72, 0.78,
                    0.88, 0.95, 0.92, 0.85, 0.72, 0.60]
            load = [b * pv_capacity_kw * 0.6 for b in base]

        # Default TOU prices
        prices = [
            280, 260, 240, 220, 210, 200,
            250, 350, 500, 600, 620, 580,
            450, 400, 380, 420, 500, 620,
            700, 800, 750, 650, 500, 380,
        ]

        soc = 0.3 * battery_capacity_mwh * 1000  # convert to kWh
        soc_max = 0.90 * battery_capacity_mwh * 1000
        soc_min = 0.10 * battery_capacity_mwh * 1000
        eff = 0.90

        hourly_schedule = []
        total_self_consumed = 0.0
        total_grid_export = 0.0
        total_benefit = 0.0
        total_load = 0.0

        for h in range(24):
            pv_kw = pv_hourly[h]
            load_kw = load[h]
            total_load += load_kw

            # Step 1: PV serves local load directly
            self_consumed = min(pv_kw, load_kw)
            total_self_consumed += self_consumed
            pv_surplus = pv_kw - self_consumed
            net_load = load_kw - self_consumed  # remaining load after PV

            # Step 2: PV surplus -> charge battery
            charge_from_pv = 0.0
            if pv_surplus > 0 and soc < soc_max:
                charge_from_pv = min(pv_surplus, (soc_max - soc) / eff, battery_power_mw * 1000)
                soc += charge_from_pv * eff
                pv_surplus -= charge_from_pv

            # Step 3: Remaining PV surplus -> grid export
            grid_export = pv_surplus
            total_grid_export += grid_export

            # Step 4: Remaining net load -> discharge from battery (if price is high)
            discharge = 0.0
            grid_import = net_load
            if net_load > 0 and soc > soc_min:
                # Discharge during peak price hours (top 10)
                is_peak = prices[h] >= sorted(prices, reverse=True)[10]
                if is_peak:
                    discharge = min(net_load, (soc - soc_min) * eff, battery_power_mw * 1000)
                    soc -= discharge
                    grid_import = net_load - discharge

            # Economics
            export_revenue = grid_export * prices[h] / 1000  # kWh * 元/MWh / 1000
            import_cost = grid_import * prices[h] / 1000
            benefit = export_revenue - import_cost
            total_benefit += benefit

            hourly_schedule.append({
                "hour": h,
                "pv_kw": round(pv_kw, 0),
                "load_kw": round(load_kw, 0),
                "self_consumed_kw": round(self_consumed, 0),
                "battery_charge_kw": round(charge_from_pv, 0),
                "battery_discharge_kw": round(discharge, 0),
                "grid_export_kw": round(grid_export, 0),
                "grid_import_kw": round(grid_import, 0),
                "soc_pct": round(soc / (battery_capacity_mwh * 1000) * 100, 1),
                "benefit": round(benefit, 0),
            })

        self_consumption_rate = total_self_consumed / max(1, sum(pv_hourly)) * 100

        lines = [
            f"【光储联合优化】{province} {target_date}",
            f"  配置: PV {pv_capacity_kw:.0f}kW + 储能 {battery_power_mw}MW/{battery_capacity_mwh}MWh",
            f"  光伏自用率: {self_consumption_rate:.1f}%",
            f"  余电上网: {total_grid_export:,.0f} kWh",
            f"  日总收益: ¥{total_benefit:,.0f} (自用节省+上网收入-购电支出)",
            f"",
            "  Hour | PV(kW) | Load(kW) | Self | BatChg | BatDis | Export | Import | SOC% | Benefit",
            "  -----|--------|----------|------|--------|--------|--------|--------|------|--------",
        ]
        for s in hourly_schedule:
            lines.append(
                f"  {s['hour']:02d}:00 | {s['pv_kw']:6.0f} | {s['load_kw']:8.0f} | "
                f"{s['self_consumed_kw']:4.0f} | {s['battery_charge_kw']:6.0f} | "
                f"{s['battery_discharge_kw']:6.0f} | {s['grid_export_kw']:6.0f} | "
                f"{s['grid_import_kw']:6.0f} | {s['soc_pct']:4.0f}% | {s['benefit']:6.0f}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】光储联合优化异常: {type(e).__name__}: {str(e)[:300]}"


def _extract_pv_hourly(pv_result_str: str, capacity_kw: float) -> list[float]:
    """Parse hourly PV output from pv_forecast's text output."""
    import re
    hourly = []
    for line in pv_result_str.splitlines():
        # Match lines like "  00:00 0 | 01:00 0 | 02:00 0 | 03:00 0"
        nums = re.findall(r'\d{2}:\d{2}\s+([\d.]+)', line)
        if nums:
            hourly.extend(float(n) for n in nums)
    if len(hourly) == 24:
        return hourly
    # Fallback: use spring profile scaled to capacity
    profile = _SOLAR_PROFILES.get("spring", _SOLAR_PROFILES["spring"])
    return [capacity_kw * (r / 1000.0) * 0.96 for r in profile]


# ═══════════════════════════════════════════════════════════════════
# S09 需量预测与预警
# ═══════════════════════════════════════════════════════════════════


@tool
def demand_forecast_alert(
    province: str,
    client_id: str,
    target_date: str,
    contract_capacity_kw: float = 5000.0,
    historical_peak_kw: str = "",
) -> str:
    """
    工商业需量预测与越限预警。

    预测次日峰值负荷，与合同需量对比，计算裕度并给出预警等级。
    当用户询问"会不会超需量"、"需量预警"、"合同容量够不够"时调用。

    Args:
        province: 省份
        client_id: 客户ID
        target_date: 日期 YYYY-MM-DD
        contract_capacity_kw: 合同需量 (kW)
        historical_peak_kw: 近期峰值负荷，逗号分隔 (最近7天)，为空使用默认值

    Returns:
        预测峰值、裕度、预警等级、风险时段、储能预放电建议。
    """
    try:
        # Parse historical peaks
        if historical_peak_kw:
            peaks = [float(x.strip()) for x in historical_peak_kw.split(",")]
        else:
            # Default: contract at ~85% load factor
            peaks = [contract_capacity_kw * (0.75 + 0.05 * i) for i in range(7)]

        # Simple forecasting: weighted average (recent days heavier)
        weights = [0.05, 0.05, 0.08, 0.10, 0.15, 0.22, 0.35]
        if len(peaks) < 7:
            weights = [1.0 / len(peaks)] * len(peaks)
        else:
            weights = weights[-len(peaks):]

        forecast_peak = sum(p * w for p, w in zip(peaks, weights))

        # Add weekday/weekend adjustment
        month = int(target_date[5:7])
        temp_factor = 1.0
        if month in (6, 7, 8):
            temp_factor = 1.12  # summer cooling
        elif month in (12, 1, 2):
            temp_factor = 1.08  # winter heating

        forecast_peak *= temp_factor

        margin_kw = contract_capacity_kw - forecast_peak
        margin_pct = margin_kw / contract_capacity_kw * 100

        # Alert level
        if margin_pct < 0:
            alert_level = "critical"
            alert_label = "🔴 越限"
        elif margin_pct < 5:
            alert_level = "warning"
            alert_label = "🟡 预警"
        else:
            alert_level = "normal"
            alert_label = "🟢 正常"

        # Risk hours (peak TOU periods)
        risk_hours = [9, 10, 11, 14, 15, 16, 17, 18, 19, 20]

        # Pre-discharge recommendation
        if alert_level in ("critical", "warning"):
            pre_discharge = (
                f"建议在预测峰值时段 ({', '.join(str(h)+':00' for h in risk_hours[:5])} 等) "
                f"启动储能放电，削减峰值约 {min(abs(margin_kw), contract_capacity_kw * 0.15):.0f} kW。"
                f"如无储能，建议临时压减非关键负荷。"
            )
        else:
            pre_discharge = "无需特殊干预，按正常计划运行。"

        lines = [
            f"【需量预测与预警】{client_id} @ {province} {target_date}",
            f"  合同需量: {contract_capacity_kw:,.0f} kW",
            f"  预测峰值: {forecast_peak:,.0f} kW",
            f"  剩余裕度: {margin_kw:,.0f} kW ({margin_pct:.1f}%)",
            f"  预警等级: {alert_label}",
            f"",
            f"  风险时段: {', '.join(f'{h}:00' for h in risk_hours)}",
            f"  温度修正: {temp_factor:.0%} (季节因子)",
            f"",
            f"  💡 建议: {pre_discharge}",
        ]

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】需量预测异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S09 空调节能优化
# ═══════════════════════════════════════════════════════════════════


@tool
def hvac_optimizer(
    province: str,
    target_date: str,
    building_type: str = "office",
    floor_area_m2: float = 10000.0,
    comfort_range: str = "22-26",
) -> str:
    """
    中央空调系统节能优化（结合分时电价）。

    策略: 在低价时段预冷/预热，高价时段适当调高/调低设定温度，
    利用建筑热惰性减少高峰用电，同时控制舒适度违规。

    当用户询问"空调怎么省电"、"空调节能策略"、"暖通优化"时调用。

    Args:
        province: 省份
        target_date: 日期 YYYY-MM-DD
        building_type: 建筑类型 (office/commercial/industrial/warehouse)
        floor_area_m2: 建筑面积 (m²)
        comfort_range: 舒适温度范围 (°C)，格式 "min-max"

    Returns:
        24点优化温度设定、预计功率、逐时电费、节能率与舒适度违规统计。
    """
    try:
        comfort_lo, comfort_hi = [float(x) for x in comfort_range.split("-")]

        # Building thermal parameters
        building_params = {
            "office": {"load_w_per_m2": 80, "thermal_mass_h": 4},
            "commercial": {"load_w_per_m2": 120, "thermal_mass_h": 3},
            "industrial": {"load_w_per_m2": 100, "thermal_mass_h": 5},
            "warehouse": {"load_w_per_m2": 60, "thermal_mass_h": 6},
        }
        params = building_params.get(building_type, building_params["office"])
        base_load_kw = params["load_w_per_m2"] * floor_area_m2 / 1000

        # Default TOU prices
        prices = [
            280, 260, 240, 220, 210, 200,
            250, 350, 500, 600, 620, 580,
            450, 400, 380, 420, 500, 620,
            700, 800, 750, 650, 500, 380,
        ]

        # Strategy: pre-cool/heat during cheap hours, relax during expensive
        # Comfort range: 22-26°C, baseline 24°C
        baseline_temp = 24.0
        schedule = []
        baseline_cost = 0.0
        optimized_cost = 0.0
        comfort_violations = 0

        # HVAC COP (simplified)
        month = int(target_date[5:7])
        cop = 3.5 if month in (6, 7, 8) else 3.0  # cooling mode assumed

        for h in range(24):
            price = prices[h]

            # Baseline: constant 24°C, load proportional to outdoor temp delta
            # Simplified: base_load_kw at baseline temp
            baseline_power = base_load_kw
            baseline_cost += baseline_power * price / 1000

            # Optimized setpoint strategy
            if price < 300:  # valley price
                setpoint = comfort_lo  # pre-cool to lower bound
                load_factor = 1.15  # slightly higher load to store "coolth"
            elif price > 550:  # peak price
                setpoint = comfort_hi  # let temperature drift up
                load_factor = 0.5  # significantly reduced load
            else:  # flat price
                setpoint = baseline_temp
                load_factor = 0.85

            optimized_power = base_load_kw * load_factor / cop
            optimized_cost += optimized_power * price / 1000

            # Comfort check: if setpoint outside range
            if setpoint < comfort_lo or setpoint > comfort_hi:
                comfort_violations += 1

            schedule.append({
                "hour": h,
                "setpoint": round(setpoint, 1),
                "power_kw": round(optimized_power, 1),
                "cost": round(optimized_power * price / 1000, 0),
            })

        savings = baseline_cost - optimized_cost
        savings_pct = savings / max(1, baseline_cost) * 100

        lines = [
            f"【空调节能优化】{province} {target_date}",
            f"  建筑类型: {building_type}  |  面积: {floor_area_m2:,.0f} m²",
            f"  舒适范围: {comfort_lo}°C - {comfort_hi}°C  |  COP: {cop}",
            f"",
            f"  基准日电费: ¥{baseline_cost:,.0f}",
            f"  优化日电费: ¥{optimized_cost:,.0f}",
            f"  日节省: ¥{savings:,.0f} ({savings_pct:.1f}%)",
            f"  舒适度违规: {comfort_violations}h (设定温度超出舒适范围的小时数)",
            f"",
            "  Hour | 电价  | 设定°C | 功率kW | 电费(元)",
            "  -----|-------|--------|--------|---------",
        ]
        for s in schedule:
            lines.append(
                f"  {s['hour']:02d}:00 | {prices[s['hour']]:4.0f} | "
                f"{s['setpoint']:6.1f} | {s['power_kw']:6.1f} | {s['cost']:7.0f}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】空调节能优化异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S09 功率因数监测
# ═══════════════════════════════════════════════════════════════════


@tool
def power_factor_monitor(
    client_id: str,
    province: str,
    current_pf: float = 0.82,
    avg_active_power_kw: float = 5000.0,
    target_pf: float = 0.95,
    electricity_price: float = 0.6,
) -> str:
    """
    功率因数监测与无功补偿建议。

    计算功率因数偏低导致的力调电费罚款，给出电容补偿容量建议。
    当用户询问"功率因数不够"、"力调电费"、"无功补偿"时调用。

    Args:
        client_id: 客户ID
        province: 省份
        current_pf: 当前功率因数 (0-1)
        avg_active_power_kw: 平均有功功率 (kW)
        target_pf: 目标功率因数，默认 0.95
        electricity_price: 电度电价 (元/kWh)，默认 0.6

    Returns:
        当前/目标功率因数、需补偿容量 (kVar)、预估罚款、补偿建议。
    """
    try:
        import math

        # Validate
        current_pf = max(0.3, min(1.0, current_pf))
        target_pf = max(0.9, min(1.0, target_pf))

        # Calculate reactive power required
        # Q = P * (tan(acos(pf1)) - tan(acos(pf2)))
        angle_current = math.acos(current_pf)
        angle_target = math.acos(target_pf)
        tan_current = math.tan(angle_current)
        tan_target = math.tan(angle_target)
        required_kvar = avg_active_power_kw * (tan_current - tan_target)

        # Force adjustment fee estimation
        # Chinese regulation: PF < 0.90 => penalty on total bill
        # Typical: -0.5% to -2.5% per 0.01 below 0.90
        if current_pf >= 0.95:
            status = "ok"
            status_label = "🟢 合格"
            pf_shortfall = 0
            penalty_rate = 0
            recommendation = "功率因数良好，无需调整。"
        elif current_pf >= 0.90:
            status = "warning"
            status_label = "🟡 临界"
            pf_shortfall = 0.95 - current_pf
            penalty_rate = pf_shortfall * 100 * 0.005  # 0.5% per 0.01 below 0.95
            recommendation = f"建议增加 {required_kvar:,.0f} kVar 电容补偿，提升至 {target_pf}。"
        else:
            status = "penalty_risk"
            status_label = "🔴 罚款风险"
            pf_shortfall = 0.90 - current_pf
            penalty_rate = pf_shortfall * 100 * 0.01  # 1% per 0.01 below 0.90
            recommendation = (
                f"紧急: 需补偿 {required_kvar:,.0f} kVar 容性无功。"
                f"建议安装自动无功补偿装置 (SVG/电容器组)，"
                f"预计投资回收期 6-12 个月。"
            )

        # Monthly penalty estimation
        monthly_kwh = avg_active_power_kw * 24 * 30
        estimated_penalty = monthly_kwh * electricity_price * penalty_rate

        lines = [
            f"【功率因数监测】{client_id} @ {province}",
            f"  当前功率因数: {current_pf:.2f}  →  目标: {target_pf:.2f}",
            f"  状态: {status_label}",
            f"  平均有功: {avg_active_power_kw:,.0f} kW",
            f"  需补偿容量: {required_kvar:,.0f} kVar",
            f"  预估月力调罚款: ¥{estimated_penalty:,.0f} (费率 {penalty_rate:.2%})",
            f"",
            f"  💡 {recommendation}",
        ]

        if required_kvar > 100:
            # Recommend specific equipment
            steps = [50, 100, 200, 300, 500, 750, 1000]
            closest = min(steps, key=lambda s: abs(s - required_kvar / 25))
            lines.append(f"  推荐设备: {closest*25:.0f} kVar 自动电容器组 (分 {max(3, int(required_kvar/closest/25))} 路投切)")

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】功率因数监测异常: {type(e).__name__}: {str(e)[:300]}"
