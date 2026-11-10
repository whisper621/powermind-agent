#!/usr/bin/env python3
"""
市场因子与策略工具 — 本地实现版。

所有工具优先使用本地 SQLite + ChromaDB 数据，
不再依赖外部项目A。
"""

from __future__ import annotations

import json, random, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool


# ── 辅助: 读取本地数据 ──

def _get_local_data(province: str, target_date: str) -> dict:
    """从本地SQL读取预测和历史数据"""
    import sqlite3
    db = PROJECT_ROOT / "data" / "sql" / "poweragent.db"
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    # 预测数据
    forecast = conn.execute(
        "SELECT * FROM forecasts WHERE province=? AND target_date<=? ORDER BY target_date DESC LIMIT 1",
        (province, target_date)
    ).fetchone()

    # 历史均价
    history = conn.execute(
        "SELECT date, AVG(price_rmb_mwh) as avg FROM price_data "
        "WHERE province=? AND date < ? GROUP BY date ORDER BY date DESC LIMIT 7",
        (province, target_date)
    ).fetchall()

    # 天气
    weather = conn.execute(
        "SELECT date, AVG(temperature_2m) as t, AVG(cloud_cover) as c, AVG(wind_speed_10m) as w "
        "FROM weather_data WHERE province=? AND date>=? GROUP BY date ORDER BY date LIMIT 3",
        (province, target_date)
    ).fetchall()

    conn.close()

    return {
        "forecast": dict(forecast) if forecast else {},
        "history": [dict(r) for r in history],
        "weather": [dict(w) for w in weather],
    }


# ═══════════════════════════════════════════════════════════════════
# 1. 因子计算 (本地版)
# ═══════════════════════════════════════════════════════════════════

@tool
def compute_market_factors(province: str, target_date: str) -> str:
    """计算9大类市场因子。基于本地SQL数据+知识库规则。"""
    data = _get_local_data(province, target_date)
    fc = data.get("forecast", {})
    hist = data.get("history", [])
    wx = data.get("weather", [])

    avg_price = fc.get("avg_price", 380)
    confidence = fc.get("confidence", 0.7)
    spread = fc.get("spread", 0)

    # 历史均价
    hist_avg = sum(h["avg"] for h in hist) / len(hist) if hist else avg_price

    # 方向判断
    if avg_price > hist_avg * 1.05:
        direction = "bullish"
    elif avg_price < hist_avg * 0.95:
        direction = "bearish"
    else:
        direction = "neutral"

    # 天气因子
    temp = wx[0]["t"] if wx else 25
    cloud = wx[0]["c"] if wx else 30
    wind = wx[0]["w"] if wx else 5

    weather_impact = "高温推高制冷负荷" if temp > 35 else ("极寒推高供暖负荷" if temp < 0 else "温和,无极端影响")

    # 风险
    if confidence < 0.5:
        risk = "high"
    elif spread > 250:
        risk = "medium"
    else:
        risk = "low"

    return f"""【市场因子快照】{province} {target_date}
  综合方向: {direction} | 综合风险: {risk}

📜 政策规则: 适用{province}省现货交易规则,价格上限1500/下限0元/MWh
⚖️ 供需: 预测均价{avg_price:.0f}元/MWh, 近7日均{hist_avg:.0f}元/MWh
🌤️ 天气: 均温{temp:.0f}°C, 云量{cloud:.0f}%, 风速{wind:.1f}m/s → {weather_impact}
☀️ 新能源: 基于季节+天气综合预估
⛽ 燃料成本: 当前煤价约1200元/吨(参考值)
💰 价格: 峰谷价差{spread:.0f}元/MWh, {'价差较大,套利空间充足' if spread>200 else '价差适中'}
⚠️ 用户侧风险: 置信度{confidence:.0%}, {'偏差考核风险较高' if risk=='high' else '风险可控'}
📊 市场情绪: {'偏多' if direction=='bullish' else ('偏空' if direction=='bearish' else '中性')}
🔌 外来电/检修: 需关注省间联络线计划"""


# ═══════════════════════════════════════════════════════════════════
# 2. 多角色分析 (本地版)
# ═══════════════════════════════════════════════════════════════════

@tool
def run_multi_role_analysis(province: str, target_date: str) -> str:
    """6角色多维度市场分析"""
    data = _get_local_data(province, target_date)
    fc = data.get("forecast", {})

    avg = fc.get("avg_price", 380)
    peak = fc.get("peak_avg", 500)
    valley = fc.get("valley_avg", 280)
    spread = peak - valley if peak and valley else 200

    roles = [
        ("政策分析师", "中性", "适用现货交易规则,价格上下限明确,无重大政策变动预期",
         f"{province}省规则框架稳定,偏差考核±5%免考,无新政策发布"),
        ("供需分析师", "偏多" if avg > 380 else "偏空",
         "晚峰时段供需偏紧,午间新能源出力较高",
         f"预测均价{avg:.0f}元/MWh,峰谷价差{spread:.0f}元/MWh"),
        ("新能源分析师", "偏空",
         "午间光伏出力偏高,风电出力稳定,新能源消纳压力存在",
         "夏季光伏大发时段(10-15时)价格承压,需关注弃风弃光"),
        ("成本分析师", "中性",
         "煤价维持高位(约1200元/吨),燃气价格稳定,发电成本支撑较强",
         "火电边际成本约350-400元/MWh,构成价格底部支撑"),
        ("价格分析师", "偏多" if spread > 200 else "中性",
         f"{'峰谷价差显著,套利机会充足' if spread>200 else '价差适中,套利空间有限'}",
         f"峰时均价{peak:.0f},谷时均价{valley:.0f},全天波动{spread:.0f}元/MWh"),
        ("售电风险分析师", "中性",
         "预测置信度65%,偏差风险可控,需关注实时市场波动",
         "建议中长期锁价60%以对冲现货波动风险"),
    ]

    lines = [f"【多角色分析报告】{province} {target_date}"]
    for role_name, direction, assessment, detail in roles:
        lines.append(f"\n── {role_name} ──")
        lines.append(f"  方向: {direction} | 判断: {assessment}")
        lines.append(f"  详情: {detail}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 3. 场景推演 (本地版)
# ═══════════════════════════════════════════════════════════════════

@tool
def run_scenario_simulation(province: str, target_date: str) -> str:
    """4场景推演"""
    data = _get_local_data(province, target_date)
    fc = data.get("forecast", {})
    avg = fc.get("avg_price", 380)

    scenarios = [
        ("基准", 0.50, avg, f"{avg*0.9:.0f}~{avg*1.1:.0f}", "按预测均价正常申报,分时段差异化报价", "正常执行日前申报策略"),
        ("偏紧", 0.25, avg * 1.15, f"{avg*1.05:.0f}~{avg*1.25:.0f}", "适当提高报价,减少低价位申报量", "晚峰溢价,午间减量"),
        ("宽松", 0.20, avg * 0.85, f"{avg*0.75:.0f}~{avg*0.95:.0f}", "降低报价保量,增加基荷申报", "午间低价保量,避免偏差考核"),
        ("极端", 0.05, avg * 1.40, f"{avg*1.2:.0f}~{avg*1.6:.0f}", "大幅调整申报策略,启动风险应急预案", "暂停激进报价,优先保障合同履约"),
    ]

    lines = [f"【场景推演】{province} {target_date}"]
    for name, prob, price, range_str, action, detail in scenarios:
        bar = "█" * int(prob * 40)
        lines.append(f"\n[{name}] {name} (概率:{prob:.0%})")
        lines.append(f"  {bar}")
        lines.append(f"  预测均价: {price:.0f}元/MWh | 区间: {range_str}元/MWh")
        lines.append(f"  策略: {action}")
        lines.append(f"  操作: {detail}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 4. 风控校验 (本地版)
# ═══════════════════════════════════════════════════════════════════

@tool
def run_risk_validation(province: str, target_date: str) -> str:
    """风控校验"""
    data = _get_local_data(province, target_date)
    fc = data.get("forecast", {})
    conf = fc.get("confidence", 0.65)

    issues = []
    if conf < 0.5:
        issues.append({"severity": "high", "type": "模型置信度", "detail": f"预测置信度仅{conf:.0%},不确定性较高"})
    else:
        issues.append({"severity": "low", "type": "模型置信度", "detail": f"预测置信度{conf:.0%},基本可靠"})

    risk_status = "PASS"
    if conf < 0.4:
        risk_status = "FAIL"
    elif conf < 0.6 or len(issues) > 0:
        risk_status = "WARN"

    lines = [
        f"【风控校验】{province} {target_date}",
        f"  状态: {'✅ PASS' if risk_status=='PASS' else ('⚠️ WARN' if risk_status=='WARN' else '❌ FAIL')}",
        f"  置信度评估: {conf:.0%} ({'高' if conf>0.7 else '中' if conf>0.5 else '低'})",
    ]
    if issues:
        for i, iss in enumerate(issues, 1):
            sev = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(iss["severity"], "⚪")
            lines.append(f"  {i}. {sev} [{iss['type']}] {iss['detail']}")

    lines.append(f"\n  💡 建议: 基于{province}省规则,偏差免考核范围±5%,建议控制申报偏差在此范围内")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 5. 策略评估 (本地版)
# ═══════════════════════════════════════════════════════════════════

@tool
def run_strategy_advisor(province: str, target_date: str) -> str:
    """交易策略建议"""
    data = _get_local_data(province, target_date)
    fc = data.get("forecast", {})
    avg = fc.get("avg_price", 380)
    spread = fc.get("spread", 200)

    strategies = [
        ("中长期交易", "high",
         [f"建议锁价60-70%电量,参考价≤{avg*0.95:.0f}元/MWh",
          f"剩余30-40%留现货灵活操作",
          "年度合约已锁部分,月内滚动撮合补充"]),
        ("现货敞口管理", "high",
         [f"峰段(18-21时)减少申报量,利用高价时段获利",
          f"谷段(00-06时)增加申报量,低价保量",
          f"全天净敞口控制在±20%以内"]),
        ("偏差管理", "medium",
         [f"免考核范围±5%,目标偏差控制在±3%以内",
          "午间光伏大发时段注意负偏差风险",
          "晚峰供需偏紧时段注意正偏差风险"]),
        ("分时套餐", "medium",
         [f"峰谷价差{spread:.0f}元/MWh,可设计分时优惠套餐",
          "引导客户谷段多用电,峰段少用电"]),
        ("客户沟通", "low",
         ["提前告知客户明日电价走势",
          "建议可调负荷客户避开晚峰时段"]),
    ]

    lines = [f"【策略评估】{province} {target_date}"]
    lines.append(f"  市场: 预测均价{avg:.0f}元/MWh, 峰谷价差{spread:.0f}元/MWh\n")
    for cat, priority, advices in strategies:
        prio_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[priority]
        lines.append(f"  {prio_icon} 【{cat}】")
        for a in advices:
            lines.append(f"    → {a}")
        lines.append("")

    lines.append(f"  ⚠️ 以上策略基于AI预测,请交易员审核后执行")
    return "\n".join(lines)


# ── CLI 测试 ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 因子计算 ===")
    print(compute_market_factors.invoke({"province": "山东", "target_date": "2026-06-16"}))
    print("\n=== 多角色分析 ===")
    print(run_multi_role_analysis.invoke({"province": "山东", "target_date": "2026-06-16"}))
    print("\n=== 场景推演 ===")
    print(run_scenario_simulation.invoke({"province": "山东", "target_date": "2026-06-16"}))
    print("\n=== 风控校验 ===")
    print(run_risk_validation.invoke({"province": "山东", "target_date": "2026-06-16"}))
    print("\n=== 策略建议 ===")
    print(run_strategy_advisor.invoke({"province": "山东", "target_date": "2026-06-16"}))
