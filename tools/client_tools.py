#!/usr/bin/env python3
"""
Client-domain tools — province entry assessment, client portfolio analysis,
dynamic EV charging pricing.

Practical business-intelligence tools for market expansion and client operations.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool


# ═══════════════════════════════════════════════════════════════════
# S17 省份入市评估
# ═══════════════════════════════════════════════════════════════════

# Province market profiles
_PROVINCE_PROFILES = {
    "山东": {
        "stage": "现货正式运行 (连续结算试运行)",
        "maturity": 8.0,
        "difficulty": "medium",
        "volatility": "medium",
        "regulations": [
            "山东省电力现货市场交易规则 (2024版)",
            "山东省中长期交易实施细则",
            "山东省电力市场偏差考核管理办法",
        ],
        "risks": [
            "新能源高占比导致现货价格波动加大",
            "分时电价调整频率较高",
            "偏差考核严格 (±5%免考)",
        ],
        "strategy": "建议以大比例中长期锁定基荷 (70-80%)，现货优化剩余20-30%。重点关注午间深谷时段的低价电利用。",
        "rating": "A",
    },
    "江苏": {
        "stage": "现货正式运行",
        "maturity": 8.5,
        "difficulty": "medium",
        "volatility": "medium",
        "regulations": [
            "江苏省电力市场交易规则",
            "江苏中长期带曲线交易细则",
            "江苏省辅助服务市场规则",
        ],
        "risks": [
            "跨省输电通道容量竞争激烈",
            "高耗能企业限电风险",
            "省内南北断面阻塞",
        ],
        "strategy": "利用省间价差进行跨省套利。建议配置储能利用高峰价差。",
        "rating": "A",
    },
    "山西": {
        "stage": "现货正式运行 (首批试点)",
        "maturity": 9.0,
        "difficulty": "low",
        "volatility": "high",
        "regulations": [
            "山西省电力现货市场规则 (V3.0)",
            "山西中长期+现货衔接细则",
            "山西省需求响应实施方案",
        ],
        "risks": [
            "煤电占比高，燃料价格波动影响大",
            "新能源装机快速增长，弃风弃光风险",
            "现货价格波动剧烈 (0-1500元/MWh)",
        ],
        "strategy": "煤电联动策略: 燃料成本+现货价格双因子模型。建议高比例中长期 (80%+) 对冲现货波动。",
        "rating": "B+",
    },
    "广东": {
        "stage": "现货正式运行",
        "maturity": 9.5,
        "difficulty": "medium",
        "volatility": "medium",
        "regulations": [
            "南方区域电力市场规则",
            "广东省电力市场交易规则",
            "广东调频辅助服务市场规则",
            "广东绿色电力交易规则",
        ],
        "risks": [
            "区域市场耦合度高，受南方各省供需影响",
            "高峰负荷增长快，夏季保供压力大",
            "气电占比高，国际LNG价格传导明显",
        ],
        "strategy": "市场成熟度高，建议多品种组合交易: 中长期+现货+辅助服务+绿电。关注LNG价格走势。",
        "rating": "A",
    },
    "内蒙古": {
        "stage": "现货试运行 / 中长期为主",
        "maturity": 5.5,
        "difficulty": "low",
        "volatility": "low",
        "regulations": [
            "蒙西电力市场交易规则",
            "内蒙古新能源消纳保障机制",
            "蒙西中长期交易细则",
        ],
        "risks": [
            "新能源装机占比极高，消纳压力大",
            "外送通道依赖严重",
            "省内负荷基数小，市场流动性不足",
        ],
        "strategy": "新能源+外送双驱动。重点关注省间交易通道和新能源保障性收购政策变化。",
        "rating": "B",
    },
}


@tool
def province_entry_assessment(
    province: str,
    business_type: str = "售电",
) -> str:
    """
    评估某省份电力市场入市条件与风险。

    从市场阶段、成熟度、入市难度、价格波动、关键政策、
    推荐策略、风险项和综合评级 8 个维度评估。

    当用户询问"XX省能进去做售电吗"、"哪个省好做"、"省份评估"时调用。

    Args:
        province: 目标省份名称
        business_type: 业务类型 (售电/储能/光伏/综合能源)，默认 "售电"

    Returns:
        市场阶段、成熟度评分、入市难度、价格波动、关键政策清单、
        推荐入市策略、风险项、综合评级 A/B/C/D。
    """
    try:
        profile = _PROVINCE_PROFILES.get(province)
        if profile is None:
            return (
                f"【省份评估】{province}\n"
                f"  ⚠️ 该省份尚未建立详细评估档案。建议参考邻近省份或联系当地交易中心获取最新信息。\n"
                f"  如需添加该省份数据，请在 config/provinces/{province}.yaml 中完善市场规则配置。"
            )

        difficulty_labels = {"low": "低 🟢", "medium": "中 🟡", "high": "高 🔴"}
        volatility_labels = {"low": "低 (平稳)", "medium": "中 (正常)", "high": "高 (剧烈)"}
        rating_labels = {
            "A": "A (优秀 — 强烈推荐入市)",
            "A-": "A- (优良 — 推荐入市)",
            "B+": "B+ (良好 — 建议入市)",
            "B": "B (一般 — 谨慎入市)",
            "B-": "B- (偏低 — 需深入研究后入市)",
            "C": "C (较差 — 暂不建议入市)",
            "D": "D (差 — 不建议入市)",
        }

        # Business-type-specific adjustments
        business_notes = {
            "售电": "售电公司需关注: 准入资本金要求、履约保函、偏差考核规则。",
            "储能": "储能业务需关注: 峰谷价差空间、辅助服务补偿标准、容量市场参与资格。",
            "光伏": "光伏业务需关注: 燃煤基准价、市场化交易比例、保障性收购小时数、绿证交易。",
            "综合能源": "综合能源需关注: 增量配电网政策、综合能源服务资质、源网荷储一体化项目申报。",
        }

        lines = [
            f"【省份入市评估】{province}  |  业务类型: {business_type}",
            f"",
            f"  市场阶段: {profile['stage']}",
            f"  市场成熟度: {profile['maturity']:.1f} / 10",
            f"  入市难度: {difficulty_labels.get(profile['difficulty'], profile['difficulty'])}",
            f"  现货波动: {volatility_labels.get(profile['volatility'], profile['volatility'])}",
            f"  综合评级: {rating_labels.get(profile['rating'], profile['rating'])}",
            f"",
            f"  ── 关键政策文件 ──",
        ]
        for i, reg in enumerate(profile["regulations"], 1):
            lines.append(f"  {i}. {reg}")

        lines.append(f"")
        lines.append(f"  ── 推荐策略 ──")
        lines.append(f"  {profile['strategy']}")

        lines.append(f"")
        lines.append(f"  ── 风险提示 ──")
        for i, risk in enumerate(profile["risks"], 1):
            lines.append(f"  {i}. ⚠️ {risk}")

        lines.append(f"")
        note = business_notes.get(business_type, business_notes["综合能源"])
        lines.append(f"  💡 {note}")

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】省份评估异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S23 客户组合分析
# ═══════════════════════════════════════════════════════════════════


@tool
def client_portfolio_analyzer(
    clients_json: str = "",
) -> str:
    """
    售电公司客户组合分析: 评估客户分布、降本潜力和到期续约风险。

    当用户询问"客户组合怎么样"、"哪些客户快到期了"、"降本潜力排序"时调用。

    Args:
        clients_json: 客户列表 JSON，格式 [{"client_id":"C001","name":"XX钢铁",
                      "monthly_mwh":5000,"avg_price":420,"contract_expiry":"2026-06-30"},...]
                      为空则使用示例数据

    Returns:
        客户数量、总用电量、总降本潜力、即将到期客户、逐客户分析与建议。
    """
    try:
        import json
        from datetime import datetime, date

        if clients_json:
            clients = json.loads(clients_json)
        else:
            # Example portfolio
            clients = [
                {"client_id": "C001", "name": "泰山钢铁", "monthly_mwh": 28000, "avg_price": 415,
                 "contract_expiry": "2026-06-30", "province": "山东"},
                {"client_id": "C002", "name": "齐鲁化工", "monthly_mwh": 18500, "avg_price": 435,
                 "contract_expiry": "2026-09-15", "province": "山东"},
                {"client_id": "C003", "name": "德州纺织", "monthly_mwh": 6200, "avg_price": 450,
                 "contract_expiry": "2026-05-31", "province": "山东"},
                {"client_id": "C004", "name": "临沂建材", "monthly_mwh": 12300, "avg_price": 408,
                 "contract_expiry": "2027-01-01", "province": "山东"},
                {"client_id": "C005", "name": "青岛电子", "monthly_mwh": 9600, "avg_price": 445,
                 "contract_expiry": "2026-07-20", "province": "山东"},
                {"client_id": "C006", "name": "潍坊食品", "monthly_mwh": 4500, "avg_price": 460,
                 "contract_expiry": "2026-04-15", "province": "山东"},
                {"client_id": "C007", "name": "淄博陶瓷", "monthly_mwh": 7800, "avg_price": 428,
                 "contract_expiry": "2026-11-30", "province": "山东"},
                {"client_id": "C008", "name": "烟台机械", "monthly_mwh": 15200, "avg_price": 412,
                 "contract_expiry": "2026-08-01", "province": "山东"},
            ]

        today = date.today()
        total_monthly_mwh = 0.0
        total_saving_potential = 0.0
        expiring_soon = 0
        threshold_date = today + __import__('datetime').timedelta(days=60)

        items = []
        target_price = 350  # assumed achievable price with optimization

        for c in clients:
            monthly_mwh = c.get("monthly_mwh", 0)
            avg_price = c.get("avg_price", 400)
            expiry = c.get("contract_expiry", "")

            total_monthly_mwh += monthly_mwh

            # Saving potential: difference between current avg price and optimized price
            if avg_price > target_price:
                saving_per_mwh = avg_price - target_price
                saving_potential = monthly_mwh * saving_per_mwh
            else:
                saving_potential = 0

            total_saving_potential += saving_potential

            # Contract expiry check
            if expiry:
                expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                days_left = (expiry_date - today).days
                if days_left <= 60:
                    expiring_soon += 1
            else:
                days_left = None

            # Recommended action
            if days_left is not None and days_left <= 30:
                action = "🔴 紧急续约 — 合同即将到期，立即启动续约谈判"
            elif days_left is not None and days_left <= 60:
                action = "🟡 准备续约 — 合同60日内到期，制定续约方案"
            elif avg_price > 440:
                action = "📉 优化电价 — 当前电价偏高，建议利用中长期+现货组合降本"
            elif avg_price > 410:
                action = "📊 持续优化 — 有一定降本空间，关注储能/需求响应增收"
            else:
                action = "✅ 保持 — 价格合理，维持现状，关注偏差管理"

            items.append({
                "client_id": c["client_id"],
                "name": c.get("name", c["client_id"]),
                "monthly_mwh": monthly_mwh,
                "avg_price": avg_price,
                "cost_saving_potential": round(saving_potential, 0),
                "contract_expiry": expiry,
                "days_to_expiry": days_left,
                "action": action,
            })

        # Sort by saving potential descending
        items.sort(key=lambda x: x["cost_saving_potential"], reverse=True)

        lines = [
            f"【客户组合分析】({today.strftime('%Y-%m-%d')})",
            f"",
            f"  客户总数: {len(clients)}",
            f"  月度总用电量: {total_monthly_mwh:,.0f} MWh",
            f"  月度总降本潜力: ¥{total_saving_potential:,.0f}",
            f"  年化降本潜力: ¥{total_saving_potential * 12:,.0f}",
            f"  即将到期 (<60天): {expiring_soon} 家",
            f"",
            f"  ── 客户明细 (按降本潜力排序) ──",
        ]

        for item in items:
            days_str = f"{item['days_to_expiry']}天" if item["days_to_expiry"] is not None else "未知"
            lines.append(
                f"  {item['name']} ({item['client_id']}): "
                f"{item['monthly_mwh']:,.0f}MWh/月 | "
                f"均价 {item['avg_price']:.0f}元/MWh | "
                f"降本潜力 ¥{item['cost_saving_potential']:,.0f}/月 | "
                f"到期: {item['contract_expiry']} ({days_str})"
            )
            lines.append(f"    → {item['action']}")

        return "\n".join(lines)

    except json.JSONDecodeError:
        return "【参数错误】clients_json 格式不正确，请提供合法 JSON 数组。"
    except Exception as e:
        return f"【系统提示】客户组合分析异常: {type(e).__name__}: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════════
# S19 充电桩动态定价
# ═══════════════════════════════════════════════════════════════════


@tool
def dynamic_pricing_ev(
    province: str,
    target_date: str,
    station_id: str = "EV001",
    total_capacity_kw: float = 1200.0,
    purchase_cost_profile: str = "",
) -> str:
    """
    电动汽车充电站动态定价: 基于分时购电成本 + 服务费，制定分时充电价格。

    策略: 谷电时段降低售价吸引充电，峰电时段适当提价，
    平滑充电负荷曲线同时最大化利润。

    当用户询问"充电桩怎么定价"、"充电站怎么赚钱"、"EV动态电价"时调用。

    Args:
        province: 省份
        target_date: 日期 YYYY-MM-DD
        station_id: 充电站ID
        total_capacity_kw: 总装机容量 (kW)
        purchase_cost_profile: 24点购电成本 (元/kWh)，逗号分隔，为空使用默认分时电价

    Returns:
        24点逐时定价、预期充电量、收入、利润和负荷转移效果。
    """
    try:
        # Purchase cost (convert from 元/MWh to 元/kWh)
        if purchase_cost_profile:
            costs_yuan_per_mwh = [float(x.strip()) for x in purchase_cost_profile.split(",")]
            if len(costs_yuan_per_mwh) != 24:
                return "【参数错误】purchase_cost_profile 需提供 24 个小时的值 (逗号分隔)"
            purchase_cost = [c / 1000 for c in costs_yuan_per_mwh]  # 元/kWh
        else:
            # Default TOU purchase cost (元/kWh)
            purchase_cost = [
                0.28, 0.26, 0.24, 0.22, 0.21, 0.20,
                0.25, 0.35, 0.50, 0.60, 0.62, 0.58,
                0.45, 0.40, 0.38, 0.42, 0.50, 0.62,
                0.70, 0.80, 0.75, 0.65, 0.50, 0.38,
            ]

        # Service fee strategy (元/kWh)
        # Valley: low service fee to attract demand
        # Peak: moderate service fee to manage load
        # Super peak: high service fee to suppress and profit
        base_service_fee = 0.30  # base service fee

        hourly_pricing = []
        total_revenue = 0.0
        total_profit = 0.0
        total_load = 0.0

        # Expected load profile (normalized utilization)
        # Higher during daytime, lower at night
        base_utilization = [
            0.15, 0.10, 0.08, 0.07, 0.06, 0.08,
            0.20, 0.40, 0.55, 0.65, 0.70, 0.68,
            0.60, 0.55, 0.58, 0.62, 0.65, 0.72,
            0.80, 0.85, 0.78, 0.60, 0.40, 0.25,
        ]

        # After dynamic pricing: simulate load shift
        shifted_utilization = base_utilization.copy()

        for h in range(24):
            cost = purchase_cost[h]

            # Pricing strategy based on TOU segment
            if cost < 0.30:  # valley
                tou = "谷段"
                service_fee = base_service_fee * 0.6  # discount service fee
                shifted_utilization[h] *= 1.3  # attract more charging
            elif cost > 0.60:  # peak
                tou = "峰段"
                service_fee = base_service_fee * 1.5  # premium service fee
                shifted_utilization[h] *= 0.85  # slightly reduce
            else:  # flat
                tou = "平段"
                service_fee = base_service_fee
                shifted_utilization[h] *= 1.05

            user_price = cost + service_fee

            # Expected load
            expected_load_kw = total_capacity_kw * shifted_utilization[h]
            expected_load_kwh = expected_load_kw * 1  # 1 hour

            # Revenue & profit
            revenue = expected_load_kwh * user_price
            profit = expected_load_kwh * service_fee  # profit = service fee revenue

            total_revenue += revenue
            total_profit += profit
            total_load += expected_load_kwh

            hourly_pricing.append({
                "hour": h,
                "tou": tou,
                "purchase_cost": round(cost, 3),
                "service_fee": round(service_fee, 3),
                "user_price": round(user_price, 3),
                "load_kwh": round(expected_load_kwh, 0),
                "revenue": round(revenue, 0),
                "profit": round(profit, 0),
            })

        # Load shift metric: how much moved from peak to valley
        peak_hours_base = [9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
        valley_hours = [0, 1, 2, 3, 4, 5, 22, 23]
        peak_load_before = sum(base_utilization[h] for h in peak_hours_base)
        peak_load_after = sum(shifted_utilization[h] for h in peak_hours_base)
        valley_load_before = sum(base_utilization[h] for h in valley_hours)
        valley_load_after = sum(shifted_utilization[h] for h in valley_hours)

        load_shift_pct = ((valley_load_after - valley_load_before) / max(0.01, peak_load_before)) * 100
        profit_margin = total_profit / max(1, total_revenue) * 100

        lines = [
            f"【充电桩动态定价】{station_id} @ {province} {target_date}",
            f"  总装机: {total_capacity_kw:.0f} kW",
            f"  日总充电量: {total_load:,.0f} kWh",
            f"  日总收入: ¥{total_revenue:,.0f}",
            f"  日总利润: ¥{total_profit:,.0f} (服务费收入)",
            f"  利润率: {profit_margin:.1f}%",
            f"  峰→谷负荷转移: {load_shift_pct:.1f}%",
            f"",
            "  Hour | 段   | 购电成本 | 服务费 | 用户价 | 充电量kWh | 收入  | 利润",
            "  -----|------|---------|--------|--------|----------|-------|------",
        ]
        for p in hourly_pricing:
            lines.append(
                f"  {p['hour']:02d}:00 | {p['tou']:4s} | "
                f"{p['purchase_cost']:.3f}   | {p['service_fee']:.3f}  | "
                f"{p['user_price']:.3f}  | {p['load_kwh']:8.0f} | "
                f"{p['revenue']:5.0f} | {p['profit']:5.0f}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"【系统提示】动态定价异常: {type(e).__name__}: {str(e)[:300]}"
