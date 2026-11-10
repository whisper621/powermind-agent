#!/usr/bin/env python3
"""
PowerMind 售电交易侧 - 完整数据生成器 V3。

按三个场景逐一分析所需数据:

场景A "日前申报策略":
  ✅ 24点电价预测 (15天)           → 已生成
  ❌ 历史日前出清价 (90天实际值)    → 本次补充: 用于对比+因子计算+模型输入
  ✅ 政策规则                      → 已生成 (YAML + 政策TXT)
  ✅ 交易复盘经验                   → 已生成
  ❌ 日前申报记录                   → 本次补充: 历史申报价 vs 出清价

场景B "实时电价交易策略":
  ❌ 实时电价快照 (近7天逐15分钟)   → 本次补充
  ❌ 持仓数据                       → 本次补充: 当前持仓+日前申报记录
  ✅ 日前出清价 (对比基准)           → 本次补充 (场景A的历史数据)

场景C "中长期锁价策略":
  ❌ 中长期合约持仓                  → 本次补充: 合约列表+价格+电量+到期日
  ❌ 滚动撮合历史成交价              → 本次补充: 月度滚动撮合行情
  ❌ 远期价格预测                   → 已生成 (复用场景A的预测)
  ❌ 历史基差数据                   → 本次补充: 中长期合约价 vs 现货价

全部存入 data/ 对应子目录中。

设计:
  - data/source_data/ → 结构化数据 CSV/JSON 源文件 (人工准备)
  - data/source_docs/ → 文本知识 TXT/MD 源文件 (人工准备)
  - data/vectordb/    → ChromaDB 向量库 (自动生成)
  - data/sql/         → SQLite 数据库 (自动生成)
"""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data"
random.seed(42)

PROVINCE = "山东"


# ═══════════════════════════════════════════════════════════════════
# 场景A 补充: 历史日前出清价 (90天实际) + 历史申报记录
# ═══════════════════════════════════════════════════════════════════

def generate_historical_prices(days: int = 90):
    """
    生成历史日前出清电价 —— 这是昨天及之前的实际出清结果。
    与"预测数据"不同，这是已经发生的真实价格(模拟)。

    用途:
      - LSTM/因子计算需要的输入特征
      - 对比"申报价 vs 实际出清价"
      - 评估历史预测准确率
    """
    staging = DATA / "source_data" / PROVINCE / "grid_operation"
    staging.mkdir(parents=True, exist_ok=True)

    today = datetime.now()
    rows = []

    for d in range(days, 0, -1):
        date_obj = today - timedelta(days=d)
        date_str = date_obj.strftime("%Y-%m-%d")
        month = date_obj.month
        weekday = date_obj.weekday()

        # 季节基准
        if month in (6, 7, 8):
            base = 420
        elif month in (12, 1, 2):
            base = 400
        else:
            base = 360

        # 周末负荷低 → 电价略低
        if weekday >= 5:
            base *= 0.92

        daily_prices = []
        for hour in range(24):
            if 0 <= hour < 6:
                factor = 0.65
            elif 6 <= hour < 10:
                factor = 0.90
            elif 10 <= hour < 15:
                factor = 0.80 if month in (6, 7, 8) else 1.0
            elif 15 <= hour < 18:
                factor = 1.05
            elif 18 <= hour < 21:
                factor = 1.35
            else:
                factor = 0.85

            noise = random.gauss(0, 25)
            price = max(0, min(1500, base * factor + noise))
            daily_prices.append(round(price, 2))

        avg = round(sum(daily_prices) / 24, 2)
        peak = round(sum(daily_prices[18:21]) / 3, 2)
        valley = round(sum(daily_prices[0:6]) / 6, 2)
        spread = round(peak - valley, 2)

        # 负荷数据 (MW)
        base_load = 48000 + random.gauss(0, 5000)
        peak_load = base_load * 1.2
        wind_output = round(random.uniform(2000, 6000), 0)
        solar_output = round(random.uniform(0, 8000), 0)

        rows.append({
            "province": PROVINCE,
            "date": date_str,
            "avg_price": avg,
            "peak_avg": peak,
            "valley_avg": valley,
            "spread": spread,
            "max_price": round(max(daily_prices), 2),
            "min_price": round(min(daily_prices), 2),
            "hourly_prices": json.dumps(daily_prices, ensure_ascii=False),
            "base_load_mw": round(base_load, 0),
            "peak_load_mw": round(peak_load, 0),
            "wind_output_mw": wind_output,
            "solar_output_mw": solar_output,
            "weekday": weekday,
        })

    # 保存为 CSV (可直接导入 PostgreSQL price_data 表)
    csv_path = staging / f"{PROVINCE}_day_ahead_price_history_{rows[-1]['date']}_to_{rows[0]['date']}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[场景A-历史电价] {csv_path} ({len(rows)} 天, {len(rows)*24} 个数据点)")
    return str(csv_path), rows


def generate_bidding_history(days: int = 30):
    """
    生成历史日前申报记录 —— 每天申报了多少钱、出清了多少钱、偏差多少。

    用途:
      - 场景A: "上次类似行情我申报了多少" - 经验参考
      - 场景B: 当前持仓的对标基准
      - 复盘: 申报准确率统计
    """
    staging = DATA / "source_data" / PROVINCE / "long_term_contracts"
    staging.mkdir(parents=True, exist_ok=True)

    today = datetime.now()
    records = []

    for d in range(days, 0, -1):
        date_obj = today - timedelta(days=d)
        date_str = date_obj.strftime("%Y-%m-%d")
        month = date_obj.month

        if month in (6, 7, 8):
            base = 420
        elif month in (12, 1, 2):
            base = 400
        else:
            base = 360

        # 申报策略: 基于前一天出清价的随机调整
        bid_adjustment = random.choice([-0.02, -0.01, 0, 0.01, 0.02, 0.03])
        bid_avg = round(base * (1 + bid_adjustment) + random.gauss(0, 15), 2)
        clearing_avg = round(base + random.gauss(0, 30), 2)
        deviation_pct = round((clearing_avg - bid_avg) / bid_avg * 100, 2)
        volume_mwh = round(random.uniform(200, 500), 0)

        records.append({
            "province": PROVINCE,
            "trade_date": date_str,
            "bid_avg_price": bid_avg,
            "clearing_avg_price": clearing_avg,
            "deviation_pct": deviation_pct,
            "deviation_status": "pass" if abs(deviation_pct) <= 5 else "fail",
            "volume_mwh": volume_mwh,
            "peak_bid_price": round(bid_avg * 1.35, 2),
            "valley_bid_price": round(bid_avg * 0.65, 2),
            "strategy_type": random.choice(["保守", "激进", "平衡"]),
            "notes": random.choice([
                "光伏出力偏高，午间降价", "晚峰紧张，溢价申报",
                "全天平价出货", "低价保量策略", "观望为主"
            ]),
        })

    csv_path = staging / f"{PROVINCE}_bidding_history_{records[-1]['trade_date']}_to_{records[0]['trade_date']}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"[场景A-申报记录] {csv_path} ({len(records)} 天)")
    return str(csv_path), records


# ═══════════════════════════════════════════════════════════════════
# 场景B 补充: 实时电价快照 + 持仓数据
# ═══════════════════════════════════════════════════════════════════

def generate_realtime_prices(days: int = 7):
    """
    生成模拟实时电价快照 (逐15分钟)。

    用途:
      - 场景B: "当前实时电价185元/MWh，偏离日前出清-32%"
      - 实时监盘: 预警触发
    """
    staging = DATA / "source_data" / PROVINCE / "spot_prices"
    staging.mkdir(parents=True, exist_ok=True)

    today = datetime.now()
    all_points = []

    for d in range(min(days, 3), 0, -1):  # 最近3天逐15分钟
        date_obj = today - timedelta(days=d)
        date_str = date_obj.strftime("%Y-%m-%d")

        for hour in range(24):
            for minute in (0, 15, 30, 45):
                time_str = f"{hour:02d}:{minute:02d}"

                if 0 <= hour < 6:
                    base_price = 260
                elif 10 <= hour < 15:
                    base_price = 340
                elif 18 <= hour < 21:
                    base_price = 520
                else:
                    base_price = 380

                noise = random.gauss(0, 40)
                price = max(0, min(1500, base_price + noise))

                all_points.append({
                    "province": PROVINCE,
                    "datetime": f"{date_str} {time_str}",
                    "date": date_str,
                    "time": time_str,
                    "rt_price": round(price, 2),
                    "da_price": round(base_price + random.gauss(0, 15), 2),  # 日前对比
                })

    csv_path = staging / f"{PROVINCE}_realtime_prices_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_points[0].keys())
        writer.writeheader()
        writer.writerows(all_points)

    print(f"[场景B-实时电价] {csv_path} ({len(all_points)} 个时间点)")
    return str(csv_path), all_points


def generate_position_data():
    """
    生成当前持仓数据。

    用途:
      - 场景B: 实时交易策略需要知道"我现在有多少电量"
      - 场景C: 中长期策略需要知道"我已经锁了多少"
    """
    staging = DATA / "source_data" / PROVINCE / "long_term_contracts"
    staging.mkdir(parents=True, exist_ok=True)

    today = datetime.now()

    # 日前持仓 (明天要交付的电量)
    positions = {
        "current_positions": {
            "day_ahead": {
                "target_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
                "contracted_volume_mwh": 320,
                "contracted_price": 365.0,
                "expected_load_mwh": 480,
                "open_position_mwh": 160,  # 待补仓/待卖出
                "status": "部分持仓",
            },
            "real_time_today": {
                "date": today.strftime("%Y-%m-%d"),
                "contracted_volume_mwh": 280,
                "avg_contracted_price": 358.0,
                "current_rt_avg": 372.5,
                "pnl_estimate": round((372.5 - 358.0) * 280, 0),
                "status": "盈利中",
            },
        },
        "long_term_holdings": [
            {
                "contract_id": "LT-2026-07-001",
                "type": "年度双边",
                "volume_mwh": 15000,
                "price": 355.0,
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
                "monthly_allocation_mwh": 1250,
                "counterparty": "华能山东",
            },
            {
                "contract_id": "MT-2026-06-003",
                "type": "月度集中竞价",
                "volume_mwh": 3500,
                "price": 362.0,
                "start_date": "2026-06-01",
                "end_date": "2026-06-30",
                "monthly_allocation_mwh": 3500,
                "counterparty": "月度市场",
            },
            {
                "contract_id": "MT-2026-07-002",
                "type": "月度集中竞价",
                "volume_mwh": 4000,
                "price": 370.0,
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
                "monthly_allocation_mwh": 4000,
                "counterparty": "月度市场",
            },
        ],
        "total_long_term_locked_mwh": 22500,
        "annual_consumption_estimate_mwh": 60000,
        "lock_ratio_pct": 37.5,
    }

    json_path = staging / f"{PROVINCE}_current_positions.json"
    json_path.write_text(json.dumps(positions, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[场景B/C-持仓数据] {json_path}")
    return str(json_path), positions


# ═══════════════════════════════════════════════════════════════════
# 场景C 补充: 中长期合约 + 滚动撮合历史 + 基差数据
# ═══════════════════════════════════════════════════════════════════

def generate_long_term_market_data(months: int = 12):
    """
    生成中长期市场行情数据。

    用途:
      - 场景C: "滚动撮合市场今天什么价"
      - 基差分析: 中长期合约价 vs 现货价的历史走势
    """
    staging = DATA / "source_data" / PROVINCE / "long_term_contracts"
    staging.mkdir(parents=True, exist_ok=True)

    today = datetime.now()
    records = []

    # 月度滚动撮合历史 (每月1日和15日)
    for m in range(months, 0, -1):
        month_start = (today.replace(day=1) - timedelta(days=m * 30)).replace(day=1)

        for day_offset in [1, 15]:
            trade_date = month_start + timedelta(days=day_offset - 1)
            if trade_date > today:
                continue

            # 不同期限的成交价
            spot_ref = 380 + random.gauss(0, 30)  # 当时现货价

            records.append({
                "province": PROVINCE,
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "contract_month": month_start.strftime("%Y-%m"),
                "contract_type": "月度滚动撮合",
                "volume_mwh": round(random.uniform(500, 3000), 0),
                "deal_price": round(spot_ref * random.uniform(0.88, 1.02), 2),
                "spot_reference": round(spot_ref, 2),
                "basis": round(spot_ref * random.uniform(0.88, 1.02) - spot_ref, 2),  # 基差
                "bid_count": random.randint(3, 15),
                "offer_count": random.randint(5, 20),
            })

    csv_path = staging / f"{PROVINCE}_rolling_match_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"[场景C-滚动撮合] {csv_path} ({len(records)} 笔成交)")
    return str(csv_path), records


def generate_basis_history(months: int = 12):
    """
    生成中长期合约 vs 现货 基差历史数据。

    用途:
      - 场景C: "当前合约价370元/MWh vs 预测现货均价385元/MWh，基差-15，建议增持"
    """
    staging = DATA / "source_data" / PROVINCE / "long_term_contracts"
    staging.mkdir(parents=True, exist_ok=True)

    today = datetime.now()
    records = []

    for m in range(months, 0, -1):
        month_date = (today.replace(day=1) - timedelta(days=m * 30)).replace(day=1)
        month_str = month_date.strftime("%Y-%m")

        # 该月现货均价 (模拟)
        month_num = month_date.month
        if month_num in (6, 7, 8):
            spot_base = 420
        elif month_num in (12, 1, 2):
            spot_base = 400
        else:
            spot_base = 360

        spot_avg = round(spot_base + random.gauss(0, 35), 2)

        # 该月中长期合约价 (通常是现货的90-100%)
        contract_price = round(spot_avg * random.uniform(0.88, 0.98), 2)
        basis = round(contract_price - spot_avg, 2)

        records.append({
            "province": PROVINCE,
            "month": month_str,
            "spot_monthly_avg": spot_avg,
            "long_term_contract_price": contract_price,
            "basis": basis,
            "basis_pct": round(basis / spot_avg * 100, 2),
            "recommendation": "增持" if basis < -15 else ("减持" if basis > 10 else "观望"),
            "actual_next_month_spot": round(spot_base + random.gauss(0, 40), 2),
        })

    csv_path = staging / f"{PROVINCE}_basis_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"[场景C-基差数据] {csv_path} ({len(records)} 个月)")
    return str(csv_path), records


# ═══════════════════════════════════════════════════════════════════
# 总入口
# ═══════════════════════════════════════════════════════════════════

def generate_all_scenario_data():
    """生成三个场景的全部所需数据"""
    print("=" * 60)
    print("PowerMind 售电交易侧 完整数据生成 V3")
    print(f"目标省份: {PROVINCE}")
    print("=" * 60)
    print()
    print("分析三个场景所需数据...")
    print()
    print("场景A '日前申报策略' 需要:")
    print("  [OK] 24点电价预测 (15天)           - 已生成")
    print("  [NEW] 历史日前出清价 (90天实际)    - 本次补充")
    print("  [NEW] 申报记录 (30天)              - 本次补充")
    print("  [OK] 政策规则                      - 已生成")
    print("  [OK] 交易复盘经验                  - 已生成")
    print()
    print("场景B '实时电价交易' 需要:")
    print("  [NEW] 实时电价快照 (7天逐15分钟)   - 本次补充")
    print("  [NEW] 持仓数据 (日前+中长期)       - 本次补充")
    print("  [OK] 日前出清价 (对比基准)         - 本次补充 (复用场景A)")
    print()
    print("场景C '中长期锁价' 需要:")
    print("  [NEW] 中长期合约持仓               - 本次补充")
    print("  [NEW] 滚动撮合历史成交价           - 本次补充")
    print("  [NEW] 历史基差数据                 - 本次补充")
    print("  [OK] 远期价格预测                  - 已生成")
    print()
    print("=" * 60)
    print("开始生成...")
    print("=" * 60)
    print()

    results = {}

    # 场景A
    print("[场景A-1] 历史日前出清价 (90天)...")
    results["price_history_path"], results["price_history"] = generate_historical_prices(90)

    print("[场景A-2] 历史申报记录 (30天)...")
    results["bidding_history_path"], results["bidding_history"] = generate_bidding_history(30)

    # 场景B
    print("[场景B-1] 实时电价快照 (7天逐15分钟)...")
    results["realtime_path"], results["realtime"] = generate_realtime_prices(7)

    print("[场景B-2] 持仓数据...")
    results["positions_path"], results["positions"] = generate_position_data()

    # 场景C
    print("[场景C-1] 滚动撮合历史成交价...")
    results["rolling_match_path"], results["rolling_match"] = generate_long_term_market_data(12)

    print("[场景C-2] 历史基差数据...")
    results["basis_path"], results["basis"] = generate_basis_history(12)

    print()
    print("=" * 60)
    print("全场景数据生成完成！")
    print("=" * 60)
    print()
    print("数据文件清单:")
    for key, path in results.items():
        if key.endswith("_path"):
            print(f"  {key}: {path}")

    return results


if __name__ == "__main__":
    generate_all_scenario_data()
