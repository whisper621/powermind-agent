#!/usr/bin/env python3
"""Build the deterministic PowerMind V1 demo database.

The demo runtime reads every market number, recommendation and knowledge
document from SQLite.  No model training, LLM, remote API or network call is
required at runtime.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.paths import get_db_path


SEED_VERSION = "demo-v1.3.0"
PROVINCE = "山东"


SCHEMA = """
CREATE TABLE IF NOT EXISTS demo_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS price_data (
    province TEXT NOT NULL, date TEXT NOT NULL, hour INTEGER NOT NULL,
    price_rmb_mwh REAL NOT NULL, load_mw REAL NOT NULL, renewable_mw REAL NOT NULL,
    data_source TEXT NOT NULL,
    PRIMARY KEY (province, date, hour)
);
CREATE TABLE IF NOT EXISTS forecasts (
    province TEXT NOT NULL, target_date TEXT NOT NULL, hour INTEGER NOT NULL,
    predicted_price REAL NOT NULL, lower_bound REAL NOT NULL, upper_bound REAL NOT NULL,
    confidence REAL NOT NULL, model_version TEXT NOT NULL, generated_at TEXT NOT NULL,
    data_source TEXT NOT NULL,
    PRIMARY KEY (province, target_date, hour)
);
CREATE TABLE IF NOT EXISTS weather_data (
    province TEXT NOT NULL, date TEXT NOT NULL, hour INTEGER NOT NULL,
    temperature_2m REAL NOT NULL, cloud_cover REAL NOT NULL, wind_speed_10m REAL NOT NULL,
    precipitation_mm REAL NOT NULL, data_source TEXT NOT NULL,
    PRIMARY KEY (province, date, hour)
);
CREATE TABLE IF NOT EXISTS market_factors (
    province TEXT NOT NULL, target_date TEXT NOT NULL, factor_code TEXT NOT NULL,
    factor_name TEXT NOT NULL, score REAL NOT NULL, direction TEXT NOT NULL,
    observation TEXT NOT NULL, data_source TEXT NOT NULL,
    PRIMARY KEY (province, target_date, factor_code)
);
CREATE TABLE IF NOT EXISTS bidding_records (
    record_id TEXT PRIMARY KEY, province TEXT NOT NULL, trade_date TEXT NOT NULL,
    bid_avg_price REAL NOT NULL, clearing_avg_price REAL NOT NULL,
    deviation_pct REAL NOT NULL, volume_mwh REAL NOT NULL, strategy_type TEXT NOT NULL,
    result TEXT NOT NULL, data_source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS contracts (
    contract_id TEXT PRIMARY KEY, province TEXT NOT NULL, contract_type TEXT NOT NULL,
    start_date TEXT NOT NULL, end_date TEXT NOT NULL, volume_mwh REAL NOT NULL,
    price_rmb_mwh REAL NOT NULL, counterparty TEXT NOT NULL, status TEXT NOT NULL,
    data_source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS basis_history (
    province TEXT NOT NULL, month TEXT NOT NULL, spot_avg REAL NOT NULL,
    contract_price REAL NOT NULL, basis REAL NOT NULL, recommendation TEXT NOT NULL,
    data_source TEXT NOT NULL,
    PRIMARY KEY (province, month)
);
CREATE TABLE IF NOT EXISTS knowledge_documents (
    doc_id TEXT PRIMARY KEY, doc_type TEXT NOT NULL, province TEXT NOT NULL,
    title TEXT NOT NULL, document_no TEXT NOT NULL, effective_date TEXT NOT NULL,
    source_name TEXT NOT NULL, source_locator TEXT NOT NULL, keywords TEXT NOT NULL,
    content TEXT NOT NULL, data_source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS demo_scenarios (
    task_type TEXT NOT NULL, province TEXT NOT NULL, target_date TEXT NOT NULL,
    title TEXT NOT NULL, summary TEXT NOT NULL, payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL, data_source TEXT NOT NULL,
    PRIMARY KEY (task_type, province, target_date)
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
    request_id TEXT NOT NULL, event_type TEXT NOT NULL, component TEXT NOT NULL,
    detail TEXT NOT NULL, success INTEGER NOT NULL, business_critical INTEGER NOT NULL,
    metadata_json TEXT NOT NULL
);
"""


def _hour_factor(hour: int) -> float:
    if hour < 6:
        return 0.70
    if hour < 10:
        return 1.02
    if hour < 15:
        return 0.84
    if hour < 18:
        return 1.05
    if hour < 21:
        return 1.36
    return 0.88


def _monthly_date(base: date, months_back: int) -> date:
    total = base.year * 12 + base.month - 1 - months_back
    return date(total // 12, total % 12 + 1, 1)


def _scenario_payload(
    task_type: str,
    target: date,
    avg: float,
    peak: float,
    valley: float,
    hourly: list[float],
) -> dict:
    common = {
        "data_mode": "纯数据库演示假数据",
        "forecast_avg": round(avg, 1),
        "forecast_peak": round(peak, 1),
        "forecast_valley": round(valley, 1),
        "forecast_peak_avg": round(peak, 1),
        "forecast_valley_avg": round(valley, 1),
        "forecast_max": round(max(hourly), 1),
        "forecast_max_hour": hourly.index(max(hourly)),
        "forecast_min": round(min(hourly), 1),
        "forecast_min_hour": hourly.index(min(hourly)),
        "time_block_avg_spread": round(peak - valley, 1),
        "intraday_range": round(max(hourly) - min(hourly), 1),
        "confidence": 0.86,
        "manual_confirmation_required": True,
    }
    if task_type == "daily_pulse":
        return common | {
            "headline": f"{target:%m月%d日}预计晚峰偏紧、午间光伏压价，全天均价约{avg:.1f}元/MWh。",
            "actions": ["晚峰18-21时关注负荷上修", "午间保留低价补仓窗口", "15:00复核最新气象假数据"],
            "risks": ["高温负荷高于演示基线", "风电波动扩大峰谷价差"],
        }
    if task_type == "bidding_advice":
        segment_specs = [
            ("00:00-06:00", 0, 6, 92, 8, "低价保量"),
            ("06:00-12:00", 6, 12, 118, -3, "平衡申报"),
            ("12:00-18:00", 12, 18, 126, -20, "利用光伏窗口"),
            ("18:00-24:00", 18, 24, 144, 12, "晚峰保供"),
        ]
        segments = []
        for period, start, end, volume, adjustment, action in segment_specs:
            forecast_price = round(sum(hourly[start:end]) / (end - start), 1)
            segments.append({
                "period": period,
                "volume_mwh": volume,
                "forecast_price": forecast_price,
                "bid_price": round(forecast_price + adjustment, 1),
                "adjustment": adjustment,
                "action": action,
            })
        total_volume = sum(item["volume_mwh"] for item in segments)
        weighted_avg = round(
            sum(item["volume_mwh"] * item["bid_price"] for item in segments) / total_volume,
            1,
        )
        return common | {
            "headline": "建议采用分段量价、晚峰保供、午间谨慎追价的平衡方案。",
            "recommended_bid_avg": weighted_avg,
            "total_volume_mwh": total_volume,
            "segments": segments,
            "alternative_scenarios": [
                {
                    "name": "保守",
                    "rule": "所有时段申报价上浮8元/MWh，优先成交",
                    "weighted_avg": round(weighted_avg + 8, 1),
                },
                {
                    "name": "进取",
                    "rule": "仅12:00-18:00申报价下调15元/MWh，接受少量未成交",
                    "weighted_avg": round(weighted_avg - 15 * 126 / total_volume, 1),
                },
            ],
            "alternatives": ["保守：均价上浮8元/MWh，优先成交", "进取：午间下调15元/MWh，接受少量未成交"],
            "stop_conditions": ["预测均价变化超过8%", "负荷预测变化超过5%", "合规规则版本变化"],
        }
    if task_type == "long_term_locking":
        return common | {
            "headline": "当前锁定比例37.5%，建议分两批增持至50%，保留现货敞口。",
            "position_period": target.strftime("%Y-%m"),
            "position_basis": "预制场景已将不同合同按目标月折算为同周期电量",
            "forecast_demand_mwh": 60000,
            "locked_volume_mwh": 22500,
            "remaining_exposure_mwh": 37500,
            "current_lock_ratio": 37.5,
            "target_lock_ratio": 50.0,
            "recommended_add_volume_mwh": 7500,
            "price_ceiling": round(avg - 18, 1),
            "batches": [
                {"window": "本周滚动撮合", "volume_mwh": 4500, "max_price": round(avg - 22, 1)},
                {"window": "次周复核后", "volume_mwh": 3000, "max_price": round(avg - 15, 1)},
            ],
            "alternatives": ["基差收窄至-5元/MWh时暂停增持", "高温预警升级时目标比例提高至55%"],
            "stop_conditions": ["成交价超过价格上限", "月度负荷预测下修超过8%"],
        }
    if task_type == "real_time_alert":
        day_ahead = round(avg - 6, 1)
        snapshot = round(avg + 9, 1)
        return common | {
            "headline": "SQLite演示快照显示价格高于日前基准，建议先复核正式行情再小幅调整敞口。",
            "snapshot_type": "SQLite存储演示快照",
            "as_of": f"{target.isoformat()} 15:00:00+08:00",
            "day_ahead_price": day_ahead,
            "snapshot_price": snapshot,
            "spread": round(snapshot - day_ahead, 1),
            "windows": [
                {"window": "当前窗口", "action": "复核后小幅补仓", "volume_mwh": 24, "price_boundary": round(avg + 15, 1), "trigger": "正式行情确认价差仍为正"},
                {"window": "16:00-18:00", "action": "保持弹性", "volume_mwh": 18, "price_boundary": round(avg + 22, 1), "trigger": "负荷预测上修且光伏退坡"},
                {"window": "18:00-21:00", "action": "优先覆盖晚峰", "volume_mwh": 36, "price_boundary": round(peak + 18, 1), "trigger": "晚峰缺口经人工复核"},
            ],
            "stop_conditions": ["快照超过15分钟未刷新", "正式行情与演示快照方向相反", "合规规则版本变化", "当前持仓数据未人工确认"],
        }
    return common | {"headline": "政策问题由数据库RAG检索并由合规Agent解释。"}


def seed_demo_database(db_path: str | Path | None = None, base_date: date | None = None) -> dict[str, int | str]:
    """Idempotently seed a rolling demo window around ``base_date``."""
    path = Path(db_path) if db_path else get_db_path("powermind_demo.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    base = base_date or date.today()
    september_end = date(base.year, 9, 30)
    demo_end = max(base + timedelta(days=14), september_end)
    horizon_days = (demo_end - base).days + 1
    generated_at = datetime.combine(base, datetime.min.time()).isoformat()

    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for table in (
        "price_data",
        "forecasts",
        "weather_data",
        "market_factors",
        "bidding_records",
        "contracts",
        "basis_history",
        "knowledge_documents",
        "demo_scenarios",
    ):
        conn.execute(f"DELETE FROM {table} WHERE data_source LIKE 'demo-v%'")
    conn.execute(
        "DELETE FROM demo_metadata WHERE key IN "
        "('seed_version', 'base_date', 'province', 'demo_end_date', 'data_notice')"
    )

    price_rows = []
    for offset in range(-120, horizon_days):
        day = base + timedelta(days=offset)
        season = 388 + 20 * math.sin((day.timetuple().tm_yday - 172) / 365 * 2 * math.pi)
        weekend = 0.94 if day.weekday() >= 5 else 1.0
        for hour in range(24):
            price = max(20, season * weekend * _hour_factor(hour) + 9 * math.sin((offset + hour) * 0.71))
            load = 45500 * weekend + 6200 * _hour_factor(hour) + 400 * math.sin(offset)
            renewable = 3500 + (6200 * math.sin((hour - 6) / 12 * math.pi) if 6 <= hour <= 18 else 800)
            price_rows.append((PROVINCE, day.isoformat(), hour, round(price, 2), round(load, 1), round(max(500, renewable), 1), SEED_VERSION))
    conn.executemany("INSERT OR REPLACE INTO price_data VALUES (?,?,?,?,?,?,?)", price_rows)

    forecast_rows, weather_rows, factor_rows, scenario_rows = [], [], [], []
    for offset in range(horizon_days):
        day = base + timedelta(days=offset)
        hourly = []
        for hour in range(24):
            predicted = 402 * _hour_factor(hour) + 11 * math.sin((offset + hour) * 0.53)
            hourly.append(predicted)
            forecast_rows.append((PROVINCE, day.isoformat(), hour, round(predicted, 2), round(predicted - 42, 2), round(predicted + 48, 2), 0.86, "stored-demo-forecast-v1", generated_at, SEED_VERSION))
            summer_peak = date(day.year, 8, 5)
            seasonal_temp = 30 - abs((day - summer_peak).days) * 0.08
            temp = seasonal_temp + 5 * math.sin((hour - 8) / 24 * 2 * math.pi)
            cloud = 42 + 24 * math.sin((hour + offset) * 0.33)
            wind = 3.8 + 1.4 * math.sin((hour + 3) * 0.27)
            rain = 1.2 if offset % 11 in {3, 9} and 14 <= hour <= 18 else 0.0
            weather_rows.append((PROVINCE, day.isoformat(), hour, round(temp, 1), round(max(5, min(95, cloud)), 1), round(wind, 1), rain, SEED_VERSION))
        avg = sum(hourly) / 24
        peak = sum(hourly[18:21]) / 3
        valley = sum(hourly[0:6]) / 6
        factors = [
            ("supply_demand", "供需", 0.72, "偏紧", "晚峰负荷处于演示高位"),
            ("weather", "天气", 0.65, "看涨", "高温推动制冷负荷"),
            ("renewable", "新能源", -0.58, "看跌", "午间光伏出力抑制价格"),
            ("fuel", "燃料", 0.18, "中性", "演示煤价保持稳定"),
            ("policy", "政策", 0.10, "中性", "演示规则版本无变化"),
            ("load", "负荷", 0.69, "看涨", "统调负荷预计上行"),
            ("basis", "基差", -0.32, "适度增持", "中长期价格低于现货预测"),
            ("deviation", "偏差", 0.44, "关注", "偏差接近演示免考核边界"),
            ("liquidity", "流动性", 0.36, "正常", "撮合买卖盘数量均衡"),
        ]
        factor_rows.extend((PROVINCE, day.isoformat(), *f, SEED_VERSION) for f in factors)
        for task_type in ("daily_pulse", "bidding_advice", "long_term_locking", "real_time_alert", "policy_search"):
            payload = _scenario_payload(task_type, day, avg, peak, valley, hourly)
            scenario_rows.append((task_type, PROVINCE, day.isoformat(), payload["headline"], payload["headline"], json.dumps(payload, ensure_ascii=False), generated_at, SEED_VERSION))

    conn.executemany("INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?)", forecast_rows)
    conn.executemany("INSERT OR REPLACE INTO weather_data VALUES (?,?,?,?,?,?,?,?)", weather_rows)
    conn.executemany("INSERT OR REPLACE INTO market_factors VALUES (?,?,?,?,?,?,?,?)", factor_rows)
    conn.executemany("INSERT OR REPLACE INTO demo_scenarios VALUES (?,?,?,?,?,?,?,?)", scenario_rows)

    bids = []
    for offset in range(-45, horizon_days):
        day = base + timedelta(days=offset)
        bid = 370 + 16 * math.sin(offset * 0.41)
        clear = bid + 11 * math.sin(offset * 0.77)
        deviation = (clear - bid) / bid * 100
        bids.append((f"BID-{day:%Y%m%d}", PROVINCE, day.isoformat(), round(bid, 2), round(clear, 2), round(deviation, 2), 420 + abs(offset) % 7 * 12, "平衡", "通过" if abs(deviation) <= 5 else "复盘", SEED_VERSION))
    conn.executemany("INSERT OR REPLACE INTO bidding_records VALUES (?,?,?,?,?,?,?,?,?,?)", bids)

    contracts = [
        ("LT-YEAR-001", PROVINCE, "年度双边", f"{base.year}-01-01", f"{base.year}-12-31", 15000, 355.0, "演示发电企业A", "执行中", SEED_VERSION),
        ("LT-ROLL-003", PROVINCE, "滚动撮合", base.isoformat(), (base + timedelta(days=60)).isoformat(), 3000, 372.0, "演示交易对手B", "待交割", SEED_VERSION),
    ]
    months_to_end = (demo_end.year - base.year) * 12 + demo_end.month - base.month
    for months_ahead in range(months_to_end + 1):
        month = _monthly_date(base, -months_ahead)
        next_month = _monthly_date(month, -1)
        contracts.append((
            f"LT-MONTH-{month:%Y%m}",
            PROVINCE,
            "月度集中竞价",
            month.isoformat(),
            next_month.isoformat(),
            4500 + months_ahead * 300,
            round(366.0 + months_ahead * 4.5, 1),
            "演示月度市场",
            "执行中" if months_ahead == 0 else "待交割",
            SEED_VERSION,
        ))
    conn.executemany("INSERT OR REPLACE INTO contracts VALUES (?,?,?,?,?,?,?,?,?,?)", contracts)

    basis_rows = []
    for months_back in range(18, -months_to_end - 1, -1):
        month = _monthly_date(base, months_back)
        spot = 386 + 24 * math.sin(months_back * 0.52)
        contract = spot - 12 - 8 * math.sin(months_back * 0.31)
        basis = contract - spot
        basis_rows.append((PROVINCE, month.strftime("%Y-%m"), round(spot, 2), round(contract, 2), round(basis, 2), "增持" if basis < -15 else "观望", SEED_VERSION))
    conn.executemany("INSERT OR REPLACE INTO basis_history VALUES (?,?,?,?,?,?,?)", basis_rows)

    docs = [
        ("POL-SD-SPOT-001", "policy", PROVINCE, "山东电力现货交易规则要点（演示版）", "DEMO-SD-SPOT-001", "2026-01-01", "PowerMind演示政策库", "knowledge://demo/policy/spot", "山东,现货,日前,申报,限价", "演示规则：日前申报按24时段组织，价格建议区间为-80至1300元/MWh；所有量价建议必须由交易员在正式交易系统中人工确认。本条仅用于系统测试，不是正式政策原文。", SEED_VERSION),
        ("POL-SD-DEV-002", "policy", PROVINCE, "山东偏差考核规则要点（演示版）", "DEMO-SD-DEV-002", "2026-01-01", "PowerMind演示政策库", "knowledge://demo/policy/deviation", "山东,偏差,考核,免考核,5%", "演示规则：日内电量偏差绝对值不超过5%时记为免考核区间，超过部分进入复核并按演示系数1.2计算风险成本。极端天气和调度指令可进入人工免责复核。本条仅用于系统测试。", SEED_VERSION),
        ("POL-SD-DISC-003", "policy", PROVINCE, "交易信息披露与留痕要求（演示版）", "DEMO-SD-AUDIT-003", "2026-01-01", "PowerMind演示政策库", "knowledge://demo/policy/audit", "披露,证据,留痕,审计,人工确认", "演示要求：预测来源、策略依据、合规结论、人工确认状态应在同一任务编号下留痕；AI不得直接执行申报、成交或资金操作。", SEED_VERSION),
        ("POL-SD-LT-004", "policy", PROVINCE, "中长期合同风险控制要点（演示版）", "DEMO-SD-LT-004", "2026-01-01", "PowerMind演示政策库", "knowledge://demo/policy/long-term", "中长期,锁价,合约,敞口,基差", "演示规则：中长期锁价应披露当前锁定比例、剩余敞口、价格上限和停止条件；单次建议不得被系统自动提交。", SEED_VERSION),
        ("CASE-HEAT-001", "expert_case", PROVINCE, "高温晚峰申报复盘（演示案例）", "CASE-2025-HEAT", "2025-08-10", "PowerMind演示经验库", "knowledge://demo/case/heat", "高温,晚峰,负荷,申报", "相似演示案例中，高温使18-21时负荷上修6.8%，晚峰成交价较日均高31%；分段申报比全天统一价降低了未成交风险。", SEED_VERSION),
        ("CASE-SOLAR-002", "expert_case", PROVINCE, "午间光伏压价复盘（演示案例）", "CASE-2025-SOLAR", "2025-05-16", "PowerMind演示经验库", "knowledge://demo/case/solar", "光伏,午间,低价,补仓", "相似演示案例中，11-14时光伏出力高于预测12%，午间价格比日前预测低18%；保留午间补仓窗口改善了综合购电成本。", SEED_VERSION),
        ("CASE-BASIS-003", "expert_case", PROVINCE, "负基差分批锁价复盘（演示案例）", "CASE-2025-BASIS", "2025-11-03", "PowerMind演示经验库", "knowledge://demo/case/basis", "基差,锁价,分批,中长期", "相似演示案例中，基差低于-15元/MWh时分两批增持，避免了一次性锁仓后的价格回撤风险；第二批设置了明确价格上限。", SEED_VERSION),
        ("CASE-DEV-004", "expert_case", PROVINCE, "偏差边界复盘（演示案例）", "CASE-2025-DEV", "2025-09-21", "PowerMind演示经验库", "knowledge://demo/case/deviation", "偏差,考核,负荷预测,复核", "相似演示案例中，负荷预测在截止前上修4.6%，通过二次复核把最终偏差控制在演示免考核边界内。", SEED_VERSION),
    ]
    conn.executemany("INSERT OR REPLACE INTO knowledge_documents VALUES (?,?,?,?,?,?,?,?,?,?,?)", docs)

    metadata = {
        "seed_version": SEED_VERSION,
        "base_date": base.isoformat(),
        "province": PROVINCE,
        "demo_end_date": demo_end.isoformat(),
        "data_notice": "全部为展示和测试假数据，不得用于真实交易决策",
    }
    conn.executemany("INSERT OR REPLACE INTO demo_metadata(key,value) VALUES (?,?)", metadata.items())
    conn.commit()

    counts = {}
    for table in ("price_data", "forecasts", "weather_data", "market_factors", "bidding_records", "contracts", "basis_history", "knowledge_documents", "demo_scenarios"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return {"db_path": str(path), **counts}


def ensure_demo_database(db_path: str | Path | None = None, base_date: date | None = None) -> Path:
    """Ensure today's rolling data exists and return the database path."""
    result = seed_demo_database(db_path=db_path, base_date=base_date)
    return Path(str(result["db_path"]))


if __name__ == "__main__":
    print(json.dumps(seed_demo_database(), ensure_ascii=False, indent=2))
