#!/usr/bin/env python3
"""
市场数据 RAG 导入器 — 将 SQLite 中的结构化市场数据生成语义摘要，存入 ChromaDB。

★ 核心问题: 当前 ChromaDB 只有政策+专家心得，没有市场行情数据。
  当用户问 "去年7月电价多少" 或 "光伏出力最高的月份" 时，RAG 无法检索到答案。

解决方案:
  1. 从 SQLite price_data/load_data/weather_data 读取原始数据
  2. 生成日/月级别的市场快照摘要文本
  3. 存入 ChromaDB, data_type="market_snapshot", province 标注
  4. RAG 可以按省+日期+特征进行语义检索

数据来源 (用户补齐后):
  - price_data 表: 日前出清电价 (96点/天)
  - load_data 表: 直调负荷/风电/光伏/核电/联络线 (15min)
  - weather_data 表: 温度/云量/风速/辐射 (小时级)

CSV 导入方式:
  1. 先把 CSV 导入 SQLite → import_all_data.py 或直接 INSERT
  2. 再运行本脚本 → 从 SQLite 生成 ChromaDB 摘要
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")
COLLECTION_NAME = "power_market_kb"

# 项目A的SQLite数据库路径
DB_PATH = (
    Path(r"D:\GitHub_work\china-power-market-agent-main\power_market_agent")
    / "knowledge_base"
    / "knowledge_base.db"
)


def get_collection():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def generate_daily_summaries(
    province: str,
    start_date: str,
    end_date: str,
) -> int:
    """
    从 SQLite 读取每日数据, 生成日级市场快照摘要, 写入 ChromaDB。

    每一条摘要格式:
      "{province}省日前出清电价 - {date}
       日均价: {avg_price} 元/MWh
       峰时均价 (18-21): {peak_avg} 元/MWh
       谷时均价 (00-06): {valley_avg} 元/MWh
       最高价: {max_price} 元/MWh ({max_hour}:00)
       最低价: {min_price} 元/MWh ({min_hour}:00)
       峰谷价差: {spread} 元/MWh
       统调负荷: {load} MW
       风电出力: {wind} MW
       光伏出力: {solar} MW
       温度: {temp}°C | 降水: {rain}mm | 风速: {ws}km/h
       市场特征: {features}"
    """
    if not DB_PATH.exists():
        print(f"  ⚠️ SQLite 数据库不存在: {DB_PATH}")
        print("  请先将 CSV 数据导入 SQLite")
        return 0

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    collection = get_collection()

    # 获取省份ID
    pid_row = conn.execute(
        "SELECT id FROM provinces WHERE name=?", (province,)
    ).fetchone()
    pid = pid_row["id"] if pid_row else None

    imported = 0
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end_dt - start_dt).days + 1

    for i in range(days):
        date_str = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")

        # 从 price_data 读电价
        price_rows = conn.execute(
            "SELECT hour, day_ahead_price FROM price_data "
            "WHERE province=? AND date=? ORDER BY hour",
            (province, date_str),
        ).fetchall()

        # 从 load_data 读负荷
        load_rows = conn.execute(
            "SELECT hour, load_actual, wind_actual, solar_actual, "
            "nuclear_output, tie_line_import "
            "FROM load_data WHERE province=? AND date=? ORDER BY hour",
            (province, date_str),
        ).fetchall()

        # 从 weather_data 读天气
        weather_rows = conn.execute(
            "SELECT AVG(temperature_2m) as avg_temp, "
            "SUM(precipitation) as total_rain, "
            "AVG(wind_speed_10m) as avg_wind, "
            "AVG(cloud_cover) as avg_cloud "
            "FROM weather_data w JOIN cities c ON w.city_id=c.id "
            "JOIN provinces p ON c.province_id=p.id "
            "WHERE p.name=? AND w.date=?",
            (province, date_str),
        ).fetchone()

        # 如果没有任何数据, 跳过
        if not price_rows and not load_rows:
            continue

        # 计算电价统计
        prices = [r["day_ahead_price"] for r in price_rows if r["day_ahead_price"]]
        if prices:
            avg_price = sum(prices) / len(prices)
            max_price = max(prices)
            min_price = min(prices)
            peak_prices = [r["day_ahead_price"] for r in price_rows if 18 <= r["hour"] <= 21 and r["day_ahead_price"]]
            valley_prices = [r["day_ahead_price"] for r in price_rows if 0 <= r["hour"] <= 6 and r["day_ahead_price"]]
            peak_avg = sum(peak_prices) / len(peak_prices) if peak_prices else avg_price
            valley_avg = sum(valley_prices) / len(valley_prices) if valley_prices else avg_price
            max_hour = next((r["hour"] for r in price_rows if r["day_ahead_price"] == max_price), 0)
            min_hour = next((r["hour"] for r in price_rows if r["day_ahead_price"] == min_price), 0)
            spread = max_price - min_price
        else:
            avg_price = max_price = min_price = peak_avg = valley_avg = spread = 0
            max_hour = min_hour = 0

        # 计算负荷统计
        loads = [r["load_actual"] for r in load_rows if r["load_actual"]]
        winds = [r["wind_actual"] for r in load_rows if r["wind_actual"]]
        solars = [r["solar_actual"] for r in load_rows if r["solar_actual"]]
        avg_load = sum(loads) / len(loads) if loads else 0
        avg_wind = sum(winds) / len(winds) if winds else 0
        avg_solar = sum(solars) / len(solars) if solars else 0

        # 天气
        temp = weather_rows["avg_temp"] if weather_rows and weather_rows["avg_temp"] else 0
        rain = weather_rows["total_rain"] if weather_rows and weather_rows["total_rain"] else 0
        ws = weather_rows["avg_wind"] if weather_rows and weather_rows["avg_wind"] else 0
        cloud = weather_rows["avg_cloud"] if weather_rows and weather_rows["avg_cloud"] else 0

        # 生成市场特征标签
        features = []
        if avg_price > 500:
            features.append("高电价日")
        elif avg_price < 200:
            features.append("低电价日")
        if spread > 150:
            features.append("峰谷价差大")
        if avg_solar > 8000:
            features.append("光伏满发")
        if avg_wind > 8000:
            features.append("风电满发")
        if temp > 32:
            features.append("高温日")
        elif temp < 5:
            features.append("低温日")
        if rain > 5:
            features.append("降雨日")
        if avg_price > 800:
            features.append("极端高价")
        elif avg_price < 50:
            features.append("极端低价/负价")
        if not features:
            features.append("正常交易日")

        # 构造摘要文本
        doc_text = (
            f"{province}省日前出清电价 - {date_str}\n"
            f"日均价: {avg_price:.1f} 元/MWh\n"
            f"峰时均价 (18-21时): {peak_avg:.1f} 元/MWh\n"
            f"谷时均价 (00-06时): {valley_avg:.1f} 元/MWh\n"
            f"最高价: {max_price:.1f} 元/MWh ({max_hour}:00)\n"
            f"最低价: {min_price:.1f} 元/MWh ({min_hour}:00)\n"
            f"峰谷价差: {spread:.1f} 元/MWh\n"
            f"统调负荷: {avg_load:.0f} MW\n"
            f"风电出力: {avg_wind:.0f} MW\n"
            f"光伏出力: {avg_solar:.0f} MW\n"
            f"温度: {temp:.1f}°C | 降水: {rain:.1f}mm | 风速: {ws:.1f}km/h | 云量: {cloud:.0f}%\n"
            f"市场特征: {', '.join(features)}"
        )

        doc_id = f"market-snapshot-{province}-{date_str}"

        try:
            collection.add(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{
                    "source": "market_data_import",
                    "data_type": "market_snapshot",
                    "province": province,
                    "date": date_str,
                    "avg_price": round(avg_price, 1),
                    "peak_avg": round(peak_avg, 1),
                    "valley_avg": round(valley_avg, 1),
                    "spread": round(spread, 1),
                    "avg_load": round(avg_load, 0),
                    "avg_wind": round(avg_wind, 0),
                    "avg_solar": round(avg_solar, 0),
                    "temperature": round(temp, 1),
                    "features": ",".join(features),
                }],
            )
            imported += 1
        except Exception as e:
            print(f"  ⚠️ {date_str}: {e}")

    conn.close()
    print(f"  ✅ 导入 {imported} 条日级市场快照 ({province} {start_date}~{end_date})")
    return imported


def generate_monthly_summaries(
    province: str,
    year: int,
) -> int:
    """生成月度汇总摘要。"""
    collection = get_collection()
    imported = 0

    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        if month == 12:
            end = f"{year}-12-31"
        else:
            end = f"{year}-{month+1:02d}-01"

        try:
            results = collection.get(
                where={
                    "$and": [
                        {"data_type": "market_snapshot"},
                        {"province": province},
                    ]
                },
                limit=200,
                include=["metadatas"],
            )
            month_data = [
                m for m in (results.get("metadatas") or [])
                if m and m.get("date", "").startswith(f"{year}-{month:02d}")
            ]
        except Exception:
            month_data = []

        if not month_data:
            continue

        prices = [m.get("avg_price", 0) for m in month_data if m.get("avg_price")]
        spreads = [m.get("spread", 0) for m in month_data if m.get("spread")]
        features_list = []
        for m in month_data:
            f = m.get("features", "")
            if f:
                features_list.extend(f.split(","))

        from collections import Counter
        top_features = Counter(features_list).most_common(3)

        month_avg_price = sum(prices) / len(prices) if prices else 0
        month_avg_spread = sum(spreads) / len(spreads) if spreads else 0

        doc_text = (
            f"{province}省 {year}年{month}月 电力市场月报摘要\n"
            f"月均日前出清电价: {month_avg_price:.1f} 元/MWh\n"
            f"月均峰谷价差: {month_avg_spread:.1f} 元/MWh\n"
            f"有效交易日: {len(month_data)} 天\n"
            f"高频市场特征: {', '.join(f'{k}({v}天)' for k, v in top_features)}"
        )

        doc_id = f"market-monthly-{province}-{year}-{month:02d}"

        try:
            collection.add(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{
                    "source": "market_data_import",
                    "data_type": "market_monthly_summary",
                    "province": province,
                    "year": year,
                    "month": month,
                    "avg_price": round(month_avg_price, 1),
                    "avg_spread": round(month_avg_spread, 1),
                    "trading_days": len(month_data),
                }],
            )
            imported += 1
        except Exception as e:
            print(f"  ⚠️ {year}-{month:02d}: {e}")

    return imported


def import_csv_price_to_sqlite(csv_path: str, province: str):
    """
    将 CSV 格式的日前出清电价导入 SQLite price_data 表。
    这是两步流程的第 1 步: CSV → SQLite
    第 2 步: 运行 generate_daily_summaries → ChromaDB
    """
    import csv
    if not DB_PATH.exists():
        print("  ⚠️ SQLite 数据库不存在")
        return 0

    conn = sqlite3.connect(str(DB_PATH))

    # 确保表存在
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            province TEXT NOT NULL,
            date TEXT NOT NULL,
            hour INTEGER,
            day_ahead_price REAL,
            day_ahead_volume REAL,
            real_time_price REAL,
            UNIQUE(province, date, hour)
        );
        CREATE TABLE IF NOT EXISTS load_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            province TEXT NOT NULL,
            date TEXT NOT NULL,
            hour INTEGER,
            load_actual REAL,
            load_forecast REAL,
            wind_actual REAL,
            wind_forecast REAL,
            solar_actual REAL,
            solar_forecast REAL,
            nuclear_output REAL,
            tie_line_import REAL,
            UNIQUE(province, date, hour)
        );
    """)

    imported = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row.get("date", "")
            hour_str = row.get("hour", "")
            price = row.get("day_ahead_price", "")
            volume = row.get("day_ahead_volume", "")
            rt_price = row.get("real_time_price", "")

            if not date_str or not price:
                continue

            try:
                hour = int(hour_str) if hour_str else None
                conn.execute(
                    """INSERT OR REPLACE INTO price_data
                       (province, date, hour, day_ahead_price, day_ahead_volume, real_time_price)
                       VALUES (?,?,?,?,?,?)""",
                    (province, date_str, hour,
                     float(price) if price else None,
                     float(volume) if volume else None,
                     float(rt_price) if rt_price else None),
                )
                imported += 1
            except Exception as e:
                print(f"  ⚠️ Row {imported}: {e}")

    conn.commit()
    conn.close()
    return imported


# ── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="市场数据 RAG 导入器")
    parser.add_argument("--province", "-p", default="山东", help="省份")
    parser.add_argument("--start-date", default="2025-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-12-31", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--csv", help="CSV文件路径 (先导入SQLite)")
    parser.add_argument("--csv-type", default="price", choices=["price", "load"],
                        help="CSV类型: price(电价) / load(负荷)")
    parser.add_argument("--monthly", action="store_true", help="生成月度汇总")
    parser.add_argument("--year", type=int, default=2025, help="月度汇总年份")
    args = parser.parse_args()

    print("=" * 60)
    print("  市场数据 RAG 导入器")
    print("=" * 60)

    # Step 1: CSV → SQLite
    if args.csv:
        if args.csv_type == "price":
            count = import_csv_price_to_sqlite(args.csv, args.province)
            print(f"\n📥 CSV → SQLite: {count} 条电价记录")
        elif args.csv_type == "load":
            print("  负荷数据导入待实现 (格式与price类似)")
        print("  下一步: 运行 --generate 生成 ChromaDB 摘要")

    # Step 2: SQLite → ChromaDB (日摘要)
    count = generate_daily_summaries(args.province, args.start_date, args.end_date)
    print(f"\n📤 SQLite → ChromaDB: {count} 条日级市场快照")

    # Step 3: 月度汇总
    if args.monthly:
        m_count = generate_monthly_summaries(args.province, args.year)
        print(f"📊 月度汇总: {m_count} 个月")

    # 验证
    collection = get_collection()
    try:
        results = collection.get(
            where={"data_type": "market_snapshot"},
            limit=3,
            include=["documents", "metadatas"],
        )
        print(f"\n🔍 验证检索: 已有 {len(results.get('ids',[]))} 条可检索 (显示3条样本)")
        for i, doc_id in enumerate(results.get("ids", [])[:3]):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            doc = results["documents"][i][:200] if results.get("documents") else ""
            print(f"  [{doc_id}] {meta.get('province','?')} {meta.get('date','?')} | avg={meta.get('avg_price','?')}元 | {meta.get('features','')}")
    except Exception as e:
        print(f"  ⚠️ 检索验证失败: {e}")
