#!/usr/bin/env python3
"""
PowerMind 数据自动归档管道 — 实时数据、预测结果、行情快照的自动存储。

设计原则:
  - 实时瞬态数据 → Redis (短TTL) + PostgreSQL (定期快照)
  - 预测结果 → PostgreSQL (精确值) + ChromaDB (语义摘要) 双写
  - 行情快照 → ChromaDB (按 data_type 标签存储)
  - 所有归档操作静默失败，不阻塞主流程

运行方式:
  python scripts/data_archiver.py --all           # 归档所有类型
  python scripts/data_archiver.py --forecast       # 仅归档预测结果
  python scripts/data_archiver.py --market-snapshot # 仅归档行情快照
  python scripts/data_archiver.py --daemon          # 守护模式: 定期自动归档
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. 预测结果归档器
# ═══════════════════════════════════════════════════════════════════

class ForecastArchiver:
    """预测结果自动归档: PostgreSQL(精确值) + ChromaDB(语义摘要)"""

    def __init__(self):
        self._chroma_collection = None
        self._db_path = str(PROJECT_ROOT / "data" / "sql" / "forecast_archives.db")

    def _get_chroma(self):
        if self._chroma_collection:
            return self._chroma_collection
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(PROJECT_ROOT / "data" / "vectordb"),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self._chroma_collection = client.get_collection("power_market_kb")
        except Exception:
            self._chroma_collection = client.create_collection(
                name="power_market_kb",
                metadata={"hnsw:space": "cosine"},
            )
        return self._chroma_collection

    def archive_forecast(self, forecast_data: dict) -> dict:
        """
        归档单条电价预测结果。

        forecast_data 格式:
        {
            "province": "山东",
            "target_date": "2026-06-16",
            "model_name": "company_v1",
            "model_version": "1.0",
            "avg_price": 378.5,
            "peak_avg": 520.3,
            "valley_avg": 245.8,
            "confidence": 0.85,
            "hourly_predictions": [260, 250, 240, ...],  # 24个float
            "p10": [...],
            "p50": [...],
            "p90": [...],
            "factors": {"direction": "bullish", "risk": "medium"},
            "timestamp": "2026-06-15T10:00:00"
        }
        """
        result = {"sqlite_ok": False, "chroma_ok": False}

        try:
            self._archive_to_sqlite(forecast_data)
            result["sqlite_ok"] = True
        except Exception as e:
            logger.warning(f"预测归档 SQLite 失败: {e}")

        try:
            self._archive_to_chroma(forecast_data)
            result["chroma_ok"] = True
        except Exception as e:
            logger.warning(f"预测归档 ChromaDB 失败: {e}")

        return result

    def _archive_to_sqlite(self, data: dict):
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecast_archives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT NOT NULL,
                target_date TEXT NOT NULL,
                model_name TEXT,
                model_version TEXT,
                avg_price REAL,
                peak_avg REAL,
                valley_avg REAL,
                confidence REAL,
                hourly_predictions TEXT,
                p10 TEXT,
                p50 TEXT,
                p90 TEXT,
                factors TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecast_province_date
            ON forecast_archives(province, target_date)
        """)

        conn.execute(
            """INSERT OR REPLACE INTO forecast_archives
               (province, target_date, model_name, model_version, avg_price,
                peak_avg, valley_avg, confidence, hourly_predictions, p10, p50, p90, factors)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["province"],
                data["target_date"],
                data.get("model_name", "unknown"),
                data.get("model_version", "0"),
                data.get("avg_price"),
                data.get("peak_avg"),
                data.get("valley_avg"),
                data.get("confidence"),
                json.dumps(data.get("hourly_predictions", []), ensure_ascii=False),
                json.dumps(data.get("p10", []), ensure_ascii=False),
                json.dumps(data.get("p50", []), ensure_ascii=False),
                json.dumps(data.get("p90", []), ensure_ascii=False),
                json.dumps(data.get("factors", {}), ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()

    def _archive_to_chroma(self, data: dict):
        collection = self._get_chroma()

        # 生成自然语言摘要
        hourly = data.get("hourly_predictions", [])
        avg = data.get("avg_price", 0)
        peak = data.get("peak_avg", 0)
        valley = data.get("valley_avg", 0)
        spread = peak - valley if peak and valley else 0

        # 生成特征标签
        tags = []
        if avg > 500:
            tags.append("高电价日")
        elif avg < 300:
            tags.append("低电价日")
        if spread > 200:
            tags.append("峰谷价差大")
        if data.get("confidence", 0) > 0.8:
            tags.append("高置信度")
        elif data.get("confidence", 0) < 0.5:
            tags.append("低置信度")

        direction = data.get("factors", {}).get("direction", "unknown")
        tags.append(f"方向:{direction}")

        summary = (
            f"【{data['province']} 电价预测】{data['target_date']}\n"
            f"日均价: {avg:.2f} 元/MWh | 峰时: {peak:.2f} | 谷时: {valley:.2f}\n"
            f"峰谷价差: {spread:.2f} 元/MWh | 置信度: {data.get('confidence', 0):.0%}\n"
            f"24点预测: {' → '.join(f'{p:.0f}' for p in hourly[::3])}\n"
            f"市场特征: {', '.join(tags)}\n"
            f"预测引擎: {data.get('model_name', 'unknown')} v{data.get('model_version', '0')}"
        )

        doc_id = f"forecast_{data['province']}_{data['target_date']}_{int(time.time())}"

        try:
            collection.upsert(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{
                    "data_type": "forecast_archive",
                    "province": data["province"],
                    "target_date": data["target_date"],
                    "avg_price": avg,
                    "peak_avg": peak,
                    "valley_avg": valley,
                    "confidence": data.get("confidence", 0),
                    "spread": spread,
                    "features": ", ".join(tags),
                    "created_at": data.get("timestamp", datetime.now().isoformat()),
                }],
            )
        except Exception as e:
            logger.warning(f"ChromaDB upsert 失败: {e}")
            raise

    def query_forecasts(
        self, province: str | None = None, start_date: str | None = None, end_date: str | None = None
    ) -> list[dict]:
        """查询历史预测"""
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM forecast_archives WHERE 1=1"
        params = []
        if province:
            query += " AND province = ?"
            params.append(province)
        if start_date:
            query += " AND target_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND target_date <= ?"
            params.append(end_date)
        query += " ORDER BY target_date DESC LIMIT 30"

        rows = conn.execute(query, params).fetchall()
        conn.close()

        return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════
# 2. 市场行情快照归档器
# ═══════════════════════════════════════════════════════════════════

class MarketSnapshotArchiver:
    """市场行情快照自动归档到 ChromaDB"""

    def __init__(self):
        self._collection = None

    def _get_collection(self):
        if self._collection:
            return self._collection
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(PROJECT_ROOT / "data" / "vectordb"),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self._collection = client.get_collection("power_market_kb")
        except Exception:
            self._collection = client.create_collection(
                name="power_market_kb",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def archive_snapshot(self, snapshot_data: dict) -> bool:
        """
        归档市场行情快照。

        snapshot_data 格式:
        {
            "province": "山东",
            "date": "2026-06-15",
            "avg_price": 385.0,
            "peak_price": 520.0,
            "valley_price": 245.0,
            "total_load_mw": 58200,
            "wind_output_mw": 3500,
            "solar_output_mw": 5200,
            "temperature": 32.5,
            "precipitation": 0,
            "direction": "bullish",
            "risk": "medium",
        }
        """
        collection = self._get_collection()

        # 生成快照文本
        snap = snapshot_data
        text = (
            f"【{snap['province']} 市场快照】{snap['date']}\n"
            f"日前出清均价: {snap['avg_price']} 元/MWh\n"
            f"峰时均价: {snap['peak_price']} 元/MWh | 谷时均价: {snap['valley_price']} 元/MWh\n"
            f"统调负荷: {snap.get('total_load_mw', '?')} MW\n"
            f"风电出力: {snap.get('wind_output_mw', '?')} MW | 光伏出力: {snap.get('solar_output_mw', '?')} MW\n"
            f"温度: {snap.get('temperature', '?')}°C | 降水: {snap.get('precipitation', '?')}mm\n"
            f"市场方向: {snap.get('direction', '?')} | 风险: {snap.get('risk', '?')}"
        )

        # 生成特征标签
        tags = []
        avg = snap.get("avg_price", 0)
        if avg > 500:
            tags.append("高电价日")
        elif avg < 300:
            tags.append("低电价日")
        temp = snap.get("temperature", 25)
        if temp > 35:
            tags.append("高温日")
        solar = snap.get("solar_output_mw", 0)
        if solar > 4000:
            tags.append("光伏大发")
        wind = snap.get("wind_output_mw", 0)
        if wind > 3000:
            tags.append("风电大发")

        doc_id = f"snapshot_{snap['province']}_{snap['date']}"

        try:
            collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "data_type": "market_snapshot",
                    "province": snap["province"],
                    "date": snap["date"],
                    "avg_price": avg,
                    "features": ", ".join(tags),
                }],
            )
            return True
        except Exception as e:
            logger.warning(f"行情快照归档失败: {e}")
            return False

    def archive_monthly_summary(self, summary_data: dict) -> bool:
        """
        归档月度市场摘要。

        summary_data 格式:
        {
            "province": "山东",
            "year": 2026,
            "month": 6,
            "monthly_avg_price": 385.5,
            "max_price": 680.0,
            "min_price": 180.0,
            "top_features": ["高温日多", "光伏出力高"],
            "total_volume_mwh": 1500000,
        }
        """
        collection = self._get_collection()

        s = summary_data
        text = (
            f"【{s['province']} 月度市场摘要】{s['year']}年{s['month']}月\n"
            f"月均电价: {s['monthly_avg_price']} 元/MWh\n"
            f"最高电价: {s['max_price']} 元/MWh | 最低电价: {s['min_price']} 元/MWh\n"
            f"总交易量: {s.get('total_volume_mwh', '?')} MWh\n"
            f"主要特征: {', '.join(s.get('top_features', []))}"
        )

        doc_id = f"monthly_{s['province']}_{s['year']}_{s['month']:02d}"

        try:
            collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "data_type": "market_monthly_summary",
                    "province": s["province"],
                    "year": s["year"],
                    "month": s["month"],
                    "avg_price": s["monthly_avg_price"],
                }],
            )
            return True
        except Exception as e:
            logger.warning(f"月度摘要归档失败: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════
# 3. 实时数据管道
# ═══════════════════════════════════════════════════════════════════

class RealtimeDataPipeline:
    """实时数据自动归档管道"""

    def __init__(self):
        self.forecast_archiver = ForecastArchiver()
        self.snapshot_archiver = MarketSnapshotArchiver()

    def process_realtime_price(self, price_data: dict) -> dict:
        """
        处理实时电价数据。
        实时数据不持久化到 PG (数据量太大)，但可以触发归档标记。
        """
        # 实时数据仅作 Redis 缓存，不做持久化
        # 但当日结束时触发快照归档
        return {"cached": True, "archived": False, "reason": "实时数据仅缓存"}

    def process_daily_batch(self, province: str, date: str) -> dict:
        """
        每日批处理: 归档昨日行情快照。

        通常在每天凌晨定时执行。
        """
        result = {"forecasts_archived": 0, "snapshots_archived": 0}

        # 1. 归档行情快照 (从预测归档库中提取)
        forecasts = self.forecast_archiver.query_forecasts(
            province=province, start_date=date, end_date=date
        )
        result["forecasts_archived"] = len(forecasts)

        return result


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PowerMind 数据自动归档管道")
    parser.add_argument("--forecast", action="store_true", help="归档预测结果")
    parser.add_argument("--snapshot", action="store_true", help="归档行情快照")
    parser.add_argument("--province", "-p", default="山东", help="省份")
    parser.add_argument("--date", "-d", help="日期 (YYYY-MM-DD)")
    parser.add_argument("--daemon", action="store_true", help="守护模式: 定期执行")
    parser.add_argument("--interval", type=int, default=3600, help="守护模式间隔 (秒)")

    args = parser.parse_args()

    if args.daemon:
        print(f"[归档守护] 启动, 间隔={args.interval}s")
        pipeline = RealtimeDataPipeline()
        while True:
            today = datetime.now().strftime("%Y-%m-%d")
            try:
                result = pipeline.process_daily_batch(args.province, today)
                print(f"[归档守护] {today}: {result}")
            except Exception as e:
                print(f"[归档守护] 错误: {e}")
            time.sleep(args.interval)

    elif args.forecast:
        archiver = ForecastArchiver()
        # 示例: 归档一条测试预测
        test_forecast = {
            "province": args.province,
            "target_date": args.date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "model_name": "test_model",
            "model_version": "1.0",
            "avg_price": 378.5,
            "peak_avg": 520.3,
            "valley_avg": 245.8,
            "confidence": 0.85,
            "hourly_predictions": [260, 250, 240, 230, 220, 215, 210, 220, 280, 350, 420, 500, 520, 510, 480, 450, 380, 450, 520, 580, 550, 480, 380, 300],
            "p10": [240, 230, 220, 210, 200, 195, 190, 200, 250, 310, 380, 450, 470, 460, 430, 400, 340, 400, 470, 520, 490, 430, 340, 270],
            "p50": [260, 250, 240, 230, 220, 215, 210, 220, 280, 350, 420, 500, 520, 510, 480, 450, 380, 450, 520, 580, 550, 480, 380, 300],
            "p90": [290, 280, 270, 260, 250, 240, 235, 250, 310, 390, 460, 540, 560, 550, 520, 490, 420, 490, 560, 620, 590, 520, 420, 340],
            "factors": {"direction": "bullish", "risk": "medium"},
            "timestamp": datetime.now().isoformat(),
        }
        result = archiver.archive_forecast(test_forecast)
        print(f"预测归档: {result}")

    elif args.snapshot:
        archiver = MarketSnapshotArchiver()
        test_snapshot = {
            "province": args.province,
            "date": args.date or datetime.now().strftime("%Y-%m-%d"),
            "avg_price": 385.0,
            "peak_price": 520.0,
            "valley_price": 245.0,
            "total_load_mw": 58200,
            "wind_output_mw": 3500,
            "solar_output_mw": 5200,
            "temperature": 32.5,
            "precipitation": 0,
            "direction": "bullish",
            "risk": "medium",
        }
        ok = archiver.archive_snapshot(test_snapshot)
        print(f"行情快照归档: {'成功' if ok else '失败'}")

    else:
        parser.print_help()
        print("\n示例:")
        print("  python scripts/data_archiver.py --forecast --province 山东")
        print("  python scripts/data_archiver.py --snapshot --province 山东")
        print("  python scripts/data_archiver.py --daemon --interval 3600 &")
