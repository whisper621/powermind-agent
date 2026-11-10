#!/usr/bin/env python3
"""
PowerMind 数据流入管道 —— API 获取的结构化数据直接写入 SQL/ChromaDB。

设计原则:
  - API 获取的数值数据 (电价/负荷/天气/预测) → 直接写 SQL，不保留中间 CSV 文件
  - API 获取的文本数据 (政策更新) → 写 ChromaDB + source_docs/ 留档
  - 实时流数据 → Redis 缓存 (不持久化)
  - 人类上传的文档 → source_docs/ → AutoImporter → ChromaDB

多省份支持:
  - SQL 所有表均有 `province` 列，通过 WHERE province='山东' 精准查询
  - ChromaDB 所有文档均有 `province` metadata，通过 where={"province":"山东"} 过滤
  - source_docs/ 按省份分子目录 (仅人类上传的文档)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    数据流入管道 —— 所有 API 爬取的数据通过此管道进入知识库。

    用法:
      pipeline = DataIngestionPipeline()
      pipeline.ingest_price_point("山东", "2026-06-15", 14, 385.5, {"load_mw": 52000})
      pipeline.ingest_forecast("山东", {...})
      pipeline.ingest_weather("山东", "2026-06-15", {...})
      pipeline.ingest_contract("山东", {...})
    """

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db_path = DATA_ROOT / "sql" / "poweragent.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))
            self._conn.row_factory = sqlite3.Row
            self._ensure_tables()
        return self._conn

    def _ensure_tables(self):
        """建表 (所有表均含 province, created_at 列)"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT NOT NULL,
                date TEXT NOT NULL,
                hour INTEGER NOT NULL CHECK(hour BETWEEN 0 AND 23),
                price_rmb_mwh REAL,
                load_mw REAL,
                wind_output_mw REAL,
                solar_output_mw REAL,
                data_source TEXT DEFAULT 'api',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(province, date, hour)
            );
            CREATE INDEX IF NOT EXISTS idx_price_province_date ON price_data(province, date);

            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT NOT NULL,
                target_date TEXT NOT NULL,
                model_name TEXT,
                model_version TEXT,
                avg_price REAL,
                peak_avg REAL,
                valley_avg REAL,
                spread REAL,
                confidence REAL,
                direction TEXT,
                risk TEXT,
                hourly_predictions TEXT,
                p10 TEXT, p50 TEXT, p90 TEXT,
                data_source TEXT DEFAULT 'api',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(province, target_date, model_name)
            );
            CREATE INDEX IF NOT EXISTS idx_forecast_province_date ON forecasts(province, target_date);

            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT NOT NULL,
                date TEXT NOT NULL,
                hour INTEGER NOT NULL CHECK(hour BETWEEN 0 AND 23),
                temperature_2m REAL,
                cloud_cover REAL,
                wind_speed_10m REAL,
                shortwave_radiation REAL,
                relative_humidity REAL,
                precipitation_mm REAL,
                data_source TEXT DEFAULT 'open_meteo',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(province, date, hour)
            );
            CREATE INDEX IF NOT EXISTS idx_weather_province_date ON weather_data(province, date);

            CREATE TABLE IF NOT EXISTS contracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT NOT NULL,
                contract_id TEXT NOT NULL,
                contract_type TEXT,
                volume_mwh REAL,
                price_rmb_mwh REAL,
                start_date TEXT,
                end_date TEXT,
                counterparty TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(province, contract_id)
            );

            CREATE TABLE IF NOT EXISTS bidding_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                bid_avg_price REAL,
                clearing_avg_price REAL,
                deviation_pct REAL,
                deviation_status TEXT,
                volume_mwh REAL,
                strategy_type TEXT,
                notes TEXT,
                data_source TEXT DEFAULT 'api',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS rolling_match_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                contract_month TEXT,
                volume_mwh REAL,
                deal_price_rmb_mwh REAL,
                spot_reference REAL,
                basis REAL,
                data_source TEXT DEFAULT 'api',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS basis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT NOT NULL,
                month TEXT NOT NULL,
                spot_monthly_avg REAL,
                contract_price REAL,
                basis REAL,
                basis_pct REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(province, month)
            );
        """)
        self.conn.commit()

    # ══ 数据流入方法 ══

    def ingest_price_point(self, province: str, date: str, hour: int, price: float,
                           extra: dict | None = None) -> bool:
        """摄入单个小时电价数据点 (API 实时推送)"""
        extra = extra or {}
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO price_data
                   (province, date, hour, price_rmb_mwh, load_mw, wind_output_mw, solar_output_mw)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (province, date, hour, price,
                 extra.get("load_mw"), extra.get("wind_output_mw"), extra.get("solar_output_mw"))
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[Ingest] price_point 失败: {e}")
            return False

    def ingest_price_batch(self, province: str, records: list[dict]) -> int:
        """批量摄入电价数据"""
        count = 0
        for r in records:
            if self.ingest_price_point(
                province, r["date"], r["hour"], r["price_rmb_mwh"],
                {k: r.get(k) for k in ("load_mw", "wind_output_mw", "solar_output_mw")}
            ):
                count += 1
        return count

    def ingest_forecast(self, province: str, forecast: dict) -> bool:
        """摄入单条电价预测 (双写: SQL + ChromaDB 语义摘要)"""
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO forecasts
                   (province, target_date, model_name, model_version, avg_price,
                    peak_avg, valley_avg, spread, confidence, direction, risk,
                    hourly_predictions, p10, p50, p90)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (province, forecast["target_date"],
                 forecast.get("model_name", "api"), forecast.get("model_version", "1.0"),
                 forecast.get("avg_price"), forecast.get("peak_avg"),
                 forecast.get("valley_avg"), forecast.get("spread"),
                 forecast.get("confidence"), forecast.get("direction"),
                 forecast.get("risk"),
                 json.dumps(forecast.get("hourly_predictions", [])),
                 json.dumps(forecast.get("p10", [])),
                 json.dumps(forecast.get("p50", [])),
                 json.dumps(forecast.get("p90", [])))
            )
            self.conn.commit()

            # ChromaDB 语义摘要
            self._chroma_upsert_forecast(province, forecast)
            return True
        except Exception as e:
            logger.error(f"[Ingest] forecast 失败: {e}")
            return False

    def ingest_weather(self, province: str, date: str, weather_data: list[dict]) -> int:
        """批量摄入天气数据 (24小时逐时)"""
        count = 0
        for w in weather_data:
            try:
                self.conn.execute(
                    """INSERT OR REPLACE INTO weather_data
                       (province, date, hour, temperature_2m, cloud_cover, wind_speed_10m,
                        shortwave_radiation, relative_humidity, precipitation_mm)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (province, date, w["hour"],
                     w.get("temperature_2m"), w.get("cloud_cover"),
                     w.get("wind_speed_10m"), w.get("shortwave_radiation"),
                     w.get("relative_humidity"), w.get("precipitation_mm"))
                )
                count += 1
            except Exception as e:
                logger.error(f"[Ingest] weather 失败: {e}")
        self.conn.commit()
        return count

    def ingest_contract(self, province: str, contract: dict) -> bool:
        """摄入/更新中长期合约"""
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO contracts
                   (province, contract_id, contract_type, volume_mwh, price_rmb_mwh,
                    start_date, end_date, counterparty)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (province, contract["contract_id"], contract.get("type"),
                 contract.get("volume_mwh"), contract.get("price"),
                 contract.get("start_date"), contract.get("end_date"),
                 contract.get("counterparty"))
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[Ingest] contract 失败: {e}")
            return False

    def ingest_bidding_record(self, province: str, record: dict) -> bool:
        """摄入申报记录"""
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO bidding_records
                   (province, trade_date, bid_avg_price, clearing_avg_price,
                    deviation_pct, deviation_status, volume_mwh, strategy_type, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (province, record["trade_date"], record.get("bid_avg_price"),
                 record.get("clearing_avg_price"), record.get("deviation_pct"),
                 record.get("deviation_status"), record.get("volume_mwh"),
                 record.get("strategy_type"), record.get("notes"))
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[Ingest] bidding 失败: {e}")
            return False

    # ══ 查询方法 ══

    def query_prices(self, province: str, start_date: str, end_date: str) -> list[dict]:
        """查询历史电价"""
        rows = self.conn.execute(
            "SELECT * FROM price_data WHERE province=? AND date BETWEEN ? AND ? ORDER BY date, hour",
            (province, start_date, end_date)
        ).fetchall()
        return [dict(r) for r in rows]

    def query_latest_forecast(self, province: str) -> Optional[dict]:
        """查询最新预测"""
        row = self.conn.execute(
            "SELECT * FROM forecasts WHERE province=? ORDER BY target_date DESC LIMIT 1",
            (province,)
        ).fetchone()
        return dict(row) if row else None

    def query_contracts(self, province: str) -> list[dict]:
        """查询合约持仓"""
        rows = self.conn.execute(
            "SELECT * FROM contracts WHERE province=? AND status='active' ORDER BY end_date",
            (province,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ══ ChromaDB 辅助 ══

    def _chroma_upsert_forecast(self, province: str, forecast: dict):
        """预测摘要写入 ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=str(DATA_ROOT / "vectordb"),
                settings=Settings(anonymized_telemetry=False),
            )
            try:
                col = client.get_collection("power_market_kb")
            except Exception:
                col = client.create_collection(name="power_market_kb", metadata={"hnsw:space": "cosine"})

            avg = forecast.get("avg_price", 0)
            peak = forecast.get("peak_avg", 0)
            valley = forecast.get("valley_avg", 0)
            spread = peak - valley if peak and valley else 0
            tags = []
            if avg > 500: tags.append("高电价日")
            elif avg < 300: tags.append("低电价日")
            if spread > 200: tags.append("峰谷价差大")
            if forecast.get("confidence", 0) > 0.8: tags.append("高置信度")

            summary = (f"【{province} 电价预测】{forecast['target_date']}\n"
                       f"日均价: {avg:.2f} 元/MWh | 峰时: {peak:.2f} | 谷时: {valley:.2f}\n"
                       f"特征: {', '.join(tags)}")

            doc_id = f"forecast_{province}_{forecast['target_date']}"
            col.upsert(ids=[doc_id], documents=[summary], metadatas=[{
                "data_type": "forecast_archive", "province": province,
                "target_date": forecast["target_date"], "avg_price": avg,
                "features": ", ".join(tags),
            }])
        except Exception as e:
            logger.warning(f"[Ingest] ChromaDB upsert 失败 (非致命): {e}")

    def close(self):
        if self._conn:
            self._conn.close()


# ══ 单例 ══

_pipeline: Optional[DataIngestionPipeline] = None


def get_ingestion_pipeline() -> DataIngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DataIngestionPipeline()
    return _pipeline
