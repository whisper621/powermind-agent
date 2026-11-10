#!/usr/bin/env python3
"""
Prediction Archiver — 预测结果归档入库模块。

核心职责:
  1. 每次预测完成后，将 ForecastOutput 写入 SQLite 归档表
  2. 生成自然语言摘要写入 ChromaDB，供 HybridRAG 语义检索
  3. 支持按省份、日期、模型名查询历史预测
  4. 为月度回测（MAPE/RMSE/方向准确率）提供数据基础

双写策略:
  - SQLite forecast_archives 表: 精确数值查询、回测计算
  - ChromaDB data_type="forecast_archive": 语义检索、RAG 上下文注入
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE_DB_PATH = str(PROJECT_ROOT / "data" / "sql" / "forecast_archives.db")
CHROMA_PATH = str(PROJECT_ROOT / "data" / "vectordb")
COLLECTION_NAME = "power_market_kb"


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

def _ensure_tables() -> None:
    """Create forecast_archives table if it doesn't exist."""
    Path(ARCHIVE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ARCHIVE_DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS forecast_archives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            province TEXT NOT NULL,
            target_date TEXT NOT NULL,
            model_name TEXT NOT NULL DEFAULT 'lstm_default',
            model_version TEXT NOT NULL DEFAULT '1.0',
            avg_price REAL NOT NULL,
            peak_avg REAL,
            valley_avg REAL,
            confidence REAL,
            p10 REAL,
            p50 REAL,
            p90 REAL,
            hourly_predictions TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            UNIQUE(province, target_date, model_name)
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_province
            ON forecast_archives(province, target_date);
        CREATE TABLE IF NOT EXISTS forecast_actuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            province TEXT NOT NULL,
            date TEXT NOT NULL,
            actual_avg_price REAL,
            actual_hourly_prices TEXT,
            source TEXT DEFAULT 'market_settlement',
            recorded_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            UNIQUE(province, date)
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Archive to SQLite
# ---------------------------------------------------------------------------

def archive_to_sqlite(
    province: str,
    target_date: str,
    model_name: str,
    model_version: str,
    avg_price: float,
    peak_avg: float,
    valley_avg: float,
    confidence: float,
    hourly_predictions: list[float],
    p10: list[float] | None = None,
    p50: list[float] | None = None,
    p90: list[float] | None = None,
    metadata: dict | None = None,
) -> int:
    """Write a single forecast result into SQLite. Returns row id.

    Uses INSERT OR REPLACE so re-running for the same (province, date, model)
    updates the existing row instead of creating a duplicate.
    """
    _ensure_tables()
    conn = sqlite3.connect(ARCHIVE_DB_PATH)
    conn.execute(
        """INSERT OR REPLACE INTO forecast_archives
           (province, target_date, model_name, model_version,
            avg_price, peak_avg, valley_avg, confidence,
            p10, p50, p90, hourly_predictions, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            province,
            target_date,
            model_name,
            model_version,
            avg_price,
            peak_avg,
            valley_avg,
            confidence,
            float(p10[0]) if p10 else None,
            float(p50[0]) if p50 else None,
            float(p90[0]) if p90 else None,
            json.dumps(hourly_predictions, ensure_ascii=False),
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )
    conn.commit()
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return row_id


# ---------------------------------------------------------------------------
# Archive to ChromaDB (text summary for RAG retrieval)
# ---------------------------------------------------------------------------

def _get_chroma_collection():
    """Get or create the ChromaDB collection."""
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


def archive_to_chromadb(
    province: str,
    target_date: str,
    model_name: str,
    model_version: str,
    avg_price: float,
    peak_avg: float,
    valley_avg: float,
    confidence: float,
    hourly_predictions: list[float],
    p10: list[float] | None = None,
    p50: list[float] | None = None,
    p90: list[float] | None = None,
) -> bool:
    """Generate a natural-language summary and index into ChromaDB.

    The summary is retrievable by HybridRAG's market/price_history/dense channels
    and carries province + date metadata for province hard-filter + time decay.
    """
    # Build hourly table
    hourly_parts = []
    for h in range(24):
        price = hourly_predictions[h]
        if h < len(hourly_predictions):
            hourly_parts.append(f"  {h:02d}:00  {price:.2f} 元/MWh")

    p10_val = f"{p10[0]:.2f}" if p10 else "N/A"
    p50_val = f"{p50[0]:.2f}" if p50 else "N/A"
    p90_val = f"{p90[0]:.2f}" if p90 else "N/A"

    # Market feature tags
    features = []
    spread = max(hourly_predictions) - min(hourly_predictions)
    if avg_price > 500:
        features.append("高电价日")
    elif avg_price < 200:
        features.append("低电价日")
    if spread > 150:
        features.append("峰谷价差大")
    if confidence > 0.8:
        features.append("高置信度预测")
    elif confidence < 0.5:
        features.append("低置信度预测")
    if not features:
        features.append("正常交易日")

    doc_text = (
        f"{province}省日前电价预测 - {target_date}\n"
        f"预测模型: {model_name} v{model_version}\n"
        f"预测日均价: {avg_price:.2f} 元/MWh\n"
        f"峰时均价 (18-21时): {peak_avg:.2f} 元/MWh\n"
        f"谷时均价 (00-06时): {valley_avg:.2f} 元/MWh\n"
        f"预测置信度: {confidence:.1%}\n"
        f"P10 (90%概率高于此): {p10_val} 元/MWh\n"
        f"P50 (中位数预测): {p50_val} 元/MWh\n"
        f"P90 (90%概率低于此): {p90_val} 元/MWh\n"
        f"峰谷价差: {spread:.2f} 元/MWh\n"
        f"市场特征: {', '.join(features)}\n"
        f"24点逐时预测:\n" + "\n".join(hourly_parts)
    )

    doc_id = f"forecast-{province}-{target_date}-{model_name}"

    try:
        collection = _get_chroma_collection()
        collection.upsert(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{
                "source": "prediction_archive",
                "data_type": "forecast_archive",
                "province": province,
                "date": target_date,
                "model_name": model_name,
                "model_version": model_version,
                "avg_price": round(avg_price, 2),
                "peak_avg": round(peak_avg, 2),
                "valley_avg": round(valley_avg, 2),
                "confidence": round(confidence, 4),
                "spread": round(spread, 2),
                "features": ",".join(features),
            }],
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Convenience: archive from ForecastOutput
# ---------------------------------------------------------------------------

def archive_forecast(output) -> tuple[int, bool]:
    """Archive a ForecastOutput (from quant.interface) to both stores.

    Args:
        output: ForecastOutput dataclass instance

    Returns:
        (sqlite_row_id, chromadb_success)
    """
    sqlite_id = archive_to_sqlite(
        province=output.province,
        target_date=output.target_date,
        model_name=output.model_name,
        model_version=output.model_version,
        avg_price=output.avg_price,
        peak_avg=output.peak_avg,
        valley_avg=output.valley_avg,
        confidence=output.confidence,
        hourly_predictions=output.hourly_predictions,
        p10=output.p10,
        p50=output.p50,
        p90=output.p90,
        metadata=output.metadata,
    )

    chroma_ok = archive_to_chromadb(
        province=output.province,
        target_date=output.target_date,
        model_name=output.model_name,
        model_version=output.model_version,
        avg_price=output.avg_price,
        peak_avg=output.peak_avg,
        valley_avg=output.valley_avg,
        confidence=output.confidence,
        hourly_predictions=output.hourly_predictions,
        p10=output.p10,
        p50=output.p50,
        p90=output.p90,
    )

    return sqlite_id, chroma_ok


# ---------------------------------------------------------------------------
# Query historical forecasts
# ---------------------------------------------------------------------------

def query_forecasts(
    province: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    model_name: str | None = None,
    limit: int = 30,
) -> list[dict]:
    """Query archived forecasts from SQLite.

    Args:
        province: 省份过滤 (None = all)
        start_date: 起始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        model_name: 模型名过滤
        limit: 最大返回条数

    Returns:
        List of forecast dicts ordered by target_date DESC.
    """
    _ensure_tables()
    conn = sqlite3.connect(ARCHIVE_DB_PATH)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM forecast_archives WHERE 1=1"
    params: list = []

    if province:
        query += " AND province=?"
        params.append(province)
    if start_date:
        query += " AND target_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND target_date <= ?"
        params.append(end_date)
    if model_name:
        query += " AND model_name=?"
        params.append(model_name)

    query += " ORDER BY target_date DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        d = dict(row)
        d["hourly_predictions"] = json.loads(d.get("hourly_predictions", "[]"))
        results.append(d)

    return results


def get_latest_forecast(province: str) -> dict | None:
    """Get the most recent forecast for a province."""
    results = query_forecasts(province=province, limit=1)
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Auto-archive on module import (background registration)
# ---------------------------------------------------------------------------

_ensure_tables()


# ── CLI 测试 ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  预测归档器 · 测试")
    print("=" * 60)

    # Test: query existing archives
    results = query_forecasts(province="山东", limit=5)
    print(f"\n山东历史预测: {len(results)} 条")
    for r in results:
        print(
            f"  {r['target_date']} | "
            f"均价={r['avg_price']:.2f} 元/MWh | "
            f"模型={r['model_name']} v{r['model_version']} | "
            f"置信={r['confidence']:.0%}"
        )

    # Print db path for verification
    print(f"\nSQLite 归档库: {ARCHIVE_DB_PATH}")
    print(f"ChromaDB 路径: {CHROMA_PATH}")
    print(f"SQLite 存在: {Path(ARCHIVE_DB_PATH).exists()}")

    if Path(ARCHIVE_DB_PATH).exists():
        conn = sqlite3.connect(ARCHIVE_DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM forecast_archives").fetchone()[0]
        conn.close()
        print(f"已归档预测总数: {count} 条")
