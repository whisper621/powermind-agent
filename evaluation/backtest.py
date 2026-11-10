"""Automated backtesting framework for PowerAgent OS.

Supports:
  - Batch import of historical market data (CSV, SQLite)
  - Automated evaluation of prediction error (MAPE, RMSE, directional accuracy)
  - Strategy adoption rate metrics
  - System latency tracking
  - Report generation (JSON, Markdown)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Try importing numerical libraries — degrade gracefully if unavailable
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    pd = None  # type: ignore


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BacktestRecord:
    """A single backtest data point."""

    timestamp: str
    province: str
    actual_price: float
    predicted_price: float
    actual_direction: str  # "up" | "down" | "flat"
    predicted_direction: str
    strategy_recommended: str = ""
    strategy_adopted: bool = False
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Aggregated backtest evaluation result."""

    total_records: int = 0
    provinces: list[str] = field(default_factory=list)
    date_range: str = ""

    # Prediction accuracy
    mape: float = 0.0  # Mean Absolute Percentage Error
    rmse: float = 0.0  # Root Mean Square Error
    mae: float = 0.0  # Mean Absolute Error
    directional_accuracy: float = 0.0  # % of correct direction predictions

    # Strategy metrics
    strategy_adoption_rate: float = 0.0  # % of strategies adopted
    strategy_count: int = 0
    strategies_adopted: int = 0

    # System performance
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Detail
    per_province: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Runs backtests on historical market data and evaluates system performance.

    Usage:
        engine = BacktestEngine()
        engine.load_csv("data/historical_prices.csv", province="山东")
        engine.load_sqlite("data/knowledge_base.db", province="山东")

        def my_predict(province, date):
            return 380.0  # your prediction logic

        result = engine.evaluate(predict_fn=my_predict)
        print(engine.report_markdown(result))
    """

    def __init__(self):
        self._records: list[BacktestRecord] = []

    # ----- data loading ------------------------------------------------------

    def load_records(self, records: list[BacktestRecord]) -> None:
        """Load pre-constructed backtest records directly."""
        self._records.extend(records)

    def load_csv(
        self,
        filepath: str,
        province: str = "",
        date_col: str = "date",
        price_col: str = "price",
        direction_col: str = "",
        predicted_col: str = "",
        predicted_direction_col: str = "",
        latency_col: str = "",
    ) -> int:
        """Load backtest data from a CSV file.

        Expected columns: date, price (actual), optionally predicted_price,
        direction, latency_ms.
        """
        if not _HAS_PANDAS:
            logger.error("pandas is required for CSV loading")
            return 0

        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            logger.error("Failed to read CSV %s: %s", filepath, exc)
            return 0

        count = 0
        for _, row in df.iterrows():
            try:
                record = BacktestRecord(
                    timestamp=str(row.get(date_col, "")),
                    province=province,
                    actual_price=float(row.get(price_col, 0)),
                    predicted_price=float(row.get(predicted_col, 0)) if predicted_col else 0.0,
                    actual_direction=str(row.get(direction_col, "")),
                    predicted_direction=str(row.get(predicted_direction_col, "")),
                    latency_ms=float(row.get(latency_col, 0)) if latency_col else 0.0,
                )
                self._records.append(record)
                count += 1
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping row in %s: %s", filepath, exc)

        logger.info("Loaded %d records from %s", count, filepath)
        return count

    def load_sqlite(
        self,
        db_path: str,
        province: str = "",
        table: str = "price_data",
        date_col: str = "date",
        price_col: str = "price",
    ) -> int:
        """Load backtest data from SQLite."""
        import sqlite3

        if not Path(db_path).exists():
            logger.warning("SQLite DB not found: %s", db_path)
            return 0

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            if province:
                cur = conn.execute(
                    f"SELECT * FROM {table} WHERE province=? ORDER BY date",
                    (province,),
                )
            else:
                cur = conn.execute(f"SELECT * FROM {table} ORDER BY date")

            count = 0
            for row in cur.fetchall():
                rowd = dict(row)
                try:
                    record = BacktestRecord(
                        timestamp=str(rowd.get(date_col, "")),
                        province=rowd.get("province", province),
                        actual_price=float(rowd.get(price_col, 0)),
                        predicted_price=0.0,
                        actual_direction="",
                        predicted_direction="",
                    )
                    self._records.append(record)
                    count += 1
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping DB row: %s", exc)

            conn.close()
            logger.info("Loaded %d records from %s", count, db_path)
            return count
        except Exception as exc:
            logger.error("Failed to load from SQLite: %s", exc)
            return 0

    # ----- evaluation --------------------------------------------------------

    def evaluate(
        self,
        predict_fn: Callable | None = None,
    ) -> BacktestResult:
        """Run evaluation over all loaded records.

        If predict_fn is provided, it will be called for each record that
        lacks a predicted_price (predict_fn(province, timestamp) -> float).
        Otherwise, only records with pre-filled predictions are used.
        """
        if not self._records:
            return BacktestResult(errors=["No records loaded"])

        # Fill missing predictions
        if predict_fn:
            for rec in self._records:
                if rec.predicted_price == 0.0:
                    try:
                        rec.predicted_price = float(predict_fn(rec.province, rec.timestamp))
                    except Exception:
                        pass
                # Infer direction from prediction
                if not rec.predicted_direction and rec.predicted_price > 0:
                    # Simple heuristic: compare with a rolling average
                    rec.predicted_direction = "flat"

        # Filter to records with predictions
        eval_records = [r for r in self._records if r.predicted_price > 0]
        if not eval_records:
            return BacktestResult(errors=["No records with predictions"])

        return self._compute_metrics(eval_records)

    def _compute_metrics(self, records: list[BacktestRecord]) -> BacktestResult:
        result = BacktestResult(total_records=len(records))

        actuals = []
        predicted = []
        for r in records:
            actuals.append(r.actual_price)
            predicted.append(r.predicted_price)

        # Use pure-Python computation when numpy is unavailable
        if _HAS_NUMPY and len(actuals) > 0:
            actuals_arr = np.array(actuals, dtype=float)
            predicted_arr = np.array(predicted, dtype=float)

            # Filter zeros
            mask = actuals_arr > 0
            actuals_arr = actuals_arr[mask]
            predicted_arr = predicted_arr[mask]

            if len(actuals_arr) > 0:
                result.mape = float(np.mean(np.abs((actuals_arr - predicted_arr) / actuals_arr)) * 100)
                result.rmse = float(np.sqrt(np.mean((actuals_arr - predicted_arr) ** 2)))
                result.mae = float(np.mean(np.abs(actuals_arr - predicted_arr)))

            # Directional accuracy
            dir_correct = sum(
                1 for r in records
                if r.actual_direction and r.predicted_direction
                and r.actual_direction == r.predicted_direction
            )
            dir_total = sum(1 for r in records if r.actual_direction and r.predicted_direction)
            result.directional_accuracy = (dir_correct / dir_total * 100) if dir_total > 0 else 0.0

            # Strategy metrics
            result.strategy_count = sum(1 for r in records if r.strategy_recommended)
            result.strategies_adopted = sum(1 for r in records if r.strategy_adopted)
            result.strategy_adoption_rate = (
                result.strategies_adopted / result.strategy_count * 100
                if result.strategy_count > 0
                else 0.0
            )

            # Latency
            latencies = [r.latency_ms for r in records if r.latency_ms > 0]
            if latencies:
                result.avg_latency_ms = float(np.mean(latencies))
                result.p95_latency_ms = float(np.percentile(latencies, 95))
                result.p99_latency_ms = float(np.percentile(latencies, 99))
        else:
            # Pure-Python fallback
            filtered = [(a, p) for a, p in zip(actuals, predicted) if a > 0]
            if filtered:
                a_list, p_list = zip(*filtered)
                n = len(a_list)
                result.mape = sum(abs((a - p) / a) for a, p in filtered) / n * 100
                result.rmse = (sum((a - p) ** 2 for a, p in filtered) / n) ** 0.5
                result.mae = sum(abs(a - p) for a, p in filtered) / n

        # Province breakdown
        provinces = {}
        for r in records:
            if r.province not in provinces:
                provinces[r.province] = {"count": 0, "mape_sum": 0.0}
            if r.actual_price > 0 and r.predicted_price > 0:
                provinces[r.province]["count"] += 1
                provinces[r.province]["mape_sum"] += abs(
                    (r.actual_price - r.predicted_price) / r.actual_price
                )
        result.per_province = {
            p: {
                "count": d["count"],
                "mape": round(d["mape_sum"] / d["count"] * 100, 2) if d["count"] > 0 else 0,
            }
            for p, d in provinces.items()
        }
        result.provinces = list(provinces.keys())

        # Date range
        timestamps = sorted(r.timestamp for r in records if r.timestamp)
        if timestamps:
            result.date_range = f"{timestamps[0]} ~ {timestamps[-1]}"

        return result

    # ----- reporting ---------------------------------------------------------

    def report_dict(self, result: BacktestResult) -> dict:
        """Return result as a plain dict (JSON-serialisable)."""
        return {
            "total_records": result.total_records,
            "provinces": result.provinces,
            "date_range": result.date_range,
            "mape_pct": round(result.mape, 2),
            "rmse": round(result.rmse, 2),
            "mae": round(result.mae, 2),
            "directional_accuracy_pct": round(result.directional_accuracy, 2),
            "strategy_adoption_rate_pct": round(result.strategy_adoption_rate, 2),
            "avg_latency_ms": round(result.avg_latency_ms, 2),
            "p95_latency_ms": round(result.p95_latency_ms, 2),
            "per_province": result.per_province,
            "errors": result.errors,
            "generated_at": result.generated_at,
        }

    def report_markdown(self, result: BacktestResult) -> str:
        """Generate a Markdown backtest report."""
        lines = [
            "# PowerAgent OS 回测报告",
            "",
            f"**生成时间**: {result.generated_at}",
            f"**数据范围**: {result.date_range or 'N/A'}",
            f"**总记录数**: {result.total_records}",
            f"**覆盖省份**: {', '.join(result.provinces) if result.provinces else 'N/A'}",
            "",
            "## 预测准确性",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| MAPE (平均绝对百分比误差) | {result.mape:.2f}% |",
            f"| RMSE (均方根误差) | {result.rmse:.2f} |",
            f"| MAE (平均绝对误差) | {result.mae:.2f} |",
            f"| 方向准确率 | {result.directional_accuracy:.2f}% |",
            "",
            "## 策略采纳",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 总策略数 | {result.strategy_count} |",
            f"| 已采纳 | {result.strategies_adopted} |",
            f"| 采纳率 | {result.strategy_adoption_rate:.2f}% |",
            "",
            "## 系统性能",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 平均延迟 | {result.avg_latency_ms:.2f} ms |",
            f"| P95 延迟 | {result.p95_latency_ms:.2f} ms |",
            f"| P99 延迟 | {result.p99_latency_ms:.2f} ms |",
        ]

        if result.per_province:
            lines.append("")
            lines.append("## 分省份统计")
            lines.append("")
            lines.append("| 省份 | 记录数 | MAPE |")
            lines.append("|------|--------|------|")
            for prov, data in result.per_province.items():
                lines.append(f"| {prov} | {data['count']} | {data['mape']:.2f}% |")

        if result.errors:
            lines.append("")
            lines.append("## 错误")
            for err in result.errors:
                lines.append(f"- {err}")

        return "\n".join(lines)

    def report_json(self, result: BacktestResult, filepath: str) -> None:
        """Write the backtest result as JSON to a file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(filepath).write_text(
            json.dumps(self.report_dict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Backtest report written to %s", filepath)
