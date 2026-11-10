"""Repository interfaces — abstract storage from business logic.

Agents, tools, and LangGraph nodes MUST NOT import sqlite3 directly.
All data access goes through these interfaces.  Demo adapter uses SQLite.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MarketDataRepository(Protocol):
    def get_price_history(
        self, province: str, target_date: str, days: int = 7
    ) -> list[dict]:
        ...

    def get_weather(
        self, province: str, target_date: str, days: int = 3
    ) -> list[dict]:
        ...

    def get_contracts(self, province: str) -> list[dict]:
        ...


class DemoSQLiteMarketDataRepository:
    """SQLite-backed MarketDataRepository for Demo.

    Replace with PostgreSQL adapter for production — no agent code changes.
    """

    def __init__(self, db_path: str | None = None) -> None:
        import os

        if db_path is None:
            from infrastructure.paths import get_db_path
            db_path = os.environ.get("POWERMIND_DEMO_DB") or str(get_db_path("powermind_demo.db"))
        self._db_path = db_path

    @property
    def db_path(self) -> str:
        return self._db_path

    def _connect(self):
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_price_history(
        self, province: str, target_date: str, days: int = 7
    ) -> list[dict]:
        import sqlite3
        from pathlib import Path
        if not Path(self._db_path).exists():
            return []
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT date, AVG(price_rmb_mwh) as avg_price "
                "FROM price_data WHERE province=? AND date < ? "
                "GROUP BY date ORDER BY date DESC LIMIT ?",
                (province, target_date, days),
            ).fetchall()
            conn.close()
            return [
                {
                    "date": r["date"],
                    "avg_price": round(r["avg_price"], 1),
                    "province": province,
                    "unit": "元/MWh",
                    "data_type": "模拟历史",
                }
                for r in reversed(rows)
            ]
        except Exception:
            return []

    def get_forecast(self, province: str, target_date: str) -> list[dict]:
        """Return the stored 24-hour demo forecast; never runs a model."""
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT hour, predicted_price, lower_bound, upper_bound, confidence, "
                "model_version, data_source FROM forecasts "
                "WHERE province=? AND target_date=? ORDER BY hour",
                (province, target_date),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_market_factors(self, province: str, target_date: str) -> list[dict]:
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT factor_code, factor_name, score, direction, observation, data_source "
                "FROM market_factors WHERE province=? AND target_date=? ORDER BY ABS(score) DESC",
                (province, target_date),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_bidding_history(
        self, province: str, limit: int = 10, before_date: str | None = None
    ) -> list[dict]:
        try:
            conn = self._connect()
            if before_date:
                rows = conn.execute(
                    "SELECT * FROM bidding_records WHERE province=? AND trade_date<? "
                    "ORDER BY trade_date DESC LIMIT ?",
                    (province, before_date, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM bidding_records WHERE province=? ORDER BY trade_date DESC LIMIT ?",
                    (province, limit),
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_basis_history(self, province: str, limit: int = 12) -> list[dict]:
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM basis_history WHERE province=? ORDER BY month DESC LIMIT ?",
                (province, limit),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_scenario(self, task_type: str, province: str, target_date: str) -> dict:
        import json
        try:
            conn = self._connect()
            row = conn.execute(
                "SELECT title, summary, payload_json, data_source FROM demo_scenarios "
                "WHERE task_type=? AND province=? AND target_date=?",
                (task_type, province, target_date),
            ).fetchone()
            conn.close()
            if row is None:
                return {}
            result = dict(row)
            result["payload"] = json.loads(result.pop("payload_json"))
            return result
        except Exception:
            return {}

    def get_weather(
        self, province: str, target_date: str, days: int = 3
    ) -> list[dict]:
        from pathlib import Path
        if not Path(self._db_path).exists():
            return []
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT date, AVG(temperature_2m) as avg_temp, "
                "AVG(cloud_cover) as avg_cloud "
                "FROM weather_data WHERE province=? AND date>=? "
                "GROUP BY date ORDER BY date LIMIT ?",
                (province, target_date, days),
            ).fetchall()
            conn.close()
            return [
                {
                    "date": r["date"],
                    "temperature": round(r["avg_temp"], 1) if r["avg_temp"] is not None else None,
                    "cloud_cover": round(r["avg_cloud"], 1) if r["avg_cloud"] is not None else None,
                    "province": province,
                }
                for r in rows
            ]
        except Exception:
            return []

    def get_contracts(self, province: str) -> list[dict]:
        from pathlib import Path
        if not Path(self._db_path).exists():
            return []
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM contracts WHERE province=? LIMIT 20",
                (province,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def health_check(self) -> bool:
        from pathlib import Path
        return Path(self._db_path).exists()
