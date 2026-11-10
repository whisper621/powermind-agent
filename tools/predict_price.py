#!/usr/bin/env python3
"""
电价预测 Tool —— 对接可插拔预测引擎 (ForecasterRegistry)。

热插拔: 底层模型切换时，此文件完全不用改。
  registry.set_active("lstm_v3") 即可切换到新模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool
from quant.registry import ForecasterRegistry
from quant.interface import ForecastInput


@tool
def predict_electricity_price(target_date: str, province: str = "山东") -> str:
    """
    预测指定省份日前电力市场出清电价（24点逐时预测）。

    底层使用当前激活的预测模型（通过 ForecasterRegistry 热插拔管理）。
    当用户询问"预测某天电价"、"明天电价多少"、"电价走势"时，必须调用此工具。

    Args:
        target_date: 目标预测日期，格式 YYYY-MM-DD
        province: 省份名称，默认 "山东"

    Returns:
        包含24点逐时电价预测、日均价、峰谷均价和置信度的结构化结果。
    """
    try:
        # ★ 先查SQL里是否有该省该日的预测存档
        stored = _check_stored_forecast(province, target_date)
        if stored:
            result = stored  # 使用存储的预测
        else:
            # 没有存档 → 跑预测模型
            registry = ForecasterRegistry.get_instance()
            input_data = ForecastInput(province=province, target_date=target_date)
            result = registry.predict(input_data)
            # 双写归档
            _archive_prediction(result)

        # 构造 24 点分时表格
        hourly_lines = []
        for i in range(0, 24, 4):
            chunk = result.hourly_predictions[i : i + 4]
            hourly_lines.append(
                f"  {i:02d}:00-{min(i+3,23):02d}:00 | "
                + " | ".join(f"{p:7.2f}" for p in chunk)
            )

        model_tag = f"{result.model_name} v{result.model_version}"

        # ★ 注入天气和历史上下文
        weather_block = _get_weather_summary(province, target_date)
        history_block = _get_history_summary(province, target_date)

        return (
            f"【电价预测结果 · {result.province} · {result.target_date}】\n"
            f"  预测引擎: {model_tag}\n"
            f"  日均价: {result.avg_price:.2f} 元/MWh\n"
            f"  峰时均价 (18-21时): {result.peak_avg:.2f} 元/MWh\n"
            f"  谷时均价 (00-06时): {result.valley_avg:.2f} 元/MWh\n"
            f"  峰谷价差: {result.peak_avg - result.valley_avg:.2f} 元/MWh\n"
            f"  置信度: {result.confidence:.0%}\n"
            f"\n  24点逐时预测 (元/MWh):\n"
            + "\n".join(hourly_lines)
            + f"\n{weather_block}\n{history_block}"
            + f"\n\n  提示: 此为{model_tag}预测值，实际电价受供需/天气/政策影响。"
        )

    except RuntimeError as e:
        return (
            f"【系统提示】预测引擎未初始化: {e}\n"
            f"请先注册并激活预测模型:\n"
            f"  from quant.plugins.lstm_plugin import LSTMPlugin\n"
            f"  registry = ForecasterRegistry.get_instance()\n"
            f"  registry.register('lstm', LSTMPlugin())\n"
            f"  registry.set_active('lstm')"
        )
    except Exception as e:
        return f"【系统提示】电价预测异常: {type(e).__name__}: {str(e)[:300]}"


# ── 查询已存储的预测 ─────────────────────────────────────────

def _check_stored_forecast(province: str, target_date: str):
    """先查 SQL 是否有该省该日的预测存档"""
    try:
        import sqlite3, json
        db_path = str(PROJECT_ROOT / "data" / "sql" / "poweragent.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM forecasts WHERE province=? AND target_date=? ORDER BY created_at DESC LIMIT 1",
            (province, target_date)
        ).fetchone()
        conn.close()
        if not row:
            return None
        # 构造 ForecastOutput 兼容对象
        class StoredForecast:
            province = row["province"]
            target_date = row["target_date"]
            model_name = row["model_name"] or "stored"
            model_version = row["model_version"] or "1.0"
            avg_price = row["avg_price"]
            peak_avg = row["peak_avg"]
            valley_avg = row["valley_avg"]
            confidence = row["confidence"] or 0.7
            hourly_predictions = json.loads(row["hourly_predictions"] or "[]")
            p10 = json.loads(row["p10"] or "[]") if row["p10"] else None
            p50 = json.loads(row["p50"] or "[]") if row["p50"] else None
            p90 = json.loads(row["p90"] or "[]") if row["p90"] else None
        return StoredForecast
    except Exception:
        return None


# ── 天气和历史摘要 ─────────────────────────────────────────

def _get_weather_summary(province: str, target_date: str) -> str:
    """从SQL读取天气数据并格式化"""
    try:
        import sqlite3
        db_path = str(PROJECT_ROOT / "data" / "sql" / "poweragent.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT date, AVG(temperature_2m) as t, AVG(cloud_cover) as c, "
            "AVG(wind_speed_10m) as w, SUM(precipitation_mm) as p "
            "FROM weather_data WHERE province=? AND date>=? GROUP BY date ORDER BY date LIMIT 3",
            (province, target_date)
        ).fetchall()
        conn.close()
        if not rows:
            return ""
        lines = ["\n  ★ 天气数据 (来自Open-Meteo):"]
        for r in rows:
            temp_flag = "⚠️高温" if r[1] > 35 else ("❄️低温" if r[1] < 0 else "正常")
            rain_flag = f"降水{r[4]:.1f}mm" if r[4] > 0 else "无降水"
            lines.append(
                f"  | {r[0]} | 均温{r[1]:.0f}°C({temp_flag}) | 云量{r[2]:.0f}% | "
                f"风速{r[3]:.1f}m/s | {rain_flag} |"
            )
        return "\n".join(lines)
    except Exception:
        return ""

def _get_history_summary(province: str, target_date: str) -> str:
    """从SQL读取近期历史电价"""
    try:
        import sqlite3
        db_path = str(PROJECT_ROOT / "data" / "sql" / "poweragent.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT date, AVG(price_rmb_mwh) as avg FROM price_data "
            "WHERE province=? AND date < ? GROUP BY date ORDER BY date DESC LIMIT 7",
            (province, target_date)
        ).fetchall()
        conn.close()
        if not rows:
            return ""
        lines = ["\n  ★ 近7日历史电价 (元/MWh):"]
        for r in reversed(rows):
            lines.append(f"  {r[0]}: {r[1]:.0f}")
        avg_7d = sum(r[1] for r in rows) / len(rows)
        lines.append(f"  7日均值: {avg_7d:.0f} 元/MWh")
        return "\n".join(lines)
    except Exception:
        return ""


# ── 启动时自动注册默认模型 ─────────────────────────────────────────

def _auto_register_default():
    """在模块导入时自动注册 LSTM 为默认预测引擎。"""
    try:
        registry = ForecasterRegistry.get_instance()
        if registry.model_count == 0:
            from quant.plugins.lstm_plugin import LSTMPlugin
            plugin = LSTMPlugin(
                model_path=str(PROJECT_ROOT / "models" / "lstm_epf.pth")
            )
            registry.register("lstm_default", plugin)
            # 不强制 set_active — register 会自动设为默认
    except Exception:
        pass  # 首次导入可能没有依赖，运行时再处理


def _archive_prediction(result) -> None:
    """预测结果双写归档：写入 SQLite + ChromaDB，静默失败不影响主流程。"""
    try:
        from tools.prediction_archiver import archive_forecast
        sqlite_id, chroma_ok = archive_forecast(result)
    except Exception:
        pass  # 归档失败不应阻断预测主流程


_auto_register_default()
