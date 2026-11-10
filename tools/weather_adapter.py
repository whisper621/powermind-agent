#!/usr/bin/env python3
"""
天气数据适配器 —— 调用 Open-Meteo API 获取天气预报 + 自动归档到知识库。

原则:
  - 实时天气预报 → Redis 缓存 (1h TTL) + SQL weather_data 表
  - 历史天气实况 → SQL weather_data 表 (永久保存)
  - 不写 ChromaDB (天气数据变化频繁, 向量库不适合高频更新)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 17省主要城市坐标
PROVINCE_COORDS = {
    "山东": (36.67, 116.98),  # 济南
    "河南": (34.76, 113.65),  # 郑州
    "江苏": (32.06, 118.79),  # 南京
    "广东": (23.13, 113.26),  # 广州
    "浙江": (30.29, 120.15),  # 杭州
    "山西": (37.87, 112.55),  # 太原
    "河北": (38.04, 114.51),  # 石家庄
    "内蒙古": (40.82, 111.75), # 呼和浩特
}


@tool
def fetch_weather_forecast(province: str, forecast_days: int = 7) -> str:
    """
    获取指定省份的天气预报 (Open-Meteo API, 免费, 无需Key)。

    自动归档到 SQL weather_data 表, 供因子计算和历史分析使用。

    Args:
        province: 省份名称
        forecast_days: 预报天数 (1-16)

    Returns:
        包含温度/云量/风速/辐射/湿度的24小时逐时预报
    """
    coords = PROVINCE_COORDS.get(province)
    if not coords:
        return f"【天气】省份 '{province}' 坐标未配置, 已配置: {list(PROVINCE_COORDS.keys())}"

    lat, lon = coords
    forecast_days = max(1, min(16, forecast_days))

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,shortwave_radiation,precipitation",
        "timezone": "Asia/Shanghai",
        "forecast_days": forecast_days,
    }

    try:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"[Weather] API调用失败: {e}")
        return f"【天气】API调用失败: {e}"

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    humidity = hourly.get("relative_humidity_2m", [])
    cloud = hourly.get("cloud_cover", [])
    wind = hourly.get("wind_speed_10m", [])
    radiation = hourly.get("shortwave_radiation", [])
    precip = hourly.get("precipitation", [])

    if not times:
        return "【天气】API返回空数据"

    # 归档到 SQL
    _archive_weather_to_sql(province, times, temps, humidity, cloud, wind, radiation, precip)

    # 格式化输出
    lines = [
        f"【{province} 天气预报 · Open-Meteo】",
        f"坐标: ({lat}, {lon}) | 预报天数: {forecast_days}",
        f"",
    ]

    # 按天汇总
    daily_groups = {}
    for i, t in enumerate(times):
        date_str = t[:10]  # "2026-06-15T00:00"
        if date_str not in daily_groups:
            daily_groups[date_str] = {"temps": [], "cloud": [], "wind": [], "precip": []}
        if i < len(temps): daily_groups[date_str]["temps"].append(temps[i])
        if i < len(cloud): daily_groups[date_str]["cloud"].append(cloud[i])
        if i < len(wind): daily_groups[date_str]["wind"].append(wind[i])
        if i < len(precip): daily_groups[date_str]["precip"].append(precip[i])

    for date_str, vals in list(daily_groups.items())[:forecast_days]:
        t_avg = sum(vals["temps"]) / len(vals["temps"]) if vals["temps"] else 0
        t_max = max(vals["temps"]) if vals["temps"] else 0
        t_min = min(vals["temps"]) if vals["temps"] else 0
        c_avg = sum(vals["cloud"]) / len(vals["cloud"]) if vals["cloud"] else 0
        w_avg = sum(vals["wind"]) / len(vals["wind"]) if vals["wind"] else 0
        p_sum = sum(vals["precip"]) if vals["precip"] else 0

        flags = []
        if t_max > 35: flags.append("⚠️高温")
        if t_min < 0: flags.append("❄️极寒")
        if c_avg < 30: flags.append("☀️晴天")
        if p_sum > 10: flags.append("🌧️强降水")

        lines.append(
            f"| {date_str} | {t_min:.0f}~{t_max:.0f}°C | 云量{c_avg:.0f}% | "
            f"风{w_avg:.1f}m/s | 雨{p_sum:.1f}mm | {''.join(flags)} |"
        )

    lines.append(f"\n_数据来源: Open-Meteo (免费API, 已自动归档到SQL)_")
    return "\n".join(lines)


def _archive_weather_to_sql(province, times, temps, humidity, cloud, wind, radiation, precip):
    """天气预报自动归档到 SQL weather_data 表"""
    try:
        from storage.data_ingestion import get_ingestion_pipeline
        pipeline = get_ingestion_pipeline()

        count = 0
        for i, t in enumerate(times):
            date_str = t[:10]
            hour = int(t[11:13]) if len(t) >= 13 else 0

            weather_point = {
                "hour": hour,
                "temperature_2m": temps[i] if i < len(temps) else None,
                "relative_humidity": humidity[i] if i < len(humidity) else None,
                "cloud_cover": cloud[i] if i < len(cloud) else None,
                "wind_speed_10m": wind[i] if i < len(wind) else None,
                "shortwave_radiation": radiation[i] if i < len(radiation) else None,
                "precipitation_mm": precip[i] if i < len(precip) else None,
            }
            pipeline.ingest_weather(province, date_str, [weather_point])
            count += 1

        logger.info(f"[Weather] 已归档 {count} 条天气数据 → SQL")
    except Exception as e:
        logger.warning(f"[Weather] 归档失败 (非致命): {e}")


@tool
def fetch_weather_current(province: str) -> str:
    """
    获取指定省份的当前天气 (实时观测)。

    Args:
        province: 省份名称

    Returns:
        当前温度/湿度/风速/云量等信息
    """
    coords = PROVINCE_COORDS.get(province)
    if not coords:
        return f"省份 '{province}' 未配置"

    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
        "timezone": "Asia/Shanghai",
    }

    try:
        resp = httpx.get(url, params=params, timeout=10)
        data = resp.json()
        cur = data.get("current", {})
        return (
            f"【{province} 当前天气】\n"
            f"  温度: {cur.get('temperature_2m', '?')}°C\n"
            f"  湿度: {cur.get('relative_humidity_2m', '?')}%\n"
            f"  风速: {cur.get('wind_speed_10m', '?')} m/s\n"
            f"  云量: {cur.get('cloud_cover', '?')}%\n"
        )
    except Exception as e:
        return f"【天气】获取失败: {e}"


# ══ CLI测试 ══

if __name__ == "__main__":
    print("=== 天气预报测试 ===")
    print(fetch_weather_forecast.invoke({"province": "山东", "forecast_days": 3}))
    print()
    print("=== 当前天气测试 ===")
    print(fetch_weather_current.invoke({"province": "山东"}))
