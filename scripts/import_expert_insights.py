#!/usr/bin/env python3
"""
专家实战心得导入器 — 将各省交易专家的复盘日志、极端行情心得、博弈策略
按省份标注后存入 ChromaDB，支持 RAG 按省精准检索。

数据格式 (CSV 或 JSON):
  必填字段: province, category, title, content
  可选字段: date, market_stage, tags

使用方式:
  1. 专家按模版写好心得 → 保存为 CSV
  2. 运行: python scripts/import_expert_insights.py --file expert_notes_shandong.csv
  3. RAG 自动按省份检索
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")
COLLECTION_NAME = "power_market_kb"

# 专家心得分类体系
EXPERT_CATEGORIES = [
    "复盘日志",        # 每日交易后总结
    "极端行情应对",     # 负电价/尖峰电价/连续涨停的操作手记
    "博弈心理",        # 识别对手策略、恐慌抛售判断
    "偏差考核避坑",    # 接近红线的经验、补仓时机
    "申报策略心得",    # 各时段报价技巧
    "省间套利实战",    # 跨省交易的通道选择经验
    "市场阶段策略",    # 不同市场周期(淡季/旺季/迎峰度夏)的通用策略
    "其他",            # 未分类心得
]


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


def import_from_csv(csv_path: str, province: str) -> int:
    """
    从 CSV 导入专家心得。

    CSV 列：category, title, content[, date, market_stage, tags]
    province 通过命令行指定，确保数据准确标注。
    """
    collection = get_collection()
    imported = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, 1):
            category = row.get("category", "其他")
            title = row.get("title", "")
            content = row.get("content", "")
            date_str = row.get("date", "")
            tags = row.get("tags", "")

            if not title or not content:
                print(f"  ⚠️ Row {row_idx}: 跳过 (缺 title 或 content)")
                continue

            # 构造文档
            doc_text = (
                f"【{province}省·{category}】{title}\n"
                f"日期: {date_str or '未标注'}\n"
                f"标签: {tags or '无'}\n"
                f"内容:\n{content[:3000]}"
            )
            doc_id = f"expert-{province}-{category}-{row_idx}-{datetime.now().strftime('%Y%m%d')}"

            try:
                collection.add(
                    ids=[doc_id],
                    documents=[doc_text],
                    metadatas=[{
                        "source": "expert_insight_csv",
                        "data_type": "expert_insight",
                        "province": province,
                        "category": category,
                        "date": date_str,
                        "tags": tags,
                        "title": title[:100],
                    }],
                )
                imported += 1
                print(f"  ✅ [{category}] {title[:50]}")
            except Exception as e:
                print(f"  ❌ Row {row_idx}: {e}")

    return imported


def import_single_note(
    province: str, category: str, title: str, content: str,
    date_str: str = "", tags: str = ""
) -> bool:
    """导入单条专家心得（编程接口）。"""
    collection = get_collection()
    doc_text = (
        f"【{province}省·{category}】{title}\n"
        f"日期: {date_str or '未标注'}\n"
        f"标签: {tags or '无'}\n"
        f"内容:\n{content[:3000]}"
    )
    doc_id = f"expert-{province}-{category}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        collection.add(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{
                "source": "expert_insight_manual",
                "data_type": "expert_insight",
                "province": province,
                "category": category,
                "date": date_str,
                "tags": tags,
                "title": title[:100],
            }],
        )
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def import_price_history_csv(csv_path: str, province: str) -> int:
    """
    从 CSV 导入历史电价数据到 ChromaDB（★ 用于 RAG 电价模式检索）。

    CSV 列：date, day_ahead_avg, peak_price, valley_price, notes
    每行代表一天的日前出清电价摘要。
    """
    collection = get_collection()
    imported = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row.get("date", "")
            da_avg = row.get("day_ahead_avg", "")
            peak = row.get("peak_price", "")
            valley = row.get("valley_price", "")
            notes = row.get("notes", "")

            if not date_str:
                continue

            doc_text = (
                f"{province}省日前出清电价 - {date_str}\n"
                f"日均价: {da_avg} 元/MWh\n"
                f"峰时均价(18-21): {peak} 元/MWh\n"
                f"谷时均价(00-06): {valley} 元/MWh\n"
                f"备注: {notes or '无'}"
            )
            doc_id = f"price-{province}-{date_str}"

            try:
                collection.add(
                    ids=[doc_id],
                    documents=[doc_text],
                    metadatas=[{
                        "source": "price_history_csv",
                        "data_type": "price_history",
                        "province": province,
                        "date": date_str,
                        "day_ahead_avg": float(da_avg) if da_avg else 0.0,
                    }],
                )
                imported += 1
            except Exception as e:
                print(f"  ⚠️ {date_str}: {e}")

    return imported


def list_expert_insights(province: str = None) -> list[dict]:
    """列出已入库的专家心得。"""
    collection = get_collection()
    where_filter = {"data_type": "expert_insight"}
    if province:
        where_filter["province"] = province
    try:
        results = collection.get(where=where_filter, limit=50)
        return [
            {"id": rid, "metadata": meta, "content": doc[:100]}
            for rid, meta, doc in zip(
                results["ids"], results["metadatas"] or [{}], results["documents"] or [""]
            )
        ]
    except Exception:
        return []


# ── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="专家心得 & 电价历史导入器")
    parser.add_argument("--csv", help="CSV 文件路径")
    parser.add_argument("--province", "-p", required=True, help="省份（如 山东、山西）")
    parser.add_argument("--type", "-t", default="expert",
                        choices=["expert", "price"],
                        help="导入类型: expert(专家心得) / price(电价历史)")
    parser.add_argument("--list", action="store_true", help="列出已入库的专家心得")
    args = parser.parse_args()

    if args.list:
        items = list_expert_insights(args.province)
        print(f"\n已入库专家心得 ({len(items)} 条):")
        for item in items[:20]:
            meta = item.get("metadata", {})
            print(f"  [{meta.get('category','?')}] {meta.get('title','')[:50]} "
                  f"({meta.get('province','?')} {meta.get('date','?')})")
        sys.exit(0)

    if not args.csv:
        print("请指定 --csv 文件路径 或使用 --list 查看已入库内容。")
        print(f"\n支持的分类: {', '.join(EXPERT_CATEGORIES)}")
        sys.exit(1)

    if args.type == "expert":
        count = import_from_csv(args.csv, args.province)
        print(f"\n✅ 导入 {count} 条专家心得 (省份: {args.province})")
    elif args.type == "price":
        count = import_price_history_csv(args.csv, args.province)
        print(f"\n✅ 导入 {count} 条电价历史 (省份: {args.province})")
