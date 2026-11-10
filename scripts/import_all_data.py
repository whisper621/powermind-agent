#!/usr/bin/env python3
"""
PowerMind Agent — 全量数据导入器。

将项目 A 和项目 B 的全部数据资产分类导入新项目的 ChromaDB。
数据分为三大类:
  A. 结构化数据 (SQLite) → CSV 中间文件 → ChromaDB
  B. 半结构化数据 (JSON/CSV) → 直接向量化 → ChromaDB
  C. 文本文档 (Markdown/TXT/PDF) → 分块 → ChromaDB
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

A_ROOT = Path(r"D:\GitHub_work\china-power-market-agent-main\power_market_agent")
B_ROOT = Path(r"D:\GitHub_work\PowerTrade-LLM-Agent-main")

CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")
COLLECTION_NAME = "power_market_kb"


# ═══════════════════════════════════════════════════════════════════
# 1. ChromaDB 初始化
# ═══════════════════════════════════════════════════════════════════

def get_or_create_collection():
    """获取或创建 ChromaDB collection。"""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ═══════════════════════════════════════════════════════════════════
# 2. 数据加载函数
# ═══════════════════════════════════════════════════════════════════

def load_sqlite_data() -> list[dict]:
    """从项目 A 的 SQLite 提取所有结构化数据。"""
    items = []
    db_path = A_ROOT / "knowledge_base" / "knowledge_base.db"
    if not db_path.exists():
        return items

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 省份基础信息
    for row in conn.execute("SELECT * FROM provinces").fetchall():
        items.append({
            "id": f"sql-province-{row['code']}",
            "text": f"省份: {row['name']}({row['code']}) | 区域: {row['region']} | 能源结构: {row['energy_mix']}",
            "metadata": {
                "source": "sqlite:provinces",
                "data_type": "province_info",
                "province": row["name"],
                "region": row["region"] or "",
            },
        })

    # 城市坐标
    for row in conn.execute(
        "SELECT c.name as city, c.lat, c.lon, p.name as province "
        "FROM cities c JOIN provinces p ON c.province_id=p.id"
    ).fetchall():
        items.append({
            "id": f"sql-city-{row['city']}",
            "text": f"城市: {row['city']} | 省份: {row['province']} | 坐标: ({row['lat']}, {row['lon']})",
            "metadata": {
                "source": "sqlite:cities",
                "data_type": "city_info",
                "province": row["province"],
                "city": row["city"],
            },
        })

    # 因子快照
    for row in conn.execute("SELECT * FROM factor_snapshots").fetchall():
        factors = json.loads(row["factors_json"])
        text_parts = [f"省份: {row['province']} | 日期: {row['date']}"]
        for cat, sigs in factors.get("factors", {}).items():
            for s in sigs:
                text_parts.append(f"{cat}: {s['name']} ({s['direction']}, 强度{s['strength']})")
        items.append({
            "id": f"sql-factor-{row['province']}-{row['date']}",
            "text": " | ".join(text_parts[:20]),
            "metadata": {
                "source": "sqlite:factor_snapshots",
                "data_type": "market_factor",
                "province": row["province"],
                "date": row["date"],
                "direction": row["overall_direction"] or "",
                "risk": row["overall_risk"] or "",
            },
        })

    conn.close()
    return items


def load_json_data() -> list[dict]:
    """从项目 A 的 JSON 数据文件提取。"""
    items = []

    # 数据源注册表
    sources_path = A_ROOT / "source_registry" / "sources.json"
    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
        for src in sources:
            items.append({
                "id": f"json-source-{src.get('name', 'unknown')}",
                "text": (
                    f"数据源: {src.get('name')} | 类型: {src.get('type')} "
                    f"| 焦点: {src.get('focus')} | 频率: {src.get('frequency')} "
                    f"| 可信度: {src.get('credibility')} | URL: {src.get('url')}"
                ),
                "metadata": {
                    "source": "source_registry",
                    "data_type": "data_source",
                    "name": src.get("name", ""),
                    "category": src.get("type", ""),
                },
            })

    # processed_data JSON
    processed_dir = A_ROOT / "processed_data"
    if processed_dir.exists():
        for f in processed_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                text = json.dumps(data, ensure_ascii=False)[:3000]
                items.append({
                    "id": f"json-processed-{f.stem}",
                    "text": text,
                    "metadata": {
                        "source": f"processed_data/{f.name}",
                        "data_type": "processed_pulse",
                        "date": f.stem[:10],
                    },
                })
            except Exception:
                pass

    # raw_data JSON
    raw_dir = A_ROOT / "raw_data"
    if raw_dir.exists():
        for f in raw_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                text = json.dumps(data, ensure_ascii=False)[:3000]
                items.append({
                    "id": f"json-raw-{f.stem}",
                    "text": text,
                    "metadata": {
                        "source": f"raw_data/{f.name}",
                        "data_type": "raw_pulse",
                        "date": f.stem[:10],
                    },
                })
            except Exception:
                pass

    return items


def load_csv_data() -> list[dict]:
    """从项目 A 和 B 的 CSV 数据文件提取。"""
    items = []

    # 项目A: external_data CSV
    csv_dirs = [
        A_ROOT / "external_data" / "province_market_background" / "package" / "data",
        A_ROOT / "external_data" / "province_market_background" / "package" / "reports",
    ]
    for csv_dir in csv_dirs:
        if not csv_dir.exists():
            continue
        for f in csv_dir.rglob("*.csv"):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                # 分块处理大文件
                chunks = [text[i:i+2000] for i in range(0, min(len(text), 10000), 2000)]
                for ci, chunk in enumerate(chunks):
                    items.append({
                        "id": f"csv-a-{f.stem}-{ci}",
                        "text": f"文件: {f.relative_to(csv_dir)}\n{chunk}",
                        "metadata": {
                            "source": f"project_a:{f.relative_to(A_ROOT)}",
                            "data_type": "csv_data",
                            "file_name": f.name,
                        },
                    })
            except Exception:
                pass

    # 项目B: 电力期货CSV
    b_csv = B_ROOT / "algorithms" / "clean_electric_data.csv"
    if b_csv.exists():
        try:
            text = b_csv.read_text(encoding="utf-8")
            # 取头部和尾部
            lines = text.split("\n")
            sample = "\n".join(lines[:50] + ["..."] + lines[-30:])
            items.append({
                "id": "csv-b-electric-futures",
                "text": f"美国电力期货数据 (1269条):\n{sample}",
                "metadata": {
                    "source": "project_b:algorithms/clean_electric_data.csv",
                    "data_type": "price_history",
                    "market": "US_EPF",
                    "records": len(lines) - 1,
                },
            })
        except Exception:
            pass

    return items


def load_markdown_docs() -> list[dict]:
    """从项目 A 和 B 的 Markdown/TXT 文档提取。"""
    items = []

    # 项目A: prompts (23 个 .md 文件)
    prompts_dir = A_ROOT / "prompts"
    if prompts_dir.exists():
        for f in prompts_dir.rglob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
                items.append({
                    "id": f"md-prompt-{f.stem}",
                    "text": f"提示词模板: {f.stem}\n{text[:3000]}",
                    "metadata": {
                        "source": f"project_a:prompts/{f.name}",
                        "data_type": "prompt_template",
                        "category": "prompt",
                    },
                })
            except Exception:
                pass

    # 项目A: knowledge_base 文档
    kb_dir = A_ROOT / "knowledge_base"
    if kb_dir.exists():
        for f in kb_dir.rglob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
                items.append({
                    "id": f"md-kb-{f.stem}",
                    "text": f"知识库文档: {f.stem}\n{text[:3000]}",
                    "metadata": {
                        "source": f"project_a:knowledge_base/{f.name}",
                        "data_type": "knowledge_doc",
                        "category": f.stem,
                    },
                })
            except Exception:
                pass

    # 项目A: daily_pulse Markdown
    pulse_dir = A_ROOT / "daily_pulse"
    if pulse_dir.exists():
        for f in pulse_dir.glob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
                items.append({
                    "id": f"md-pulse-{f.stem}",
                    "text": text[:3000],
                    "metadata": {
                        "source": f"project_a:daily_pulse/{f.name}",
                        "data_type": "daily_report",
                        "date": f.stem[:10],
                    },
                })
            except Exception:
                pass

    # 项目A: external_data Markdown
    ext_dir = A_ROOT / "external_data"
    if ext_dir.exists():
        for f in ext_dir.rglob("*.md"):
            try:
                text = f.read_text(encoding="utf-8")
                items.append({
                    "id": f"md-ext-{f.stem}",
                    "text": text[:3000],
                    "metadata": {
                        "source": f"project_a:external_data/{f.name}",
                        "data_type": "external_doc",
                    },
                })
            except Exception:
                pass

    # 项目B: prompts
    b_prompts = B_ROOT / "prompts"
    if b_prompts.exists():
        for f in b_prompts.rglob("*.txt"):
            try:
                text = f.read_text(encoding="utf-8")
                items.append({
                    "id": f"txt-prompt-b-{f.stem}",
                    "text": text[:3000],
                    "metadata": {
                        "source": f"project_b:prompts/{f.name}",
                        "data_type": "prompt_template",
                    },
                })
            except Exception:
                pass

    # 项目B: data 目录 (PDF/TXT)
    b_data = B_ROOT / "data"
    if b_data.exists():
        for f in b_data.rglob("*"):
            if f.suffix.lower() in (".txt",):
                try:
                    text = f.read_text(encoding="utf-8")
                    items.append({
                        "id": f"doc-b-{f.stem}",
                        "text": text[:3000],
                        "metadata": {
                            "source": f"project_b:data/{f.name}",
                            "data_type": "reference_doc",
                        },
                    })
                except Exception:
                    pass

    return items


def load_province_attrs() -> list[dict]:
    """从项目 A 的 province_extended_attrs.py 提取省份属性。"""
    items = []
    try:
        sys.path.insert(0, str(A_ROOT))
        from external_data.province_extended_attrs import (
            TRANSMISSION_CORRIDORS,
            CONSUMPTION_STRUCTURE,
        )

        for prov, corridors in TRANSMISSION_CORRIDORS.items():
            items.append({
                "id": f"attr-transmission-{prov}",
                "text": f"省份: {prov} | 跨省输电通道: {'; '.join(corridors)}",
                "metadata": {
                    "source": "province_extended_attrs",
                    "data_type": "transmission_corridor",
                    "province": prov,
                },
            })

        for prov, (ind, comm, res, agri) in CONSUMPTION_STRUCTURE.items():
            items.append({
                "id": f"attr-consumption-{prov}",
                "text": (
                    f"省份: {prov} | 用电结构: 工业{ind}% 商业{comm}% 居民{res}% 农业{agri}%"
                ),
                "metadata": {
                    "source": "province_extended_attrs",
                    "data_type": "consumption_structure",
                    "province": prov,
                },
            })
    except Exception:
        pass

    return items


# ═══════════════════════════════════════════════════════════════════
# 3. 主导入流程
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PowerMind Agent — 全量数据导入")
    print("=" * 60)

    # 收集所有数据
    all_items: list[dict] = []
    loaders = [
        ("SQLite 结构化数据", load_sqlite_data),
        ("JSON 半结构化数据", load_json_data),
        ("CSV 数据表", load_csv_data),
        ("Markdown/TXT 文档", load_markdown_docs),
        ("省份扩展属性", load_province_attrs),
    ]

    for label, loader in loaders:
        items = loader()
        all_items.extend(items)
        print(f"  {label}: {len(items)} 条")

    print(f"\n  总计: {len(all_items)} 条数据项")
    print(f"  去重中...")

    # 去重
    seen = set()
    deduped = []
    for item in all_items:
        key = item["text"][:200]
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    print(f"  去重后: {len(deduped)} 条")

    # 灌入 ChromaDB
    print(f"\n  灌入 ChromaDB ({CHROMA_PATH})...")
    collection = get_or_create_collection()

    batch_size = 50
    for i in range(0, len(deduped), batch_size):
        batch = deduped[i:i + batch_size]
        try:
            collection.add(
                ids=[b["id"] for b in batch],
                documents=[b["text"] for b in batch],
                metadatas=[b["metadata"] for b in batch],
            )
        except Exception as e:
            print(f"    batch {i//batch_size} error: {e}")

    print(f"  ✅ 导入完成: {collection.count()} 条文档")
    print(f"  存储路径: {CHROMA_PATH}")

    # 分类统计
    types: dict[str, int] = {}
    for item in deduped:
        t = item["metadata"].get("data_type", "other")
        types[t] = types.get(t, 0) + 1
    print(f"\n  数据类型分布:")
    for k, v in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
