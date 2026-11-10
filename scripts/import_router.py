#!/usr/bin/env python3
"""
PowerMind 数据导入路由器 —— 将 import_staging 中的文件按四大类别
路由到正确的存储引擎 (PostgreSQL / ChromaDB / Redis / 文件系统)。

基于 ARCHITECTURE_COMPLETE.md §0.5 的精确映射表:

  类别一 (结构化时序数值) → PostgreSQL (SQLite回退)
  类别二 (非结构化文本)   → ChromaDB + 文件系统原件
  类别三 (配置型)         → YAML + Redis热加载
  类别四 (实时瞬态)       → Redis (不持久化,仅缓存)

用法:
  python scripts/import_router.py --all --province 山东     # 全量导入
  python scripts/import_router.py --dry-run --province 山东  # 仅分析,不执行
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data"


# ═══════════════════════════════════════════════════════════════════
# 路由规则 —— 基于架构文档的四大类别精确映射
# ═══════════════════════════════════════════════════════════════════

ROUTING_RULES = {
    # ── 类别一: 结构化时序数值 → PostgreSQL ──
    "day_ahead_price_history": {
        "category": "类别一: 结构化时序数值",
        "primary": "PostgreSQL (price_data 表)",
        "secondary": None,
        "lifecycle": "永久保存",
        "agent_access": "SQL查询",
        "file_patterns": ["day_ahead_price_history", "price_data"],
    },
    "bidding_history": {
        "category": "类别一: 结构化时序数值",
        "primary": "PostgreSQL (bidding_records 表)",
        "secondary": None,
        "lifecycle": "永久保存",
        "agent_access": "SQL查询",
        "file_patterns": ["bidding_history"],
    },
    "forecast_archive": {
        "category": "类别一: 结构化时序数值",
        "primary": "PostgreSQL (forecast_archives 表)",
        "secondary": "ChromaDB (语义摘要, data_type=forecast_archive)",
        "lifecycle": "永久保存",
        "agent_access": "SQL查询 + RAG语义召回",
        "file_patterns": ["forecasts_", "forecast_"],
    },
    "rolling_match_history": {
        "category": "类别一: 结构化时序数值",
        "primary": "PostgreSQL (long_term_contracts 表)",
        "secondary": None,
        "lifecycle": "永久保存",
        "agent_access": "SQL查询",
        "file_patterns": ["rolling_match"],
    },
    "basis_history": {
        "category": "类别一: 结构化时序数值",
        "primary": "PostgreSQL (basis_history 表)",
        "secondary": None,
        "lifecycle": "永久保存",
        "agent_access": "SQL查询",
        "file_patterns": ["basis_history"],
    },
    "current_positions": {
        "category": "类别一: 结构化时序数值",
        "primary": "PostgreSQL (contract_holdings 表)",
        "secondary": None,
        "lifecycle": "永久保存",
        "agent_access": "SQL查询",
        "file_patterns": ["current_positions", "holdings"],
    },

    # ── 类别二: 非结构化文本 → ChromaDB + 文件系统 ──
    "policy_docs": {
        "category": "类别二: 非结构化文本",
        "primary": "ChromaDB (data_type=policy_doc)",
        "secondary": "文件系统 (PDF/TXT原件)",
        "lifecycle": "永久保存",
        "agent_access": "RAG语义检索",
        "file_patterns": ["政策", "policy", "规则", "办法", "通知", "细则"],
    },
    "expert_logs": {
        "category": "类别二: 非结构化文本",
        "primary": "ChromaDB (data_type=expert_insight)",
        "secondary": "文件系统 (CSV/MD原件)",
        "lifecycle": "永久保存",
        "agent_access": "RAG语义检索",
        "file_patterns": ["expert", "复盘", "心得", "经验", "交易记录"],
    },
    "market_snapshots": {
        "category": "类别二: 非结构化文本",
        "primary": "ChromaDB (data_type=market_snapshot)",
        "secondary": "文件系统 (JSON原件)",
        "lifecycle": "永久保存",
        "agent_access": "RAG语义检索",
        "file_patterns": ["market_snapshot", "行情快照"],
    },

    # ── 类别三: 配置型 → YAML + PostgreSQL + Redis ──
    "province_rules": {
        "category": "类别三: 配置型",
        "primary": "YAML 文件 (config/provinces/)",
        "secondary": "Redis Hash (热加载, province:{name})",
        "lifecycle": "一次配置+按需更新",
        "agent_access": "Redis微秒级读取",
        "file_patterns": [".yaml", "province"],
    },

    # ── 类别四: 实时瞬态 → Redis (不持久化) ──
    "realtime_prices": {
        "category": "类别四: 实时瞬态",
        "primary": "Redis (不持久化, 消费即弃)",
        "secondary": None,
        "lifecycle": "消费即弃",
        "agent_access": "实时流直接消费",
        "file_patterns": ["realtime_prices", "rt_price"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# 路由执行器
# ═══════════════════════════════════════════════════════════════════

def analyze_staging(province: str = "山东") -> dict:
    """分析 import_staging 中的所有文件, 返回路由分析结果"""
    staging = DATA / "source_data" / province
    if not staging.exists():
        return {"error": f"staging 目录不存在: {staging}"}

    results = {}
    unmatched = []

    # 递归查找所有文件
    all_files = list(staging.rglob("*"))
    all_files = [f for f in all_files if f.is_file() and f.suffix in (".csv", ".json", ".txt", ".md", ".pdf")]

    for file_path in all_files:
        fname = file_path.name
        matched = False

        for rule_key, rule in ROUTING_RULES.items():
            for pattern in rule["file_patterns"]:
                if pattern.lower() in fname.lower():
                    if rule_key not in results:
                        results[rule_key] = {"files": [], "rule": rule}
                    results[rule_key]["files"].append(str(file_path))
                    matched = True
                    break
            if matched:
                break

        if not matched:
            unmatched.append(str(file_path))

    return {
        "routed": results,
        "unmatched": unmatched,
        "total_files": len(all_files),
        "routed_count": sum(len(v["files"]) for v in results.values()),
    }


def import_to_chromadb(file_paths: list[str], province: str, data_type: str):
    """导入文本文件到 ChromaDB"""
    from tools.document_importer import DocumentImporter
    importer = DocumentImporter()

    for fp in file_paths:
        path = Path(fp)
        if path.is_dir():
            importer.import_directory(fp, province=province)
        elif path.suffix in (".txt", ".md", ".pdf", ".docx"):
            importer.import_file(fp, province=province)

    # 对于 JSON 行情快照, 用 MarketSnapshotArchiver
    for fp in file_paths:
        if fp.endswith(".json") and "snapshot" in fp.lower():
            snapshots = json.loads(Path(fp).read_text(encoding="utf-8"))
            from scripts.data_archiver import MarketSnapshotArchiver
            archiver = MarketSnapshotArchiver()
            for snap in (snapshots if isinstance(snapshots, list) else [snapshots]):
                archiver.archive_snapshot(snap)


def import_to_sqlite(file_paths: list[str], province: str, table_type: str):
    """导入结构化数据到 SQLite (PostgreSQL 的回退方案)"""
    import sqlite3

    db_path = str(PROJECT_ROOT / "data" / "sql" / "poweragent.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            continue

        if fp.endswith(".csv"):
            df = list(csv.DictReader(open(fp, encoding="utf-8")))

            if table_type == "price_history":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS price_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        province TEXT, date TEXT, avg_price REAL,
                        peak_avg REAL, valley_avg REAL, spread REAL,
                        max_price REAL, min_price REAL,
                        hourly_prices TEXT, base_load_mw REAL,
                        peak_load_mw REAL, wind_output_mw REAL,
                        solar_output_mw REAL, created_at TEXT
                    )
                """)
                for row in df:
                    conn.execute("""
                        INSERT OR REPLACE INTO price_data
                        (province, date, avg_price, peak_avg, valley_avg, spread, max_price, min_price, hourly_prices, base_load_mw, peak_load_mw, wind_output_mw, solar_output_mw, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get("province", province), row["date"],
                        row.get("avg_price", 0), row.get("peak_avg", 0),
                        row.get("valley_avg", 0), row.get("spread", 0),
                        row.get("max_price", 0), row.get("min_price", 0),
                        row.get("hourly_prices", "[]"),
                        row.get("base_load_mw", 0), row.get("peak_load_mw", 0),
                        row.get("wind_output_mw", 0), row.get("solar_output_mw", 0),
                        datetime.now().isoformat(),
                    ))
                print(f"  [SQLite] price_data: {len(df)} 行")

            elif table_type == "bidding_records":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bidding_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        province TEXT, trade_date TEXT,
                        bid_avg_price REAL, clearing_avg_price REAL,
                        deviation_pct REAL, deviation_status TEXT,
                        volume_mwh REAL, peak_bid_price REAL,
                        valley_bid_price REAL, strategy_type TEXT,
                        notes TEXT, created_at TEXT
                    )
                """)
                for row in df:
                    conn.execute("""
                        INSERT OR REPLACE INTO bidding_records
                        (province, trade_date, bid_avg_price, clearing_avg_price, deviation_pct, deviation_status, volume_mwh, peak_bid_price, valley_bid_price, strategy_type, notes, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get("province", province), row["trade_date"],
                        row.get("bid_avg_price", 0), row.get("clearing_avg_price", 0),
                        row.get("deviation_pct", 0), row.get("deviation_status", ""),
                        row.get("volume_mwh", 0), row.get("peak_bid_price", 0),
                        row.get("valley_bid_price", 0), row.get("strategy_type", ""),
                        row.get("notes", ""), datetime.now().isoformat(),
                    ))
                print(f"  [SQLite] bidding_records: {len(df)} 行")

            elif table_type == "rolling_match":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rolling_match_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        province TEXT, trade_date TEXT,
                        contract_month TEXT, contract_type TEXT,
                        volume_mwh REAL, deal_price REAL,
                        spot_reference REAL, basis REAL,
                        bid_count INTEGER, offer_count INTEGER,
                        created_at TEXT
                    )
                """)
                for row in df:
                    conn.execute("""
                        INSERT OR REPLACE INTO rolling_match_history
                        (province, trade_date, contract_month, contract_type, volume_mwh, deal_price, spot_reference, basis, bid_count, offer_count, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get("province", province), row["trade_date"],
                        row.get("contract_month", ""), row.get("contract_type", ""),
                        row.get("volume_mwh", 0), row.get("deal_price", 0),
                        row.get("spot_reference", 0), row.get("basis", 0),
                        row.get("bid_count", 0), row.get("offer_count", 0),
                        datetime.now().isoformat(),
                    ))
                print(f"  [SQLite] rolling_match: {len(df)} 行")

            elif table_type == "basis":
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS basis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        province TEXT, month TEXT,
                        spot_monthly_avg REAL,
                        long_term_contract_price REAL,
                        basis REAL, basis_pct REAL,
                        recommendation TEXT,
                        actual_next_month_spot REAL,
                        created_at TEXT
                    )
                """)
                for row in df:
                    conn.execute("""
                        INSERT OR REPLACE INTO basis_history
                        (province, month, spot_monthly_avg, long_term_contract_price, basis, basis_pct, recommendation, actual_next_month_spot, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get("province", province), row["month"],
                        row.get("spot_monthly_avg", 0), row.get("long_term_contract_price", 0),
                        row.get("basis", 0), row.get("basis_pct", 0),
                        row.get("recommendation", ""), row.get("actual_next_month_spot", 0),
                        datetime.now().isoformat(),
                    ))
                print(f"  [SQLite] basis_history: {len(df)} 行")

    conn.commit()
    conn.close()


def run_full_import(province: str = "山东", dry_run: bool = False):
    """全量导入: 分析 → 路由 → 导入"""
    print("=" * 60)
    print(f"PowerMind 数据导入路由器 — {province}")
    print("基于 ARCHITECTURE_COMPLETE.md 四大类别精确映射")
    print("=" * 60)
    print()

    # Step 1: 分析
    analysis = analyze_staging(province)
    if "error" in analysis:
        print(f"[错误] {analysis['error']}")
        return

    print(f"[分析] 发现 {analysis['total_files']} 个文件, "
          f"匹配 {analysis['routed_count']} 个, "
          f"未匹配 {len(analysis['unmatched'])} 个")
    print()

    # Step 2: 显示路由结果
    for rule_key, data in analysis["routed"].items():
        rule = data["rule"]
        print(f"  [{rule['category']}] {rule_key}")
        for f in data["files"]:
            print(f"    → {f}")
        print(f"    主存储: {rule['primary']}")
        if rule["secondary"]:
            print(f"    辅存储: {rule['secondary']}")
        print(f"    生命周期: {rule['lifecycle']}")
        print(f"    Agent读取: {rule['agent_access']}")
        print()

    if analysis["unmatched"]:
        print("[未匹配] 以下文件未找到路由规则:")
        for f in analysis["unmatched"]:
            print(f"  ⚠ {f}")
        print()

    if dry_run:
        print("[dry-run] 仅分析，不执行导入。")
        return

    # Step 3: 执行导入
    print("=" * 60)
    print("执行导入...")
    print("=" * 60)
    print()

    for rule_key, data in analysis["routed"].items():
        rule = data["rule"]
        files = data["files"]
        print(f"[导入] {rule_key} ({len(files)} 个文件)")

        if "ChromaDB" in rule["primary"]:
            print(f"  → ChromaDB (data_type 自动检测)")
            import_to_chromadb(files, province, rule_key)

        elif "PostgreSQL" in rule["primary"] or "SQLite" in (rule.get("secondary") or ""):
            print(f"  → SQLite (PostgreSQL 回退, {rule_key})")
            table_map = {
                "day_ahead_price_history": "price_history",
                "bidding_history": "bidding_records",
                "rolling_match_history": "rolling_match",
                "basis_history": "basis",
            }
            table_type = table_map.get(rule_key, rule_key)
            import_to_sqlite(files, province, table_type)

        elif "Redis" in rule["primary"]:
            print(f"  → 跳过 (Redis 实时数据不持久化)")

        elif "YAML" in rule["primary"]:
            print(f"  → 跳过 (YAML 配置文件, 无需导入)")

    print()
    print("=" * 60)
    print("导入完成！")
    print("=" * 60)
    print()
    print("验证命令:")
    print("  python tools/document_importer.py --list  # ChromaDB 验证")
    print("  python scripts/data_archiver.py --forecast # 预测归档验证")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PowerMind 数据导入路由器")
    parser.add_argument("--all", action="store_true", help="全量导入")
    parser.add_argument("--dry-run", action="store_true", help="仅分析路由,不执行导入")
    parser.add_argument("--province", "-p", default="山东")

    args = parser.parse_args()

    if args.all or args.dry_run:
        run_full_import(province=args.province, dry_run=args.dry_run)
    else:
        parser.print_help()
        print("\n推荐:")
        print("  python scripts/import_router.py --dry-run  # 先看看分析结果")
        print("  python scripts/import_router.py --all       # 全量导入")
