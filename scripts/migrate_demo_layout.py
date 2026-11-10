#!/usr/bin/env python3
"""One-shot migration from old data/ layout to new unified layout.

Run once:  python scripts/migrate_demo_layout.py
Safe to re-run — uses copy, never deletes originals.
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"


def migrate():
    migrations = [
        # (source, dest, description)
        (DATA / "sql", DATA / "sqlite", "SQLite databases"),
        (DATA / "logs", DATA / "runtime", "Runtime logs"),
        (DATA / "audit_log.jsonl", DATA / "runtime" / "audit_log.jsonl", "Audit log"),
        (DATA / "episodic_memory.jsonl", DATA / "runtime" / "episodic_memory.jsonl", "Episodic memory"),
    ]

    # source_docs → knowledge/source_docs
    src_docs = DATA / "source_docs"
    dest_docs = DATA / "knowledge" / "source_docs"
    if src_docs.exists() and not (dest_docs / "policies").exists():
        print(f"Copy: {src_docs} → {dest_docs}")
        dest_docs.parent.mkdir(parents=True, exist_ok=True)
        if dest_docs.exists():
            shutil.rmtree(dest_docs)
        shutil.copytree(src_docs, dest_docs)

    for src, dest, desc in migrations:
        if not src.exists():
            print(f"Skip (missing): {src}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            for item in src.iterdir():
                target = dest / item.name
                if not target.exists():
                    print(f"Copy: {item} → {target}")
                    if item.is_dir():
                        shutil.copytree(item, target)
                    else:
                        shutil.copy2(item, target)
        else:
            if not dest.exists():
                print(f"Copy: {src} → {dest}")
                shutil.copy2(src, dest)

    print("\nMigration complete.")
    print("Old paths (data/sql/, data/logs/, etc.) are preserved for reference.")
    print("Active code should now use data/sqlite/, data/runtime/, data/knowledge/.")
    print("See data/README.md for the full layout.")


if __name__ == "__main__":
    migrate()
