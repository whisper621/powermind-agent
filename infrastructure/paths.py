"""Centralised filesystem paths for the PowerMind project.

All code that needs to resolve filesystem paths MUST import from this module.
Hardcoded paths (PROJECT_ROOT / "data" / "sql" / ...) are forbidden in active code.

Usage:
    from infrastructure.paths import get_db_path, get_vectordb_path, DATA_DIR
"""

from __future__ import annotations

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Data directory layout ─────────────────────────────────────────────────
# data/
#   demo/         Demo seed/sample data (rebuildable)
#   knowledge/    Formal knowledge sources: policies, rules, docs
#   sqlite/       SQLite database files (powermind_demo.db, forecast_archives.db)
#   vectordb/     ChromaDB vector store
#   runtime/      Runtime logs, audit, memory, cache (NOT version-controlled)

DATA_DIR = PROJECT_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
SQLITE_DIR = DATA_DIR / "sqlite"
VECTORDB_DIR = DATA_DIR / "vectordb"
RUNTIME_DIR = DATA_DIR / "runtime"

# ── Config directory ──────────────────────────────────────────────────────
CONFIG_DIR = PROJECT_ROOT / "config"
FORECAST_CONFIG_PATH = CONFIG_DIR / "forecast.yaml"

# ── Prompt directory ──────────────────────────────────────────────────────
PROMPTS_DIR = PROJECT_ROOT / "prompts"
PROMPT_REGISTRY_PATH = PROMPTS_DIR / "prompt_registry.yaml"

# ── Intent directory ──────────────────────────────────────────────────────
INTENTS_DIR = PROJECT_ROOT / "intents"
TASK_INTENTS_PATH = INTENTS_DIR / "task_intents.yaml"

# ── Database paths ────────────────────────────────────────────────────────

def get_db_path(db_name: str = "powermind_demo.db") -> Path:
    """Return path to an SQLite database in the sqlite directory."""
    SQLITE_DIR.mkdir(parents=True, exist_ok=True)
    return SQLITE_DIR / db_name


def get_vectordb_path() -> Path:
    """Return path to the ChromaDB vector store directory."""
    return VECTORDB_DIR


def get_runtime_path(filename: str) -> Path:
    """Return path to a runtime file (logs, audit, memory)."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR / filename


def get_knowledge_path(*parts: str) -> Path:
    """Return path within the knowledge directory."""
    p = KNOWLEDGE_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── Legacy path aliases (for migration scripts only) ──────────────────────
LEGACY_SQL_DIR = DATA_DIR / "sql"  # pre-unification path — deprecated
LEGACY_SOURCE_DOCS_DIR = DATA_DIR / "source_docs"  # → data/knowledge/source_docs
LEGACY_LOGS_DIR = DATA_DIR / "logs"  # → data/runtime/


def ensure_directories() -> None:
    """Create all required runtime directories if they don't exist."""
    for d in (DEMO_DIR, KNOWLEDGE_DIR, SQLITE_DIR, VECTORDB_DIR, RUNTIME_DIR):
        d.mkdir(parents=True, exist_ok=True)
