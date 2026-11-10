"""Infrastructure Settings — environment-driven configuration.

Reads from environment variables with sensible defaults for local dev.
Production overrides are injected via .env or container orchestration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(key: str, default: str = "") -> str:
    """Read env var, falling back to .env file if present."""
    value = os.environ.get(key, "")
    if value:
        return value
    # Try .env file
    env_file = PROJECT_ROOT / "config" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


@dataclass
class Settings:
    """Central configuration for PowerMind infrastructure.

    All values are read from environment variables with fallback defaults.
    """

    # ── Application ──
    app_name: str = "PowerMind"
    app_env: str = _env("APP_ENV", "development")  # development | staging | production
    debug: bool = _env("DEBUG", "false").lower() == "true"

    # ── PostgreSQL ──
    pg_host: str = _env("PG_HOST", "localhost")
    pg_port: int = int(_env("PG_PORT", "5432"))
    pg_database: str = _env("PG_DATABASE", "poweragent")
    pg_user: str = _env("PG_USER", "poweragent")
    pg_password: str = _env("PG_PASSWORD", "poweragent")
    pg_pool_min_size: int = int(_env("PG_POOL_MIN_SIZE", "5"))
    pg_pool_max_size: int = int(_env("PG_POOL_MAX_SIZE", "50"))
    pg_pool_max_idle: int = int(_env("PG_POOL_MAX_IDLE", "300"))  # seconds
    pg_ssl_mode: str = _env("PG_SSL_MODE", "prefer")

    @property
    def pg_dsn(self) -> str:
        """Async PostgreSQL connection string (asyncpg driver)."""
        return (
            f"postgresql+asyncpg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
            f"?ssl={self.pg_ssl_mode}"
        )

    @property
    def pg_dsn_sync(self) -> str:
        """Sync PostgreSQL connection string (psycopg2 driver, for migrations)."""
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    # ── Redis ──
    redis_host: str = _env("REDIS_HOST", "localhost")
    redis_port: int = int(_env("REDIS_PORT", "6379"))
    redis_password: str = _env("REDIS_PASSWORD", "")
    redis_db: int = int(_env("REDIS_DB", "0"))
    redis_max_connections: int = int(_env("REDIS_MAX_CONNECTIONS", "50"))

    @property
    def redis_dsn(self) -> str:
        """Redis connection string."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── Feature flags ──
    use_postgres: bool = _env("USE_POSTGRES", "true").lower() == "true"
    use_redis: bool = _env("USE_REDIS", "true").lower() == "true"

    # ── Checkpoint retention ──
    checkpoint_retain_count: int = int(_env("CHECKPOINT_RETAIN_COUNT", "20"))
    checkpoint_prune_interval_hours: int = int(_env("CHECKPOINT_PRUNE_INTERVAL_HOURS", "6"))

    # ── Memory TTL (seconds) ──
    working_memory_ttl: int = int(_env("WORKING_MEMORY_TTL", "3600"))     # 1 hour
    episodic_memory_ttl: int = int(_env("EPISODIC_MEMORY_TTL", "2592000"))  # 30 days
    working_memory_max_entries: int = int(_env("WORKING_MEMORY_MAX", "20"))

    # ── Paths ──
    project_root: Path = PROJECT_ROOT
    provinces_dir: Path = PROJECT_ROOT / "config" / "provinces"
    data_dir: Path = PROJECT_ROOT / "data"


@lru_cache
def get_settings() -> Settings:
    """Singleton settings accessor."""
    return Settings()
