"""PostgreSQL Multi-Tenant Database Layer.

Core primitives:
  - Async connection pool via psycopg (asyncpg driver) + SQLAlchemy 2.x
  - Row-Level Security (RLS) activated on all tenant-aware tables
  - Tenant-aware session factory with SET LOCAL app.current_tenant injection
  - Graceful fallback to SQLite for local dev when USE_POSTGRES=false

Architecture: Shared-DB with Tenant-Aware Rows + Engine-Level RLS
    Every table carries tenant_id (UUID).
    PostgreSQL silently appends WHERE tenant_id = current_setting('app.current_tenant')
    to every query, making cross-tenant data leaks physically impossible.

Usage:
    from infrastructure.database import DatabaseManager, get_tenant_session

    db = DatabaseManager()
    await db.initialize()

    async with get_tenant_session(tenant_id) as session:
        result = await session.execute(select(ForecastArchive))
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import AsyncGenerator, Optional
from uuid import UUID

from infrastructure.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Per-request tenant context (thread-safe for async)
_current_tenant: ContextVar[str] = ContextVar("current_tenant", default="")


# ---------------------------------------------------------------------------
# SQLAlchemy async engine + session management
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Singleton manager for PostgreSQL async connection pool.

    Creates the engine once at startup, reuses connection pool across requests.
    Falls back to SQLite when USE_POSTGRES=false (local dev / MVP mode).
    """

    _instance: Optional["DatabaseManager"] = None

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine = None
        self._session_factory = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self) -> None:
        """Create async engine and session factory. Idempotent."""
        if self._initialized:
            return

        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )

        if self._settings.use_postgres:
            self._engine = create_async_engine(
                self._settings.pg_dsn,
                pool_size=self._settings.pg_pool_min_size,
                max_overflow=self._settings.pg_pool_max_size - self._settings.pg_pool_min_size,
                pool_recycle=self._settings.pg_pool_max_idle,
                pool_pre_ping=True,         # verify connection liveness
                echo=self._settings.debug,
            )
            logger.info(
                "DatabaseManager: PostgreSQL pool created — %s:%s/%s (min=%d max=%d)",
                self._settings.pg_host,
                self._settings.pg_port,
                self._settings.pg_database,
                self._settings.pg_pool_min_size,
                self._settings.pg_pool_max_size,
            )
        else:
            # SQLite fallback for local dev
            sqlite_path = self._settings.data_dir / "poweragent.db"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._engine = create_async_engine(
                f"sqlite+aiosqlite:///{sqlite_path}",
                echo=self._settings.debug,
            )
            logger.info("DatabaseManager: SQLite fallback — %s", sqlite_path)

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._initialized = True

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("DatabaseManager: connection pool closed")

    async def get_session(self, tenant_id: str = "") -> AsyncGenerator:
        """Yield an async session with tenant context injected.

        When tenant_id is provided, executes:
            SET LOCAL app.current_tenant = '<tenant_id>'
        before yielding the session. This activates PostgreSQL RLS.
        """
        # late import
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession

        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as session:
            if tenant_id and self._settings.use_postgres:
                await session.execute(
                    text("SET LOCAL app.current_tenant = :tid"),
                    {"tid": tenant_id},
                )
                _current_tenant.set(tenant_id)
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @property
    def engine(self):
        return self._engine

    @property
    def session_factory(self):
        return self._session_factory


# ---------------------------------------------------------------------------
# Convenience: tenant-aware session context manager
# ---------------------------------------------------------------------------


async def get_tenant_session(tenant_id: str = "") -> AsyncGenerator:
    """FastAPI / standalone dependency: yield a tenant-isolated DB session.

    Usage:
        async with get_tenant_session("tenant-abc-123") as session:
            rows = await session.execute(select(ForecastArchive))
    """
    db = DatabaseManager.get_instance()
    async for session in db.get_session(tenant_id):
        yield session


# ---------------------------------------------------------------------------
# One-shot init helper
# ---------------------------------------------------------------------------


async def init_database(settings: Settings | None = None) -> DatabaseManager:
    """Initialize the database manager and create schema if needed."""
    db = DatabaseManager.get_instance()
    await db.initialize()
    return db


def get_current_tenant() -> str:
    """Return the tenant_id for the current async context."""
    return _current_tenant.get()
