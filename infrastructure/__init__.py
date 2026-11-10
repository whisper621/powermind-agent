"""PowerAgent OS — Infrastructure Layer.

Enterprise-grade database, cache, and multi-tenant isolation primitives.
Replaces MVP SQLite with PostgreSQL + Redis for SaaS-scale deployment.

Modules:
  - settings:    Environment-driven configuration
  - database:    Async PostgreSQL connection pool + session factory
  - tenant:      Row-Level Security (RLS) multi-tenant isolation
  - cache:       Redis distributed cache for rules + memory
  - checkpoint:  AsyncPostgresSaver for LangGraph with pruning
  - migrations:  Schema creation with RLS policies
"""

from infrastructure.settings import Settings, get_settings
from infrastructure.database import (
    DatabaseManager,
    get_tenant_session,
    init_database,
)
from infrastructure.tenant import (
    TenantContext,
    TenantIsolation,
    get_tenant_context,
)
from infrastructure.cache import (
    RedisCache,
    ProvinceConfigCache,
    MemoryCache,
    get_cache,
)
from infrastructure.checkpoint import (
    CheckpointManager,
    get_checkpointer,
    prune_checkpoints,
)

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # Database
    "DatabaseManager",
    "get_tenant_session",
    "init_database",
    # Tenant
    "TenantContext",
    "TenantIsolation",
    "get_tenant_context",
    # Cache
    "RedisCache",
    "ProvinceConfigCache",
    "MemoryCache",
    "get_cache",
    # Checkpoint
    "CheckpointManager",
    "get_checkpointer",
    "prune_checkpoints",
]
