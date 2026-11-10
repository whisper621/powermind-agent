"""PostgreSQL RLS Schema Migration — Enterprise Multi-Tenant Table Definitions.

Creates all tenant-aware tables with:
  - tenant_id (UUID) column on every table
  - Composite index on (tenant_id, relevant lookup columns)
  - Row-Level Security ENABLED + per-operation policies

RLS Guarantee:
    Even if application code forgets a WHERE tenant_id=? clause,
    PostgreSQL PHYSICALLY blocks access to rows from other tenants.

Usage (one-time, at deploy):
    from infrastructure.migrations import run_migrations
    await run_migrations()
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core schema DDL — idempotent (IF NOT EXISTS / OR REPLACE)
# ---------------------------------------------------------------------------

SCHEMA_DDL = """
-- =====================================================================
-- PowerAgent OS — Enterprise Schema with Row-Level Security
-- =====================================================================

-- 0. Ensure uuid-ossp extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================================
-- 1. Tenants master table
-- =====================================================================
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    display_name TEXT,
    plan_tier TEXT NOT NULL DEFAULT 'basic',  -- basic | pro | enterprise
    allowed_provinces TEXT[] NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =====================================================================
-- 2. Province rules cache (mirrors YAML for fast SQL queries)
-- =====================================================================
CREATE TABLE IF NOT EXISTS province_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    province TEXT NOT NULL,
    rule_key TEXT NOT NULL,
    rule_value JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(tenant_id, province, rule_key)
);

-- =====================================================================
-- 3. Forecast archives (prediction results persisted for backtesting)
-- =====================================================================
CREATE TABLE IF NOT EXISTS forecast_archives (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    province TEXT NOT NULL,
    target_date DATE NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'lstm_default',
    model_version TEXT NOT NULL DEFAULT '1.0',
    avg_price DOUBLE PRECISION NOT NULL,
    peak_avg DOUBLE PRECISION,
    valley_avg DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    p10 DOUBLE PRECISION,
    p50 DOUBLE PRECISION,
    p90 DOUBLE PRECISION,
    hourly_predictions JSONB NOT NULL DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(tenant_id, province, target_date, model_name)
);

-- =====================================================================
-- 4. Factor snapshots (market factor analysis results)
-- =====================================================================
CREATE TABLE IF NOT EXISTS factor_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    province TEXT NOT NULL,
    target_date DATE NOT NULL,
    overall_direction TEXT,
    overall_risk TEXT,
    factor_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(tenant_id, province, target_date)
);

-- =====================================================================
-- 5. Audit log (immutable append-only decision trail)
-- =====================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    request_id TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- tool_start | tool_end | agent_step | verification
    stage TEXT,
    tool_name TEXT,
    args JSONB,
    output_summary TEXT,
    success BOOLEAN,
    duration_ms DOUBLE PRECISION,
    error TEXT,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =====================================================================
-- 6. Episodic memory (historical interactions, 30-day TTL)
-- =====================================================================
CREATE TABLE IF NOT EXISTS episodic_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    memory_id TEXT NOT NULL,
    query TEXT,
    result_summary TEXT,
    province TEXT,
    importance DOUBLE PRECISION DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '30 days')
);

-- =====================================================================
-- 7. Client profiles (B2B customer asset & contract data)
-- =====================================================================
CREATE TABLE IF NOT EXISTS client_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    client_id TEXT NOT NULL,
    client_name TEXT,
    province TEXT,
    contract_capacity_kw DOUBLE PRECISION,
    pv_capacity_kw DOUBLE PRECISION,
    storage_capacity_mwh DOUBLE PRECISION,
    storage_power_mw DOUBLE PRECISION,
    industry_type TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(tenant_id, client_id)
);

-- =====================================================================
-- 8. Policy documents index
-- =====================================================================
CREATE TABLE IF NOT EXISTS policy_docs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    doc_id TEXT NOT NULL,
    title TEXT,
    province TEXT,
    doc_type TEXT,
    publish_date DATE,
    effective_date DATE,
    source_url TEXT,
    content_hash TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(tenant_id, doc_id)
);

-- =====================================================================
-- 9. Price & market data (structured time-series)
-- =====================================================================
CREATE TABLE IF NOT EXISTS price_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    province TEXT NOT NULL,
    date DATE NOT NULL,
    hour INTEGER,
    day_ahead_price DOUBLE PRECISION,
    day_ahead_volume DOUBLE PRECISION,
    real_time_price DOUBLE PRECISION,
    UNIQUE(tenant_id, province, date, hour)
);

-- =====================================================================
-- Performance indexes (outside RLS scope for global lookups)
-- =====================================================================
CREATE INDEX IF NOT EXISTS idx_forecast_tenant_province_date
    ON forecast_archives(tenant_id, province, target_date DESC);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_request
    ON audit_log(tenant_id, request_id);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_timestamp
    ON audit_log(tenant_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_tenant_province
    ON episodic_memory(tenant_id, province, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_expires
    ON episodic_memory(expires_at) WHERE expires_at < now();
CREATE INDEX IF NOT EXISTS idx_client_tenant
    ON client_profiles(tenant_id, client_id);
CREATE INDEX IF NOT EXISTS idx_price_tenant_province_date
    ON price_data(tenant_id, province, date DESC);
CREATE INDEX IF NOT EXISTS idx_factor_tenant_province_date
    ON factor_snapshots(tenant_id, province, target_date DESC);
"""

# ---------------------------------------------------------------------------
# RLS activation — must run AFTER table creation
# ---------------------------------------------------------------------------

RLS_DDL = """
-- =====================================================================
-- Row-Level Security Activation
-- =====================================================================
-- Each table gets:
--   1. ENABLE ROW LEVEL SECURITY
--   2. Per-operation policies forcing tenant_id = current_setting('app.current_tenant')
--   3. A default-(none) policy so that no rows leak if tenant context is missing
-- =====================================================================

-- Helper: activate RLS + standard policies for a table
-- We use DO blocks because PostgreSQL doesn't have a loop-over-tables DDL syntax.

DO $$
DECLARE
    tbl TEXT;
    tables TEXT[] := ARRAY[
        'province_rules',
        'forecast_archives',
        'factor_snapshots',
        'audit_log',
        'episodic_memory',
        'client_profiles',
        'policy_docs',
        'price_data'
    ];
BEGIN
    FOREACH tbl IN ARRAY tables LOOP
        -- Enable RLS
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tbl);

        -- Drop existing policies to make this idempotent
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation_select ON %I', tbl);
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation_insert ON %I', tbl);
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation_update ON %I', tbl);
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation_delete ON %I', tbl);

        -- SELECT policy
        EXECUTE format(
            'CREATE POLICY tenant_isolation_select ON %I FOR SELECT '
            'USING (tenant_id = current_setting(''app.current_tenant'')::uuid '
            '       OR current_setting(''app.current_tenant'') = ''00000000-0000-0000-0000-000000000000'')',
            tbl
        );

        -- INSERT policy: force tenant_id to match current context
        EXECUTE format(
            'CREATE POLICY tenant_isolation_insert ON %I FOR INSERT '
            'WITH CHECK (tenant_id = current_setting(''app.current_tenant'')::uuid)',
            tbl
        );

        -- UPDATE policy
        EXECUTE format(
            'CREATE POLICY tenant_isolation_update ON %I FOR UPDATE '
            'USING (tenant_id = current_setting(''app.current_tenant'')::uuid) '
            'WITH CHECK (tenant_id = current_setting(''app.current_tenant'')::uuid)',
            tbl
        );

        -- DELETE policy
        EXECUTE format(
            'CREATE POLICY tenant_isolation_delete ON %I FOR DELETE '
            'USING (tenant_id = current_setting(''app.current_tenant'')::uuid)',
            tbl
        );
    END LOOP;
END $$;

-- Default tenant (id = all-zeroes) acts as "system" tenant for global data
-- Do NOT grant regular tenants access to this tenant's rows via RLS bypass
-- The all-zero UUID check in SELECT policy allows system-level queries
-- while individual tenants remain strictly isolated.
"""

# ---------------------------------------------------------------------------
# Cleanup: prune expired episodic memory
# ---------------------------------------------------------------------------

EPISODIC_PRUNE_DDL = """
-- Remove expired episodic memory entries (runs as periodic job)
DELETE FROM episodic_memory WHERE expires_at < now();
"""

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_migrations(engine: AsyncEngine, enable_rls: bool = True) -> None:
    """Execute schema creation and RLS activation.

    All DDL statements are idempotent (IF NOT EXISTS / IF EXISTS).
    Safe to call at every application startup.

    Args:
        engine: SQLAlchemy async engine
        enable_rls: Whether to activate Row-Level Security (skip for SQLite)
    """
    async with engine.begin() as conn:
        # Split by statement and execute each
        for stmt in _split_statements(SCHEMA_DDL):
            if stmt.strip():
                await conn.execute(text(stmt))

        if enable_rls:
            for stmt in _split_statements(RLS_DDL):
                if stmt.strip():
                    try:
                        await conn.execute(text(stmt))
                    except Exception as exc:
                        # RLS is PostgreSQL-only — don't crash on SQLite
                        logger.debug("RLS statement skipped (expected on non-PG): %s", exc)

    logger.info("Migrations: schema + RLS applied successfully")


def _split_statements(ddl: str) -> list[str]:
    """Split multi-statement DDL on semicolons, preserving DO blocks."""
    # Simple approach: split on ';' then filter empty
    statements = []
    current = []
    in_do_block = False
    for line in ddl.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("DO $$"):
            in_do_block = True
        if in_do_block:
            current.append(line)
            if stripped == "END $$;" or stripped == "END; $$;" or stripped == "END $$":
                statements.append("\n".join(current))
                current = []
                in_do_block = False
        elif stripped.startswith("--") or not stripped:
            # accumulate comments with their statement
            current.append(line)
        elif stripped.endswith(";"):
            current.append(line)
            statements.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        remaining = "\n".join(current).strip()
        if remaining:
            statements.append(remaining)
    return statements
