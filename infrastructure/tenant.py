"""Tenant Isolation — Multi-tenant context propagation and middleware.

Core components:
  - TenantContext:      Per-request tenant identity + allowed resources
  - TenantIsolation:    FastAPI-compatible dependency that extracts tenant from JWT/header
  - get_tenant_context: ContextVar-based accessor for the current tenant

RLS Integration:
    After resolve(), call DatabaseManager.get_session(tenant_id) which runs
    SET LOCAL app.current_tenant = '<uuid>' — activating PostgreSQL RLS.
    Every subsequent SQL query is PHYSICALLY filtered by the database engine.

Usage (FastAPI):
    from infrastructure.tenant import TenantIsolation

    tenant_iso = TenantIsolation()

    @app.get("/api/forecast")
    async def get_forecast(
        tenant: TenantContext = Depends(tenant_iso),
        session = Depends(get_tenant_db),
    ):
        rows = await session.execute(select(ForecastArchive))
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

# Per-async-request tenant identity
_current_tenant_ctx: ContextVar[Optional["TenantContext"]] = ContextVar(
    "tenant_context", default=None
)


@dataclass
class TenantContext:
    """Resolved tenant identity for the current request.

    Extracted from JWT token / API key / header by TenantIsolation middleware.
    Injected into PostgreSQL via SET LOCAL app.current_tenant before queries.
    """

    tenant_id: str
    tenant_name: str = ""
    plan_tier: str = "basic"  # basic | pro | enterprise
    allowed_provinces: list[str] = field(default_factory=list)
    allowed_companies: list[str] = field(default_factory=list)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    is_system: bool = False  # system tenant bypasses some RLS checks

    @property
    def tenant_uuid(self) -> str:
        """Return tenant_id as UUID string for PostgreSQL."""
        try:
            uuid.UUID(self.tenant_id)
            return self.tenant_id
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, self.tenant_id))

    def can_access_province(self, province: str) -> bool:
        """Check if this tenant is allowed to query data for a province."""
        if not self.allowed_provinces:
            return False
        return province in self.allowed_provinces or "全国" in self.allowed_provinces

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "plan_tier": self.plan_tier,
            "allowed_provinces": self.allowed_provinces,
            "request_id": self.request_id,
        }

    @classmethod
    def system(cls) -> "TenantContext":
        """Create a system-level context (bypasses tenant filters)."""
        return cls(
            tenant_id="00000000-0000-0000-0000-000000000000",
            tenant_name="system",
            plan_tier="enterprise",
            is_system=True,
        )

    @classmethod
    def demo(cls, province: str = "山东") -> "TenantContext":
        """Create a demo tenant context for local development."""
        return cls(
            tenant_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "demo-tenant")),
            tenant_name="Demo Tenant",
            plan_tier="pro",
            allowed_provinces=[province, "全国"],
        )


class TenantIsolation:
    """FastAPI dependency: extract tenant identity from request headers.

    Resolution order:
      1. X-Tenant-ID header (production)
      2. X-API-Key header → lookup tenant_id (API access)
      3. Authorization Bearer token → decode JWT (user sessions)
      4. Default demo tenant (local development)

    Usage:
        tenant_iso = TenantIsolation()
        app.include_router(router, dependencies=[Depends(tenant_iso)])
    """

    HEADER_TENANT_ID = "X-Tenant-ID"
    HEADER_API_KEY = "X-API-Key"

    def __init__(self, require_auth: bool = False):
        self._require_auth = require_auth

    async def __call__(self) -> TenantContext:  # FastAPI Depends() compatible
        """Resolve tenant context from current HTTP request."""
        # Try to import FastAPI's request context
        try:
            from starlette.requests import Request
            from starlette_context import context
            request: Request = context.get("request")
        except (ImportError, RuntimeError):
            request = None

        tenant_id = ""
        tenant_name = ""

        if request:
            # 1. Direct tenant-id header
            tenant_id = request.headers.get(self.HEADER_TENANT_ID, "")

            # 2. API key lookup
            if not tenant_id:
                api_key = request.headers.get(self.HEADER_API_KEY, "")
                if api_key:
                    tenant_id, tenant_name = await self._lookup_api_key(api_key)

            # 3. JWT token decode
            if not tenant_id:
                auth_header = request.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    tenant_id, tenant_name = self._decode_jwt_tenant(auth_header[7:])

        # 4. Fallback: demo tenant for local dev
        if not tenant_id:
            ctx = TenantContext.demo()
        else:
            ctx = TenantContext(
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                allowed_provinces=["山东", "全国"],  # TODO: load from tenant service
            )

        _current_tenant_ctx.set(ctx)
        return ctx

    async def _lookup_api_key(self, api_key: str) -> tuple[str, str]:
        """Resolve API key → tenant_id from database/cache."""
        # Placeholder: production would query Redis or PostgreSQL
        return ("", "")

    @staticmethod
    def _decode_jwt_tenant(token: str) -> tuple[str, str]:
        """Extract tenant_id from JWT claims."""
        # Placeholder: production would use python-jose / PyJWT
        try:
            import jwt
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload.get("tenant_id", ""), payload.get("tenant_name", "")
        except Exception:
            return ("", "")


def get_tenant_context() -> TenantContext:
    """Get the current request's tenant context. Safe for non-HTTP contexts."""
    ctx = _current_tenant_ctx.get()
    if ctx is None:
        ctx = TenantContext.demo()
        _current_tenant_ctx.set(ctx)
    return ctx
