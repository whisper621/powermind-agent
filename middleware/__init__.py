"""PowerAgent OS — Middleware Layer.

Audit logging, dynamic prompts, tenant isolation, and pipeline monitoring.
"""

from middleware.audit_logger import AuditLogger, get_audit_logger
from middleware.dynamic_prompt import PromptRegistry, get_prompt_registry
from middleware.tenant_isolation import (
    TenantContext,
    TenantIsolatedDB,
    TenantIsolatedRetrieval,
    tenant_guard,
    tenant_guard_company,
)
from middleware.monitor import (
    PipelineMonitor,
    PipelineHealth,
    StageMetrics,
    DegradationHandler,
)
from middleware.degradation_policy import (
    DegradationAction,
    DegradationDecision,
    DegradationEvent,
    DegradationPolicy,
    FailureContext,
    FailureType,
    classify_failure,
)

__all__ = [
    # Audit
    "AuditLogger",
    "get_audit_logger",
    # Prompts
    "PromptRegistry",
    "get_prompt_registry",
    # Tenant
    "TenantContext",
    "TenantIsolatedDB",
    "TenantIsolatedRetrieval",
    "tenant_guard",
    "tenant_guard_company",
    # Monitor
    "PipelineMonitor",
    "PipelineHealth",
    "StageMetrics",
    "DegradationHandler",
    # Deterministic fault policy
    "DegradationAction",
    "DegradationDecision",
    "DegradationEvent",
    "DegradationPolicy",
    "FailureContext",
    "FailureType",
    "classify_failure",
]
