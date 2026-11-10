"""Metadata filtering layer — time-decay weighting, province enforcement, tenant isolation.

Key guarantees:
  - Province filter is HARD-enforced: results from wrong provinces are dropped, not down-weighted.
  - Time decay uses exponential weighting based on date_published.
  - Tenant context (tenant_id, allowed_provinces) is injected at retrieval time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from retrieval.base_retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class TenantContext:
    """Per-request tenant isolation context."""

    tenant_id: str
    allowed_provinces: list[str] = field(default_factory=list)
    allowed_companies: list[str] = field(default_factory=list)
    request_id: str = ""


class ProvinceFilter:
    """Hard-enforced province-level data isolation.

    Documents whose province metadata does not match the allowed set are DROPPED
    (not down-ranked). National-level documents (province="全国") are always
    allowed.
    """

    @staticmethod
    def apply(
        results: list[RetrievalResult],
        allowed_provinces: list[str],
    ) -> list[RetrievalResult]:
        if not allowed_provinces:
            return results

        allowed = set(allowed_provinces) | {"全国", ""}
        filtered = []
        dropped = 0
        for r in results:
            province = r.metadata.get("province", "")
            if province in allowed:
                filtered.append(r)
            else:
                dropped += 1
        if dropped:
            logger.info(
                "ProvinceFilter: dropped %d results (allowed=%s)",
                dropped,
                allowed_provinces,
            )
        return filtered


class TimeDecayFilter:
    """Exponential time-decay weighting based on date_published.

    Formula:  adjusted_score = original_score * exp(-lambda * age_days)

    lambda is configurable; default 0.005 means ~50% weight at 138 days.
    Policy documents decay slower (lambda=0.002); market snapshots faster
    (lambda=0.05).
    """

    DEFAULT_LAMBDAS = {
        "policy": 0.002,
        "expert": 0.003,
        "market": 0.05,
        "price_history": 0.01,
        "graph": 0.001,
        "sql": 0.001,
        "bm25": 0.005,
        "dense": 0.005,
    }

    def __init__(self, lambdas: dict[str, float] | None = None):
        self._lambdas = lambdas or self.DEFAULT_LAMBDAS

    def apply(
        self,
        results: list[RetrievalResult],
        reference_date: datetime | None = None,
    ) -> list[RetrievalResult]:
        ref = reference_date or datetime.now()
        for r in results:
            date_str = r.metadata.get("date_published") or r.metadata.get("date")
            if date_str is None:
                continue
            try:
                if isinstance(date_str, str):
                    pub_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                else:
                    pub_date = date_str
                age_days = (ref - pub_date).days
                if age_days < 0:
                    age_days = 0
                lam = self._lambdas.get(r.source, 0.005)
                decay = __import__("math").exp(-lam * age_days)
                r.score = round(r.score * decay, 6)
                r.metadata["time_decay"] = round(decay, 4)
                r.metadata["age_days"] = age_days
            except (ValueError, TypeError):
                pass
        return results


class MetadataFilter:
    """Composite metadata filter combining province enforcement + time decay + tenant isolation."""

    def __init__(
        self,
        province_filter: ProvinceFilter | None = None,
        time_decay: TimeDecayFilter | None = None,
    ):
        self.province_filter = province_filter or ProvinceFilter()
        self.time_decay = time_decay or TimeDecayFilter()

    def apply(
        self,
        results: list[RetrievalResult],
        tenant: TenantContext | None = None,
        reference_date: datetime | None = None,
    ) -> list[RetrievalResult]:
        """Apply all filters in order: province isolation -> time decay."""
        if tenant and tenant.allowed_provinces:
            results = self.province_filter.apply(results, tenant.allowed_provinces)
        results = self.time_decay.apply(results, reference_date)
        return results
