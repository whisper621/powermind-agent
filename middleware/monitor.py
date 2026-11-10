"""Pipeline monitoring for the unified LangGraph runtime.

Monitoring records facts only.  Fault-handling decisions are delegated to the
deterministic DegradationPolicy; this module does not execute hard-coded
fallback functions.

Monitored metrics:
  - Retrieval latency and result count
  - Generation latency and token count
  - Verification pass/fail rate
  - Error rates per stage

Health thresholds may trigger a policy evaluation, but they never select a
business result or bypass verification by themselves.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from middleware.degradation_policy import (
    DegradationPolicy,
    FailureContext,
    classify_failure,
)

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    stage_name: str
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    last_error: str = ""
    last_run_at: str = ""

    @property
    def avg_latency_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 1.0
        return self.success_count / self.call_count


@dataclass
class PipelineHealth:
    """Aggregated health status of the entire pipeline."""

    stages: dict[str, StageMetrics] = field(default_factory=dict)
    overall_status: str = "healthy"  # healthy | degraded | critical
    degraded_stages: list[str] = field(default_factory=list)
    last_check: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.overall_status == "healthy"


class PipelineMonitor:
    """Monitors pipeline stages and triggers degradation when thresholds exceeded.

    Usage:
        monitor = PipelineMonitor()

        @monitor.track("retrieval")
        def my_retrieve(query, province):
            ...

        health = monitor.check_health()
        if not health.is_healthy:
            # trigger fallback
    """

    # Thresholds for degradation
    LATENCY_WARN_MS = 5000  # 5 seconds
    LATENCY_CRITICAL_MS = 30000  # 30 seconds
    ERROR_RATE_WARN = 0.3  # 30%
    ERROR_RATE_CRITICAL = 0.7  # 70%

    def __init__(self):
        self._metrics: dict[str, StageMetrics] = defaultdict(
            lambda: StageMetrics(stage_name="")
        )

    def record(
        self,
        stage: str,
        success: bool,
        latency_ms: float,
        error: str = "",
    ) -> None:
        """Record a single stage execution."""
        m = self._metrics[stage]
        m.stage_name = stage
        m.call_count += 1
        if success:
            m.success_count += 1
        else:
            m.failure_count += 1
            m.last_error = error
        m.total_latency_ms += latency_ms
        m.last_run_at = datetime.now().isoformat()

        # Log warning if degraded
        if m.avg_latency_ms > self.LATENCY_WARN_MS:
            logger.warning(
                "PipelineMonitor: %s latency %.0fms exceeds warn threshold",
                stage,
                m.avg_latency_ms,
            )
        if m.success_rate < (1 - self.ERROR_RATE_WARN):
            logger.warning(
                "PipelineMonitor: %s error rate %.1f%% exceeds warn threshold",
                stage,
                (1 - m.success_rate) * 100,
            )

    def check_health(self) -> PipelineHealth:
        """Aggregate health check across all monitored stages."""
        health = PipelineHealth(last_check=datetime.now().isoformat())

        for stage_name, m in self._metrics.items():
            health.stages[stage_name] = m

            is_degraded = False

            # Latency check
            if m.avg_latency_ms > self.LATENCY_CRITICAL_MS:
                is_degraded = True
            # Error rate check
            if m.call_count >= 5 and m.success_rate < (1 - self.ERROR_RATE_CRITICAL):
                is_degraded = True

            if is_degraded:
                health.degraded_stages.append(stage_name)

        if len(health.degraded_stages) >= 3:
            health.overall_status = "critical"
        elif health.degraded_stages:
            health.overall_status = "degraded"

        return health

    def get_stage_metrics(self, stage: str) -> StageMetrics | None:
        return self._metrics.get(stage)

    def reset(self) -> None:
        self._metrics.clear()

    # ----- decorator for easy tracking ---------------------------------------

    def track(self, stage: str) -> Callable:
        """Decorator that records metrics for a function."""

        def decorator(fn: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    latency = (time.perf_counter() - t0) * 1000
                    self.record(stage, success=True, latency_ms=latency)
                    return result
                except Exception as exc:
                    latency = (time.perf_counter() - t0) * 1000
                    self.record(stage, success=False, latency_ms=latency, error=str(exc))
                    raise

            return wrapper

        return decorator


# ---------------------------------------------------------------------------
# Degradation triggers
# ---------------------------------------------------------------------------


class DegradationHandler:
    """Compatibility facade over the deterministic degradation policy.

    Existing callers may keep using ``should_degrade`` for health checks.  The
    actual action is selected from a FailureContext, not from a single fixed
    fallback function registered for a stage.
    """

    def __init__(
        self,
        monitor: PipelineMonitor | None = None,
        policy: DegradationPolicy | None = None,
    ):
        self._monitor = monitor or PipelineMonitor()
        self._policy = policy or DegradationPolicy()

    def should_degrade(self, stage: str) -> bool:
        """Check if a stage should be degraded."""
        metrics = self._monitor.get_stage_metrics(stage)
        if metrics is None:
            return False
        if metrics.call_count < 3:
            return False
        return metrics.success_rate < 0.5 or metrics.avg_latency_ms > 30000

    def decide(
        self,
        stage: str,
        error: BaseException | str,
        *,
        task_type: str = "general",
        critical: bool = True,
        attempt: int = 0,
        max_retries: int = 1,
        target_node: str = "",
        alternatives: tuple[str, ...] = (),
    ):
        """Return a structured decision; execution remains in LangGraph."""

        context = FailureContext(
            component=stage,
            failure_type=classify_failure(stage, error),
            error=str(error),
            task_type=task_type,
            critical=critical,
            attempt=attempt,
            max_retries=max_retries,
            target_node=target_node,
            alternatives=alternatives,
        )
        return self._policy.decide(context)
