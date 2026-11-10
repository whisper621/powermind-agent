#!/usr/bin/env python3
"""
工具调用审计日志 —— 记录每次 Tool 调用的输入/输出/耗时。
替代 LangSmith，纯本地实现，无需外部 API Key。

所有日志写入 data/audit_log.jsonl，每行一个 JSON 事件。
"""

from __future__ import annotations

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "data" / "audit_log.jsonl"


class AuditLogger:
    """工具调用审计记录器。"""

    def __init__(self):
        self._enabled = True
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_tool_start(self, tool_name: str, args: dict) -> str:
        """记录工具调用开始，返回 trace_id。"""
        trace_id = f"{int(time.time()*1000)}-{tool_name}"
        if not self._enabled:
            return trace_id

        event = {
            "trace_id": trace_id,
            "event": "tool_start",
            "tool": tool_name,
            "args": self._safe_serialize(args),
            "timestamp": datetime.now().isoformat(),
        }
        self._write(event)
        return trace_id

    def log_tool_end(self, trace_id: str, tool_name: str, duration_ms: float,
                     success: bool, output_summary: str, error: Optional[str] = None):
        """记录工具调用完成。"""
        if not self._enabled:
            return

        event = {
            "trace_id": trace_id,
            "event": "tool_end",
            "tool": tool_name,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "output_summary": output_summary[:200],
            "error": error[:200] if error else None,
            "timestamp": datetime.now().isoformat(),
        }
        self._write(event)

    def log_agent_step(self, step: str, detail: str):
        """记录 Agent 推理步骤。"""
        if not self._enabled:
            return

        event = {
            "event": "agent_step",
            "step": step,
            "detail": detail[:300],
            "timestamp": datetime.now().isoformat(),
        }
        self._write(event)

    def log_prediction(self, province: str, date: str, model: str,
                       avg_price: float, confidence: float):
        """记录预测事件。"""
        if not self._enabled:
            return

        event = {
            "event": "prediction",
            "province": province,
            "date": date,
            "model": model,
            "avg_price": avg_price,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }
        self._write(event)

    def _write(self, event: dict, business_critical: bool = False):
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as exc:
            if business_critical:
                raise RuntimeError(
                    f"审计写入失败(业务关键): {event.get('event', '?')}"
                ) from exc
            # Debug log failures are non-blocking — only log to stderr
            import sys
            print(f"[AUDIT_WARN] 调试审计日志写入失败: {exc}", file=sys.stderr)

    def health_check(self) -> bool:
        """Check if the audit store is writable."""
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            test_event = {
                "event": "health_check",
                "timestamp": datetime.now().isoformat(),
            }
            self._write(test_event, business_critical=False)
            return True
        except Exception:
            return False

    @staticmethod
    def _safe_serialize(obj) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)[:500]

    def get_recent_events(self, n: int = 50) -> list[dict]:
        """读取最近 N 条审计事件。"""
        if not LOG_PATH.exists():
            return []
        events = []
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return events[-n:]

    # ----- full-chain pipeline audit (PowerAgent OS) -------------------------

    def log_pipeline_stage(
        self,
        request_id: str,
        stage: str,
        status: str,
        detail: dict | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a pipeline stage event for full-chain audit."""
        if not self._enabled:
            return
        event = {
            "event": "pipeline_stage",
            "request_id": request_id,
            "stage": stage,
            "status": status,
            "detail": detail or {},
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now().isoformat(),
        }
        self._write(event)

    def log_verification(
        self,
        request_id: str,
        passed: bool,
        total_claims: int,
        verified_claims: int,
        issues: list[dict] | None = None,
    ) -> None:
        """Record a verification event."""
        if not self._enabled:
            return
        event = {
            "event": "verification",
            "request_id": request_id,
            "passed": passed,
            "total_claims": total_claims,
            "verified_claims": verified_claims,
            "issues": issues or [],
            "timestamp": datetime.now().isoformat(),
        }
        self._write(event)

    def log_degradation(
        self, stage: str, reason: str, fallback_used: str
    ) -> None:
        """Backward-compatible degradation audit entry."""
        if not self._enabled:
            return
        event = {
            "event": "degradation",
            "stage": stage,
            "reason": reason[:500],
            "fallback_used": fallback_used,
            "timestamp": datetime.now().isoformat(),
        }
        self._write(event)

    def log_fault_decision(self, event_data: dict) -> None:
        """Record a structured DegradationEvent selected by policy.

        BUSINESS-CRITICAL: failure to write this event blocks formal tasks.
        """
        if not self._enabled:
            return
        event = {
            "event": "fault_decision",
            **event_data,
            "timestamp": event_data.get("occurred_at", datetime.now().isoformat()),
        }
        self._write(event, business_critical=True)

    def log_tenant_access(
        self,
        tenant_id: str,
        province: str,
        action: str,
        allowed: bool,
    ) -> None:
        """Record a tenant data-access check for compliance auditing."""
        if not self._enabled:
            return
        event = {
            "event": "tenant_access",
            "tenant_id": tenant_id,
            "province": province,
            "action": action,
            "allowed": allowed,
            "timestamp": datetime.now().isoformat(),
        }
        self._write(event)

    def get_prediction_accuracy(self, province: str = None) -> dict:
        """统计预测事件数量和平均置信度。"""
        events = self.get_recent_events(200)
        preds = [e for e in events if e.get("event") == "prediction"]
        if province:
            preds = [e for e in preds if e.get("province") == province]
        if not preds:
            return {"total": 0}
        confidences = [e.get("confidence", 0) for e in preds]
        return {
            "total": len(preds),
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "provinces": list(set(e.get("province") for e in preds)),
        }


# ── 全局单例 ─────────────────────────────────────────────────

_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
