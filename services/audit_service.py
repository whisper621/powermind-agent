"""Audit Service — interface + Demo JSONL adapter.

Formal business audit failures MUST block the task.
Debug log failures are non-blocking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "audit_log.jsonl"

logger = logging.getLogger(__name__)


@runtime_checkable
class AuditSink(Protocol):
    """Audit storage interface.  Replace JSONL with PostgreSQL later."""

    def write(self, event: "AuditEvent") -> None: ...

    def health_check(self) -> bool: ...


class AuditUnavailableError(RuntimeError):
    """Raised when a business-critical audit write fails."""
    pass


class AuditEvent:
    """Structured audit event — supersedes raw dict writes."""

    def __init__(
        self,
        event_type: str,
        component: str,
        detail: str = "",
        request_id: str = "",
        success: bool = True,
        metadata: dict | None = None,
        is_business_critical: bool = False,
    ) -> None:
        self.event_type = event_type
        self.component = component
        self.detail = detail
        self.request_id = request_id
        self.success = success
        self.metadata = metadata or {}
        self.is_business_critical = is_business_critical
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "event": self.event_type,
            "component": self.component,
            "detail": self.detail[:500],
            "request_id": self.request_id,
            "success": self.success,
            "metadata": self.metadata,
            "business_critical": self.is_business_critical,
            "timestamp": self.timestamp,
        }


class DemoJSONLAuditSink:
    """JSONL-file AuditSink for Demo.  Replace with PostgreSQL later."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path or DEFAULT_LOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: AuditEvent) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            if event.is_business_critical:
                raise AuditUnavailableError(
                    f"审计写入失败: {event.event_type} (request={event.request_id})"
                ) from exc
            else:
                logger.warning("Debug audit write failed (non-blocking): %s", exc)

    def health_check(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write("")
            return True
        except Exception:
            return False

    def get_recent(self, n: int = 50) -> list[dict]:
        if not self._path.exists():
            return []
        events = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        events.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return events[-n:]


def create_demo_audit_sink() -> DemoJSONLAuditSink:
    return DemoJSONLAuditSink()


class DemoSQLiteAuditSink:
    """Store demo execution traces beside all other fake data in SQLite."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        from infrastructure.paths import get_db_path

        self._path = Path(db_path) if db_path else get_db_path("powermind_demo.db")

    def write(self, event: AuditEvent) -> None:
        import sqlite3

        try:
            conn = sqlite3.connect(self._path)
            conn.execute(
                "INSERT INTO audit_events(timestamp,request_id,event_type,component,detail,success,business_critical,metadata_json) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    event.timestamp,
                    event.request_id,
                    event.event_type,
                    event.component,
                    event.detail[:500],
                    int(event.success),
                    int(event.is_business_critical),
                    json.dumps(event.metadata, ensure_ascii=False, default=str),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            if event.is_business_critical:
                raise AuditUnavailableError(f"SQLite审计写入失败: {event.event_type}") from exc
            logger.warning("SQLite demo audit write failed: %s", exc)

    def health_check(self) -> bool:
        return self._path.exists()

    def get_recent(self, n: int = 50) -> list[dict]:
        import sqlite3

        if not self._path.exists():
            return []
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        conn.close()
        return [dict(row) for row in reversed(rows)]
