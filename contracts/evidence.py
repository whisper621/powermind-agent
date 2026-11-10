"""Evidence data model.

Agents are *analysers*, not original evidence sources.
EvidenceRecords trace back to data tables, tools, algorithms, policy documents,
or user-provided materials — never to agent names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class EvidenceSourceType(str, Enum):
    STRUCTURED_DATA = "structured_data"       # SQL tables, data warehouses
    TOOL_RESULT = "tool_result"               # Deterministic tool output
    ALGORITHM_OUTPUT = "algorithm_output"     # Prediction / optimisation output
    POLICY_DOCUMENT = "policy_document"       # Original regulation text
    EXPERT_CASE = "expert_case"               # Reviewed historical case
    EXTERNAL_API = "external_api"             # Third-party API response
    USER_PROVIDED = "user_provided"           # User-confirmed material
    # Agent output is NEVER a valid EvidenceSourceType.


@dataclass
class EvidenceRecord:
    """A single piece of verifiable evidence.

    Every field that helps locate the original source is required.
    Agent-produced analysis may reference evidence_ids but is not itself evidence.
    """

    evidence_id: str
    evidence_type: str
    source_type: EvidenceSourceType
    source_name: str
    source_locator: str          # file path, table name, URL, clause reference
    source_version: str = ""
    province: str = ""
    applicable_scope: dict = field(default_factory=dict)
    data_time: str = ""          # When the data was observed (ISO format)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_name: str = ""
    tool_version: str = ""
    algorithm_name: str = ""
    algorithm_version: str = ""
    query_scope: dict = field(default_factory=dict)
    raw_value: object = None
    unit: str = ""
    content_fingerprint: str = ""
    verification_status: str = "unverified"  # unverified | verified | disputed

    def to_dict(self) -> dict:
        import json
        result = {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "source_locator": self.source_locator,
            "source_version": self.source_version,
            "province": self.province,
            "applicable_scope": self.applicable_scope,
            "data_time": self.data_time,
            "generated_at": self.generated_at,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_version": self.algorithm_version,
            "query_scope": self.query_scope,
            "unit": self.unit,
            "content_fingerprint": self.content_fingerprint,
            "verification_status": self.verification_status,
        }
        if self.raw_value is not None:
            try:
                result["raw_value"] = str(self.raw_value)
            except Exception:
                result["raw_value"] = "<unserializable>"
        return result
