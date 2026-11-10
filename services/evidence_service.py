"""Evidence Service — collects real EvidenceRecords, binds conclusions to evidence_ids.

Never creates synthetic evidence.  Marks unsourced claims as unverified.
"""

from __future__ import annotations

from contracts.evidence import EvidenceRecord, EvidenceSourceType


class EvidenceService:
    """Collects, deduplicates, and validates evidence records.

    Agent output is NEVER an evidence source.  This service only processes
    records from real sources: data tables, tools, algorithms, policies, cases.
    """

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    def add(self, record: EvidenceRecord) -> None:
        """Add or update an evidence record (dedup by evidence_id)."""
        self._records[record.evidence_id] = record

    def add_many(self, records: list[EvidenceRecord]) -> None:
        for r in records:
            self.add(r)

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)

    def get_all(self) -> list[EvidenceRecord]:
        return list(self._records.values())

    def get_by_source_type(self, source_type: EvidenceSourceType) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if r.source_type == source_type]

    def get_unverified(self) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if r.verification_status == "unverified"]

    def check_references(self, evidence_ids: list[str]) -> dict:
        """Check which evidence_ids exist and which are missing."""
        missing = [eid for eid in evidence_ids if eid not in self._records]
        return {
            "found": len(evidence_ids) - len(missing),
            "missing": len(missing),
            "missing_ids": missing,
            "all_present": len(missing) == 0,
        }

    def bind_conclusion(
        self,
        claim: str,
        evidence_ids: list[str],
        agent_name: str,
    ) -> dict:
        """Bind an agent conclusion to evidence_ids WITHOUT making the agent a source."""
        ref_check = self.check_references(evidence_ids)
        return {
            "claim": claim,
            "agent": agent_name,
            "evidence_ids": evidence_ids,
            "reference_check": ref_check,
            "is_verified": ref_check["all_present"],
        }

    def to_dict_list(self) -> list[dict]:
        return [r.to_dict() for r in self._records.values()]

    @staticmethod
    def from_tool_output(
        tool_name: str,
        tool_version: str,
        tool_input: dict,
        tool_output: dict,
        province: str = "",
    ) -> EvidenceRecord:
        """Wrap a tool result as an EvidenceRecord."""
        import hashlib
        import json

        content_str = json.dumps(tool_output, sort_keys=True, default=str)
        fingerprint = hashlib.sha256(content_str.encode()).hexdigest()[:16]

        return EvidenceRecord(
            evidence_id=f"tool:{tool_name}:{fingerprint[:8]}",
            evidence_type="tool_result",
            source_type=EvidenceSourceType.TOOL_RESULT,
            source_name=tool_name,
            source_locator=f"tool://{tool_name}",
            tool_name=tool_name,
            tool_version=tool_version,
            province=province,
            query_scope=tool_input,
            raw_value=tool_output,
            content_fingerprint=fingerprint,
        )
