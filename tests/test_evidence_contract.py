"""Evidence contract tests — Agents are NOT evidence sources.

Agent output can reference evidence_ids but is not itself evidence.
Hardcoded confidence scores (0.85, 0.9) are prohibited.
"""

from __future__ import annotations

import unittest

from contracts.evidence import EvidenceRecord, EvidenceSourceType


class EvidenceContractTests(unittest.TestCase):
    def test_evidence_record_from_tool_has_source(self) -> None:
        """Tool results produce EvidenceRecords with real source info."""
        record = EvidenceRecord(
            evidence_id="tool:predict:abc123",
            evidence_type="price_forecast",
            source_type=EvidenceSourceType.TOOL_RESULT,
            source_name="predict_electricity_price",
            source_locator="tool://predict_electricity_price",
            tool_name="predict_electricity_price",
            tool_version="2.1.0",
            province="山东",
        )
        self.assertEqual(record.source_type, EvidenceSourceType.TOOL_RESULT)
        self.assertNotEqual(record.source_type, "AGENT")  # Agent is not a source type
        self.assertEqual(record.tool_name, "predict_electricity_price")
        self.assertEqual(record.tool_version, "2.1.0")

    def test_evidence_record_has_verification_status(self) -> None:
        """Every record starts as unverified by default."""
        record = EvidenceRecord(
            evidence_id="test:1",
            evidence_type="test",
            source_type=EvidenceSourceType.STRUCTURED_DATA,
            source_name="test_table",
            source_locator="test.db:test_table",
        )
        self.assertEqual(record.verification_status, "unverified")

    def test_evidence_record_rejects_agent_source_type(self) -> None:
        """EvidenceSourceType enum does not include AGENT."""
        valid_types = set(EvidenceSourceType.__members__.values())
        self.assertNotIn("AGENT", {t.name for t in valid_types})
        self.assertNotIn("agent", {t.value for t in valid_types})

    def test_evidence_record_to_dict(self) -> None:
        """to_dict() serialises correctly."""
        record = EvidenceRecord(
            evidence_id="ev:001",
            evidence_type="policy_check",
            source_type=EvidenceSourceType.POLICY_DOCUMENT,
            source_name="山东省电力交易规则",
            source_locator="docs/山东/2024-交易规则.pdf:Section3.2",
            source_version="2024-v3",
            province="山东",
            data_time="2024-01-01",
        )
        d = record.to_dict()
        self.assertEqual(d["evidence_id"], "ev:001")
        self.assertEqual(d["source_type"], "policy_document")
        self.assertEqual(d["source_locator"], "docs/山东/2024-交易规则.pdf:Section3.2")

    def test_agent_name_is_not_valid_source_type(self) -> None:
        """EvidenceSourceType enum values don't include agent identifiers.

        EXPERT_CASE is a valid type (reviewed historical cases) — it is NOT the same
        as "EXPERT Agent" being a source.  Agent names like QUANT_AGENT, COMPLIANCE_AGENT,
        STRATEGY_AGENT are NOT valid source types.
        """
        # These would be invalid (agent names as source types):
        invalid_names = {"QUANT_AGENT", "COMPLIANCE_AGENT", "STRATEGY_AGENT",
                         "AGENT_OUTPUT", "LLM_OUTPUT", "MAS_AGENT"}
        for source_type in EvidenceSourceType:
            self.assertNotIn(source_type.name, invalid_names)

    def test_no_hardcoded_confidence_in_evidence_record(self) -> None:
        """EvidenceRecord has no default confidence > 0; verification starts at unverified."""
        record = EvidenceRecord(
            evidence_id="ev:002",
            evidence_type="test",
            source_type=EvidenceSourceType.TOOL_RESULT,
            source_name="test_tool",
            source_locator="tool://test",
        )
        # Confidence is NOT a field on EvidenceRecord — it's verification_status
        self.assertFalse(hasattr(record, "confidence"))


if __name__ == "__main__":
    unittest.main()
