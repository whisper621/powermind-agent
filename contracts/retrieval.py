"""Retrieval data model — channel status, RRF separation, structured bundles.

SQL precise data does NOT participate in document RRF by default.
Channel failures are distinguished from "no results found".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChannelState(str, Enum):
    SUCCESS = "success"
    NO_RESULT = "no_result"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class ChannelStatus:
    channel_name: str
    state: ChannelState
    error_type: str = ""
    error_summary: str = ""
    has_registered_alternative: bool = False
    remaining_channels_sufficient: bool = True


@dataclass
class RetrievalResult:
    """A single retrieval result with stable identifier."""
    doc_id: str
    content: str
    source: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    channel: str = ""
    rank_in_channel: int = 0
    fused_score: float = 0.0
    unified_id: str = ""          # stable dedup key
    source_file_id: str = ""
    section_or_clause_id: str = ""
    document_version: str = ""
    content_fingerprint: str = ""


@dataclass
class StructuredDataResult:
    """SQL-exact data — never fused through document RRF."""
    table_name: str
    data: list[dict]
    unit: str = ""
    data_time: str = ""
    query_conditions: dict = field(default_factory=dict)
    data_version: str = ""
    province: str = ""
    scope: dict = field(default_factory=dict)
    row_count: int = 0


@dataclass
class RetrievalBundle:
    """Complete retrieval output — documents + structured data + channel status.

    Decision authority belongs to DegradationPolicy, not to this bundle.
    """
    document_results: list[RetrievalResult] = field(default_factory=list)
    structured_results: list[StructuredDataResult] = field(default_factory=list)
    channel_status: dict[str, ChannelStatus] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rrf_explanation: list[dict] = field(default_factory=list)
