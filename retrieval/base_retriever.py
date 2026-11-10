"""Abstract retriever interface and common types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalResult:
    """Unified retrieval result across all backends."""

    doc_id: str
    content: str
    source: str  # "policy" | "expert" | "market" | "graph" | "sql" | "bm25" | "dense"
    score: float
    metadata: dict = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)

    def to_context_block(self) -> str:
        cite = ", ".join(self.citations) if self.citations else self.source
        return f"[{self.source} | {cite}] {self.content}"


class BaseRetriever(ABC):
    """Abstract base for all retrievers."""

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 10, **filters
    ) -> list[RetrievalResult]:
        """Retrieve documents matching the query."""

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the retriever backend is available."""
