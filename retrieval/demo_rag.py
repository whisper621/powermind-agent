"""SQLite-backed lexical RAG for the offline V1 demo."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from infrastructure.paths import get_db_path


class DemoRAGRetriever:
    """Retrieve seeded policy/case documents without embeddings or APIs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else get_db_path("powermind_demo.db")

    def search(
        self,
        query: str,
        *,
        province: str = "山东",
        doc_type: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        if not self.db_path.exists():
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        sql = "SELECT * FROM knowledge_documents WHERE province IN (?, '全国')"
        params: list[object] = [province]
        if doc_type:
            sql += " AND doc_type=?"
            params.append(doc_type)
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
        conn.close()

        query_lower = query.lower()
        query_terms = set(re.findall(r"[a-z0-9%+-]{2,}", query_lower))
        scored = []
        for row in rows:
            keywords = [part.strip().lower() for part in row.get("keywords", "").split(",") if part.strip()]
            keyword_hits = sum(1 for term in keywords if term in query_lower)
            ascii_hits = sum(1 for term in query_terms if term in (row.get("title", "") + row.get("content", "")).lower())
            title_hits = sum(1 for term in keywords if term in row.get("title", "").lower())
            score = keyword_hits * 2 + title_hits + ascii_hits
            row["retrieval_score"] = float(score)
            row["retrieval_method"] = "sqlite_keyword_rag"
            scored.append(row)

        scored.sort(key=lambda item: (item["retrieval_score"], item["effective_date"]), reverse=True)
        matched = [item for item in scored if item["retrieval_score"] > 0]
        return (matched or scored)[:limit]

    @staticmethod
    def format_context(documents: list[dict]) -> str:
        blocks = []
        for doc in documents:
            blocks.append(
                f"[来源:{doc['doc_id']}] {doc['title']} | {doc['document_no']} | "
                f"{doc['source_name']}\n{doc['content']}"
            )
        return "\n\n".join(blocks)


def get_demo_rag(db_path: str | Path | None = None) -> DemoRAGRetriever:
    return DemoRAGRetriever(db_path)
