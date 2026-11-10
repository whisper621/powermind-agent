"""Hybrid Retrieval Pipeline — BM25 + Dense + SQL + Graph + RRF fusion.

Orchestrates multiple retrieval backends and fuses results via Reciprocal Rank Fusion
with configurable per-channel weights.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from retrieval.base_retriever import BaseRetriever, RetrievalResult
from retrieval.graph_rag import GraphRAG
from retrieval.metadata_filter import MetadataFilter, TenantContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default channel weights for RRF (documents only — SQL excluded by default)
# SQL precise data is returned separately as StructuredDataResult and never
# participates in document RRF unless explicitly configured.

DEFAULT_DOCUMENT_CHANNEL_WEIGHTS = {
    "bm25": 0.30,
    "dense": 0.30,
    "graph": 0.20,
    "policy": 0.25,
    "expert": 0.20,
    "market": 0.20,
    "forecast": 0.08,
}

# Legacy alias kept for backward compat — DOES NOT include "sql"
DEFAULT_CHANNEL_WEIGHTS = DEFAULT_DOCUMENT_CHANNEL_WEIGHTS


# ---------------------------------------------------------------------------
# SQL Retriever
# ---------------------------------------------------------------------------


class SQLRetriever(BaseRetriever):
    """Structured data retriever using SQLite.

    Queries factor_snapshots, price_data, and policy_docs tables.
    Uses parameterised queries — no string interpolation for user input.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or self._find_db()

    @staticmethod
    def _find_db() -> str:
        candidates = [
            "data/sql/poweragent.db",
            "D:/GitHub_work/china-power-market-agent-main/power_market_agent/knowledge_base/knowledge_base.db",
            "data/knowledge_base.db",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return "data/sql/poweragent.db"

    def retrieve(
        self, query: str, top_k: int = 10, **filters
    ) -> list[RetrievalResult]:
        province = filters.get("province")
        if not Path(self._db_path).exists():
            logger.warning("SQLRetriever: db not found at %s", self._db_path)
            return []

        results: list[RetrievalResult] = []
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row

            # Detect available tables
            tables = set(r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall())

            # Query price_data (aggregated)
            if "price_data" in tables:
                try:
                    cur = conn.execute(
                        "SELECT province, date, AVG(price_rmb_mwh) as avg, "
                        "MAX(price_rmb_mwh) as mx, MIN(price_rmb_mwh) as mn "
                        "FROM price_data WHERE province=? "
                        "GROUP BY date ORDER BY date DESC LIMIT ?",
                        (province, top_k) if province else (top_k,),
                    ) if province else conn.execute(
                        "SELECT province, date, AVG(price_rmb_mwh) as avg "
                        "FROM price_data GROUP BY date ORDER BY date DESC LIMIT ?",
                        (top_k,),
                    )
                    for row in cur.fetchall():
                        results.append(RetrievalResult(
                            doc_id=f"sql:price:{row['date']}",
                            content=f"日期:{row['date']} | 均价:{row['avg']:.1f}元/MWh",
                            source="sql", score=0.8,
                            metadata={"table":"price_data","province":row["province"],"date":row["date"]},
                            citations=[f"price {row['date']}"],
                        ))
                except Exception:
                    pass

            # Query forecasts
            if "forecasts" in tables:
                try:
                    cur = conn.execute(
                        "SELECT * FROM forecasts WHERE province=? ORDER BY target_date DESC LIMIT ?",
                        (province, top_k) if province else (top_k,),
                    ) if province else conn.execute(
                        "SELECT * FROM forecasts ORDER BY target_date DESC LIMIT ?", (top_k,),
                    )
                    for row in cur.fetchall():
                        results.append(RetrievalResult(
                            doc_id=f"sql:forecast:{row['target_date']}",
                            content=f"预测:{row['target_date']} | 均价:{row['avg_price']:.1f}元/MWh | 置信度:{row['confidence']:.0%}",
                            source="sql", score=0.9,
                            metadata={"table":"forecasts","province":row["province"],"date":row["target_date"]},
                            citations=[f"forecast {row['target_date']}"],
                        ))
                except Exception:
                    pass

            # Query factor_snapshots
            if "factor_snapshots" in tables:
                try:
                    if province:
                        cur = conn.execute(
                            "SELECT * FROM factor_snapshots WHERE province=? ORDER BY date DESC LIMIT ?",
                            (province, top_k),
                        )
                    else:
                        cur = conn.execute(
                            "SELECT * FROM factor_snapshots ORDER BY date DESC LIMIT ?",
                            (top_k,),
                        )
                    for row in cur.fetchall():
                        results.append(RetrievalResult(
                            doc_id=f"sql:factor:{row['id']}",
                            content=self._row_to_text(row, "factor_snapshot"),
                            source="sql", score=0.7,
                            metadata={"table":"factor_snapshots","province":row["province"] if "province" in row.keys() else "","date":str(row["date"]) if "date" in row.keys() else ""},
                            citations=[f"factor_snapshot #{row['id']}"],
                        ))
                except Exception:
                    pass

            conn.close()
        except Exception as exc:
            logger.warning("SQLRetriever: query failed — %s", exc)

        return results[:top_k]

    @staticmethod
    def _row_to_text(row, table: str) -> str:
        return " | ".join(f"{k}: {v}" for k, v in dict(row).items() if v is not None)

    def health_check(self) -> bool:
        return Path(self._db_path).exists()


# ---------------------------------------------------------------------------
# BM25 Retriever (keyword-based, in-memory)
# ---------------------------------------------------------------------------


class BM25Retriever(BaseRetriever):
    """Simple BM25-style keyword retriever over ChromaDB documents.

    Uses the existing PowerRAGPipeline retrieval layer as backend, then
    re-ranks by keyword match density.
    """

    def __init__(self, rag_pipeline=None):
        self._rag = rag_pipeline

    def set_rag(self, rag):
        self._rag = rag

    def retrieve(
        self, query: str, top_k: int = 10, **filters
    ) -> list[RetrievalResult]:
        if self._rag is None:
            return []

        province = filters.get("province")
        try:
            policy_docs = self._rag.retrieval.retrieve_policy(query, province, top_k=top_k)
            expert_docs = self._rag.retrieval.retrieve_expert_insight(query, province, top_k=top_k)
            snapshot_docs = self._rag.retrieval.retrieve_market_snapshot(query, province, top_k=top_k)

            all_docs = policy_docs + expert_docs + snapshot_docs
            results = []
            for doc in all_docs:
                content = doc.get("content") or doc.get("text") or str(doc)
                score = self._bm25_score(query, content)
                results.append(
                    RetrievalResult(
                        doc_id=doc.get("id", ""),
                        content=content,
                        source="bm25",
                        score=round(score, 4),
                        metadata={
                            "province": doc.get("province", doc.get("metadata", {}).get("province", "")),
                            "date_published": doc.get("date", doc.get("metadata", {}).get("date", "")),
                            "data_type": doc.get("data_type", doc.get("metadata", {}).get("data_type", "")),
                        },
                        citations=[doc.get("source", doc.get("metadata", {}).get("source", ""))],
                    )
                )
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        except Exception as exc:
            logger.warning("BM25Retriever: retrieval failed — %s", exc)
            return []

    @staticmethod
    def _bm25_score(query: str, doc: str, k1: float = 1.2, b: float = 0.75) -> float:
        """Simplified BM25 scoring."""
        import math

        doc_terms = doc.lower().split()
        doc_len = len(doc_terms)
        avg_len = 300  # heuristic average
        score = 0.0
        for term in query.lower().split():
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            # Simplified IDF: assume term appears in ~10% of docs
            idf = math.log((1 + 1) / (0.1 + 0.5))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * numerator / denominator
        return score / max(1, len(query.split()))

    def health_check(self) -> bool:
        return self._rag is not None


# ---------------------------------------------------------------------------
# Dense Retriever (semantic vector)
# ---------------------------------------------------------------------------


class DenseRetriever(BaseRetriever):
    """Semantic vector retriever via ChromaDB (cosine similarity)."""

    def __init__(self, rag_pipeline=None):
        self._rag = rag_pipeline

    def set_rag(self, rag):
        self._rag = rag

    def retrieve(
        self, query: str, top_k: int = 10, **filters
    ) -> list[RetrievalResult]:
        if self._rag is None:
            return []

        province = filters.get("province")
        try:
            chroma = self._rag.retrieval._get_chroma()
            results = []
            for src, fetch_fn in [
                ("policy", lambda: self._rag.retrieval.retrieve_policy(query, province, top_k)),
                ("expert", lambda: self._rag.retrieval.retrieve_expert_insight(query, province, top_k)),
                ("market", lambda: self._rag.retrieval.retrieve_market_snapshot(query, province, top_k)),
                ("forecast", lambda: self._rag.retrieval.retrieve_forecast_archive(query, province, top_k)),
            ]:
                try:
                    docs = fetch_fn()
                except Exception:
                    continue
                for doc in docs:
                    content = doc.get("content") or doc.get("text") or str(doc)
                    results.append(
                        RetrievalResult(
                            doc_id=doc.get("id", ""),
                            content=content,
                            source="dense",
                            score=round(doc.get("score", doc.get("distance", 0.5)), 4),
                            metadata={
                                "province": doc.get("province", doc.get("metadata", {}).get("province", "")),
                                "date_published": doc.get("date", doc.get("metadata", {}).get("date", "")),
                                "data_type": doc.get("data_type", doc.get("metadata", {}).get("data_type", "")),
                            },
                            citations=[doc.get("source", doc.get("metadata", {}).get("source", ""))],
                        )
                    )
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        except Exception as exc:
            logger.warning("DenseRetriever: retrieval failed — %s", exc)
            return []

    def health_check(self) -> bool:
        return self._rag is not None


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------


def rrf_fusion(
    channel_results: dict[str, list[RetrievalResult]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion across multiple retrieval channels.

    Args:
        channel_results: {channel_name: [RetrievalResult]}
        weights: per-channel weight dict (defaults to DEFAULT_CHANNEL_WEIGHTS)
        k: RRF smoothing constant

    Returns:
        Sorted list of unique RetrievalResults by fused score.
    """
    w = weights or DEFAULT_CHANNEL_WEIGHTS

    # Build fused scores: doc_id -> accumulated score
    fused: dict[str, tuple[float, RetrievalResult]] = {}

    for channel, results in channel_results.items():
        channel_weight = w.get(channel, 0.05)
        for rank, result in enumerate(results):
            rrf_score = channel_weight / (rank + 1 + k)
            key = result.doc_id
            if key in fused:
                prev_score, prev_result = fused[key]
                fused[key] = (prev_score + rrf_score, prev_result)
            else:
                fused[key] = (rrf_score, result)

    # Sort by fused score
    sorted_results = sorted(fused.values(), key=lambda x: x[0], reverse=True)
    for score, result in sorted_results:
        result.score = round(score, 6)
    return [r for _, r in sorted_results]


# ---------------------------------------------------------------------------
# Hybrid Retrieval Pipeline
# ---------------------------------------------------------------------------


class HybridRetrievalPipeline:
    """Orchestrates BM25, Dense, SQL, Graph retrievers and fuses via RRF.

    Usage:
        pipeline = HybridRetrievalPipeline(rag=rag_pipeline, graph=graph_rag, sql=sql_retriever)
        results = pipeline.retrieve("山东现货电价", province="山东", channels=["bm25", "dense", "graph", "sql"])
    """

    def __init__(
        self,
        rag=None,
        graph: GraphRAG | None = None,
        sql: SQLRetriever | None = None,
        metadata_filter: MetadataFilter | None = None,
    ):
        self._rag = rag
        self.bm25 = BM25Retriever(rag)
        self.dense = DenseRetriever(rag)
        self.graph = graph or GraphRAG()
        self.sql = sql or SQLRetriever()
        self.metadata_filter = metadata_filter or MetadataFilter()

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        channels: list[str] | None = None,
        province: str | None = None,
        tenant: TenantContext | None = None,
    ) -> list[RetrievalResult]:
        """Run retrieval across specified channels and fuse.

        If no channels specified, uses all available.
        """
        channels = channels or list(DEFAULT_CHANNEL_WEIGHTS.keys())

        channel_results: dict[str, list[RetrievalResult]] = {}
        channel_errors: dict[str, str] = {}
        filters = {"province": province}

        # SQL operates as a SEPARATE structured data channel — not fused via document RRF
        sql_results: list[RetrievalResult] = []

        for ch in channels:
            try:
                if ch == "bm25":
                    channel_results[ch] = self.bm25.retrieve(query, top_k=top_k, **filters)
                elif ch == "dense":
                    channel_results[ch] = self.dense.retrieve(query, top_k=top_k, **filters)
                elif ch == "graph":
                    channel_results[ch] = self.graph.retrieve(query, top_k=top_k, **filters)
                elif ch == "sql":
                    # SQL returns structured data — NOT fused with document RRF
                    sql_results = self.sql.retrieve(query, top_k=top_k, **filters)
                elif ch in ("policy", "expert", "market", "price_history"):
                    channel_results[ch] = self._retrieve_from_rag(ch, query, top_k, province)
            except Exception as exc:
                logger.warning("HybridRetrieval: channel %s failed — %s", ch, exc)
                channel_errors[ch] = f"{type(exc).__name__}: {str(exc)[:200]}"
                if ch != "sql":
                    channel_results[ch] = []  # empty = no results (NOT same as channel failure)

        # RRF fusion — DOCUMENT CHANNELS ONLY (SQL excluded)
        doc_weights = {
            k: v for k, v in DEFAULT_DOCUMENT_CHANNEL_WEIGHTS.items()
            if k in channel_results or k in channels
        }
        fused = rrf_fusion(channel_results, weights=doc_weights)

        # SQL results appended AFTER RRF — their scores untouched by document ranking
        for sr in sql_results:
            sr.source = f"sql:{sr.source}"
        all_results = fused + sql_results

        # Apply metadata filtering
        all_results = self.metadata_filter.apply(all_results, tenant)

        return all_results[:top_k]

    def _retrieve_from_rag(
        self, channel: str, query: str, top_k: int, province: str | None
    ) -> list[RetrievalResult]:
        """Bridge to existing RAG retrieval layer."""
        if self._rag is None:
            return []

        method_map = {
            "policy": self._rag.retrieval.retrieve_policy,
            "expert": self._rag.retrieval.retrieve_expert_insight,
            "market": self._rag.retrieval.retrieve_market_snapshot,
            "price_history": self._rag.retrieval.retrieve_price_history,
            "forecast": self._rag.retrieval.retrieve_forecast_archive,
        }
        fetch = method_map.get(channel)
        if fetch is None:
            return []

        docs = fetch(query, province, top_k=top_k)
        results = []
        for doc in docs:
            content = doc.get("content") or doc.get("text") or str(doc)
            results.append(
                RetrievalResult(
                    doc_id=doc.get("id", ""),
                    content=content,
                    source=channel,
                    score=doc.get("score", doc.get("distance", 0.5)),
                    metadata={
                        "province": doc.get("province", doc.get("metadata", {}).get("province", "")),
                        "date_published": doc.get("date", doc.get("metadata", {}).get("date", "")),
                        "data_type": doc.get("data_type", doc.get("metadata", {}).get("data_type", "")),
                    },
                    citations=[doc.get("source", doc.get("metadata", {}).get("source", ""))],
                )
            )
        return results

    def health_check(self) -> dict[str, bool]:
        return {
            "bm25": self.bm25.health_check(),
            "dense": self.dense.health_check(),
            "graph": self.graph.health_check(),
            "sql": self.sql.health_check(),
        }
