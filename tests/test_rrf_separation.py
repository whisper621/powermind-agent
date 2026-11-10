"""RRF separation tests — SQL does NOT participate in document RRF by default.

Document channels (BM25, Dense, Graph) participate in RRF.
SQL structured results are returned separately.
"""

from __future__ import annotations

import unittest

from retrieval.hybrid_retrieval import DEFAULT_DOCUMENT_CHANNEL_WEIGHTS, DEFAULT_CHANNEL_WEIGHTS


class RRFChannelSeparationTests(unittest.TestCase):
    def test_sql_not_in_default_document_weights(self) -> None:
        """SQL channel is excluded from document RRF weights by default."""
        self.assertNotIn("sql", DEFAULT_DOCUMENT_CHANNEL_WEIGHTS)

    def test_default_weights_alias_also_excludes_sql(self) -> None:
        """Legacy DEFAULT_CHANNEL_WEIGHTS also excludes SQL."""
        self.assertNotIn("sql", DEFAULT_CHANNEL_WEIGHTS)

    def test_document_channels_include_bm25_dense_graph(self) -> None:
        """Document RRF channels include the main document retrievers."""
        self.assertIn("bm25", DEFAULT_DOCUMENT_CHANNEL_WEIGHTS)
        self.assertIn("dense", DEFAULT_DOCUMENT_CHANNEL_WEIGHTS)
        self.assertIn("graph", DEFAULT_DOCUMENT_CHANNEL_WEIGHTS)

    def test_document_channels_weight_sum_is_reasonable(self) -> None:
        """Weights are positive and non-zero."""
        for ch, w in DEFAULT_DOCUMENT_CHANNEL_WEIGHTS.items():
            self.assertGreater(w, 0, f"Channel {ch} weight must be > 0")


class RRFChannelBehaviorTests(unittest.TestCase):
    """Verify that SQL results are not fused through document RRF."""

    def test_rrf_fusion_excludes_sql_by_default(self) -> None:
        """rrf_fusion called with only document channels excludes SQL."""
        from retrieval.hybrid_retrieval import rrf_fusion, RetrievalResult

        doc_results = {
            "bm25": [
                RetrievalResult(doc_id="d1", content="政策A", source="bm25", score=0.9),
            ],
            "dense": [
                RetrievalResult(doc_id="d1", content="政策A", source="dense", score=0.85),
            ],
        }
        # Without sql channel
        fused = rrf_fusion(doc_results)
        self.assertTrue(len(fused) > 0)
        for r in fused:
            self.assertNotIn("sql", r.source)


if __name__ == "__main__":
    unittest.main()
