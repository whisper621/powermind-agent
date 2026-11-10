#!/usr/bin/env python3
"""
ChromaDB 检索器 —— 统一的 BM25 + 向量 RRF 混合检索。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHROMA_PATH = str(PROJECT_ROOT / "data" / "vectordb")
COLLECTION_NAME = "power_market_kb"

_retriever: Optional[object] = None


def get_retriever():
    """懒加载单例检索器。"""
    global _retriever
    if _retriever is not None:
        return _retriever

    try:
        from langchain_chroma import Chroma
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.runnables import RunnableLambda
        from langchain_core.documents import Document

        # 使用简单的本地 embedding（避免外部 API 依赖）
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=CHROMA_PATH,
        )

        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        db_data = vector_store.get()
        all_texts = db_data.get("documents", [])

        if not all_texts:
            _retriever = vector_retriever
            return _retriever

        bm25_retriever = BM25Retriever.from_texts(all_texts)
        bm25_retriever.k = 5

        def custom_ensemble_retriever(query: str) -> list[Document]:
            vec_docs = vector_retriever.invoke(query)
            bm25_docs = bm25_retriever.invoke(query)

            rrf_scores: dict = {}
            c = 60

            for rank, doc in enumerate(vec_docs):
                key = doc.page_content
                if key not in rrf_scores:
                    rrf_scores[key] = {"score": 0.0, "doc": doc}
                rrf_scores[key]["score"] += 0.5 / (rank + 1 + c)

            for rank, doc in enumerate(bm25_docs):
                key = doc.page_content
                if key not in rrf_scores:
                    rrf_scores[key] = {"score": 0.0, "doc": doc}
                rrf_scores[key]["score"] += 0.5 / (rank + 1 + c)

            sorted_items = sorted(
                rrf_scores.values(), key=lambda x: x["score"], reverse=True
            )
            return [item["doc"] for item in sorted_items[:5]]

        _retriever = RunnableLambda(custom_ensemble_retriever)
        return _retriever
    except Exception as e:
        print(f"[chroma_retriever] 初始化失败: {e}")
        return None
