#!/usr/bin/env python3
"""
PowerMind RAG 评估脚本 — 评估 HybridRAG 检索质量。

指标:
  - Hit Rate@k (HR@k): 正确答案是否在 top-k 中
  - Mean Reciprocal Rank (MRR): 第一个正确答案排名的倒数均值
  - Recall@k: top-k 中正确答案占比

原理:
  从 ChromaDB 知识库中采样已存储的文档片段作为查询，
  用这些片段测试检索系统能否找回到原文档。

参考: PowerTrade/scripts/evaluate_rag.py (Hit Rate@3=96.25%, MRR=0.877)
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_knowledge_base_samples(
    chroma_path: str | None = None,
    collection_name: str = "power_market_kb",
    sample_size: int = 50,
    seed: int = 42,
) -> list[dict]:
    """从 ChromaDB 知识库中采样文档片段作为测试查询"""
    import chromadb
    from chromadb.config import Settings

    path = chroma_path or str(PROJECT_ROOT / "data" / "vectordb")
    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        print(f"[评估] 知识库为空 — collection '{collection_name}' 不存在")
        return []

    all_data = collection.get()
    total_docs = len(all_data.get("ids", []))

    if total_docs == 0:
        print("[评估] 知识库为空 — 无文档")
        return []

    print(f"[评估] 知识库共 {total_docs} 个 chunks")

    # 采样
    random.seed(seed)
    indices = random.sample(range(total_docs), min(sample_size, total_docs))

    samples = []
    for idx in indices:
        doc_id = all_data["ids"][idx]
        content = all_data["documents"][idx] if all_data.get("documents") else ""
        metadata = all_data["metadatas"][idx] if all_data.get("metadatas") else {}

        # 截取前100字符作为查询
        query_text = content[:100].strip() if content else ""

        samples.append({
            "doc_id": doc_id,
            "query": query_text,
            "full_content": content[:500],
            "metadata": metadata,
        })

    print(f"[评估] 采样 {len(samples)} 个查询 (seed={seed})")
    return samples


def _extract_query_from_chunk(chunk: str, max_len: int = 80) -> str:
    """从 chunk 中提取代表性查询 (首句)"""
    # 按中文或英文标点分割
    for sep in ["。", "！", "？", ".", "!", "?", "\n"]:
        if sep in chunk[:max_len]:
            return chunk[:chunk.index(sep)].strip()
    return chunk[:max_len].strip()


def evaluate_retrieval(
    retriever_func,
    samples: list[dict],
    top_k: int = 5,
) -> dict:
    """
    评估检索质量。

    Args:
        retriever_func: 检索函数，签名为 (query: str) -> list[dict]，每个dict需含 "doc_id" 字段
        samples: 采样数据
        top_k: Top-K

    Returns:
        {"hit_rate": float, "mrr": float, "recall_at_k": float, ...}
    """
    hit_count = 0
    reciprocal_ranks = []
    recalls = []
    latencies = []

    for i, sample in enumerate(samples):
        query = sample["query"]
        target_id = sample["doc_id"]

        if not query:
            continue

        t0 = time.perf_counter()
        try:
            results = retriever_func(query)[:top_k]
        except Exception as e:
            print(f"  [{i+1}] 检索失败: {e}")
            continue
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        result_ids = [r.get("doc_id", r.get("id", "")) for r in results]

        # Hit Rate: 正确答案是否在 top-k 中
        hit = target_id in result_ids
        if hit:
            hit_count += 1

        # MRR: 第一个正确答案排名的倒数
        try:
            rank = result_ids.index(target_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

        # Recall@k: 命中的正确数 / 1 (这里每个查询只有一个正确答案)
        recalls.append(1.0 if hit else 0.0)

        if (i + 1) % 20 == 0:
            print(f"  已评估 {i+1}/{len(samples)} 条...")

    n = len(reciprocal_ranks)
    return {
        "total_queries": n,
        "hit_rate_at_k": hit_count / n if n > 0 else 0,
        "mrr": sum(reciprocal_ranks) / n if n > 0 else 0,
        "recall_at_k": sum(recalls) / n if n > 0 else 0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "top_k": top_k,
    }


# ═══════════════════════════════════════════════════════════════════
# ChromaDB 原生检索器 (不依赖 RAG Pipeline)
# ═══════════════════════════════════════════════════════════════════

def build_chroma_retriever(
    chroma_path: str | None = None,
    collection_name: str = "power_market_kb",
    top_k: int = 5,
):
    """构建 ChromaDB 原始向量检索器 (用于基准测试)"""
    import chromadb
    from chromadb.config import Settings

    path = chroma_path or str(PROJECT_ROOT / "data" / "vectordb")
    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(collection_name)

    def retrieve(query: str) -> list[dict]:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            items.append({
                "doc_id": doc_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": round(1 - (dists[i] if i < len(dists) else 1), 4),
            })
        return items

    return retrieve


# ═══════════════════════════════════════════════════════════════════
# 完整评估流程
# ═══════════════════════════════════════════════════════════════════

def run_full_evaluation(
    sample_size: int = 50,
    top_k: int = 5,
    seed: int = 42,
) -> dict:
    """执行完整 RAG 评估"""
    print("=" * 60)
    print("PowerMind RAG 检索质量评估")
    print("=" * 60)

    # 1. 采样
    samples = load_knowledge_base_samples(sample_size=sample_size, seed=seed)
    if not samples:
        return {"error": "知识库为空，无法评估"}

    # 2. 构建检索器
    retriever = build_chroma_retriever(top_k=top_k)

    # 3. 评估
    print(f"\n[评估] 开始评估 (top_k={top_k})...")
    metrics = evaluate_retrieval(retriever, samples, top_k=top_k)

    # 4. 输出
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"  Hit Rate@{top_k}:  {metrics['hit_rate_at_k']:.2%}")
    print(f"  MRR:              {metrics['mrr']:.4f}")
    print(f"  Recall@{top_k}:   {metrics['recall_at_k']:.2%}")
    print(f"  平均延迟:         {metrics['avg_latency_ms']:.1f} ms")
    print(f"  P95延迟:          {metrics['p95_latency_ms']:.1f} ms")
    print(f"  评估查询数:       {metrics['total_queries']}")

    # 5. 质量评级
    hr = metrics["hit_rate_at_k"]
    if hr >= 0.95:
        grade = "A+ (优秀)"
    elif hr >= 0.85:
        grade = "A (良好)"
    elif hr >= 0.70:
        grade = "B (合格)"
    elif hr >= 0.50:
        grade = "C (需改进)"
    else:
        grade = "D (不可用)"

    print(f"\n  综合评级: {grade}")
    print(f"\n  参考基准: PowerTrade HR@3=96.25%, MRR=0.877")

    return metrics


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PowerMind RAG 检索质量评估")
    parser.add_argument("--sample-size", type=int, default=50, help="采样数量")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")

    args = parser.parse_args()

    results = run_full_evaluation(
        sample_size=args.sample_size,
        top_k=args.top_k,
        seed=args.seed,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {args.output}")
