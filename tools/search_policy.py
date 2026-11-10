#!/usr/bin/env python3
"""Offline policy RAG tool backed by the V1 SQLite demo database."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.tools import tool


@tool
def search_policy(query: str, province: str = "全国") -> str:
    """
    从本地SQLite演示知识库检索政策、规则和偏差考核资料。
    使用 Retrieval → Context Augmentation → 模板化回答，全程不调用API或算法。

    当用户询问"XX省的现货交易规则"、"偏差考核标准"、"分时电价政策"、"新能源入市规则"等
    涉及电力市场规则文本的问题时，必须调用此工具。

    Args:
        query: 检索关键词，如 "山东 现货交易规则"、"偏差考核 ±5%"
        province: 限定省份范围，默认 "全国"

    Returns:
        带来源编号的演示回答。
    """
    from retrieval.demo_rag import get_demo_rag
    from scripts.seed_demo_v1 import ensure_demo_database

    ensure_demo_database()
    docs = get_demo_rag().search(query, province=province if province != "全国" else "山东", doc_type="policy", limit=5)
    if not docs:
        return f"【政策RAG】未命中演示资料：{query}（{province}）"

    lines = [
        "## 政策RAG检索结果（纯数据库演示）",
        "",
        f"- 查询：{query}",
        f"- 省份：{province}",
        f"- 召回方式：SQLite关键词RAG",
        "",
    ]
    for doc in docs:
        lines.extend([
            f"### [{doc['doc_id']}] {doc['title']}",
            f"- 文号：{doc['document_no']}｜来源：{doc['source_name']}｜相关分：{doc['retrieval_score']:.0f}",
            f"- 要点：{doc['content']}",
            "",
        ])
    lines.append("> ⚠️ 以上均为数据库演示假资料，只用于测试流程，不得用于真实交易或合规判断。")
    return "\n".join(lines)


# ── CLI 测试 ─────────────────────────────────────────────────

if __name__ == "__main__":
    result = search_policy.invoke({
        "query": "现货交易规则 偏差考核",
        "province": "山东",
    })
    print(result)
