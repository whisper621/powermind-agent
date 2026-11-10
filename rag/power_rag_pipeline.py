#!/usr/bin/env python3
"""
PowerMind RAG Pipeline — 完整的 Retrieval → Augmentation → Generation 三阶段流水线。

修正原架构缺陷:
  ✅ 新增 Augmentation 层: 查询改写 + 多源融合 + 上下文压缩 + 相关性重排
  ✅ 新增 Generation 层: LLM 调用 + Citation 来源标注 + 事实校验
  ✅ 新增 4 层知识检索策略: 规则 → 市场 → 知识 → 经验
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════
# 1. Retrieval Layer (检索层)
# ═══════════════════════════════════════════════════════════════════


class RetrievalLayer:
    """多路召回: ChromaDB 语义 + SQLite 结构化 + RRF 融合。"""

    def __init__(self) -> None:
        self._chroma_collection = None

    def _get_chroma(self):
        if self._chroma_collection:
            return self._chroma_collection
        import chromadb
        from chromadb.config import Settings

        chroma_path = str(PROJECT_ROOT / "data" / "vectordb")
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self._chroma_collection = client.get_collection("power_market_kb")
        except Exception:
            self._chroma_collection = client.create_collection(
                name="power_market_kb",
                metadata={"hnsw:space": "cosine"},
            )
        return self._chroma_collection

    def retrieve_policy(self, query: str, province: str, top_k: int = 5) -> list[dict]:
        """检索政策规则（★ 双路：元数据省域过滤 + 全国通查）。"""
        collection = self._get_chroma()
        try:
            # ★ 第一路: 用 ChromaDB where 精准过滤该省 + 全国性政策
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where={
                    "$or": [
                        {"province": province},
                        {"province": "全国"},
                    ]
                },
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # 回退：某些 embedding 函数或版本不支持 where 过滤
            try:
                results = collection.query(
                    query_texts=[f"{province} {query}"],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return self._fallback_sqlite_policy(query, province, top_k)

        items = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            items.append({
                "id": doc_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": round(1 - (dists[i] if i < len(dists) else 1), 4),
                "source": "chromadb",
            })
        return items

    def retrieve_expert_insight(self, query: str, province: str, top_k: int = 5) -> list[dict]:
        """
        检索专家实战心得（★ 按省份+数据类型的双层过滤）。

        与 retrieve_policy 不同，此方法专门检索 data_type='expert_insight' 的文档，
        支持按省份精准过滤，确保 "山东老交易员的心得" 不会和 "山西规则" 混淆。
        """
        collection = self._get_chroma()
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where={
                    "$and": [
                        {"data_type": "expert_insight"},
                        {"$or": [
                            {"province": province},
                            {"province": "全国"},
                        ]},
                    ]
                },
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # 回退：无 where 过滤
            try:
                results = collection.query(
                    query_texts=[f"{province} {query} 交易心得 复盘"],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return []

        items = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            items.append({
                "id": doc_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": round(1 - (dists[i] if i < len(dists) else 1), 4),
                "source": "chromadb:expert_insight",
            })
        return items

    def retrieve_market_snapshot(self, query: str, province: str, top_k: int = 5) -> list[dict]:
        """
        检索市场行情快照 (★ 新增: 从 ChromaDB 检索价格/负荷/天气等结构化数据摘要)。

        覆盖 data_type: market_snapshot (日级) + market_monthly_summary (月级)
        当用户问 "去年7月电价多少" "光伏出力最高的月份" 等历史数据问题时，
        不再只能靠 LLM 记忆，而是精确检索到真实的市场快照数据。
        """
        collection = self._get_chroma()
        try:
            results = collection.query(
                query_texts=[f"{province} {query}"],
                n_results=top_k,
                where={
                    "$or": [
                        {"data_type": "market_snapshot"},
                        {"data_type": "market_monthly_summary"},
                    ],
                },
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # 回退: 无 where 过滤
            try:
                results = collection.query(
                    query_texts=[f"{province} {query} 电价 负荷 天气"],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return []

        items = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            items.append({
                "id": doc_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": round(1 - (dists[i] if i < len(dists) else 1), 4),
                "source": "chromadb:market_snapshot",
            })
        return items

    def retrieve_forecast_archive(self, query: str, province: str, top_k: int = 5) -> list[dict]:
        """
        检索历史预测归档 (新增: 从 ChromaDB 检索 LSTM 24点电价预测摘要)。

        覆盖 data_type: forecast_archive
        当用户问 "昨天的预测电价是多少" "这周预测均价趋势" 等历史预测数据问题时，
        直接从归档库检索，不再依赖 LLM 记忆。

        每份归档文档包含: 24点预测、P10/P50/P90 信心区间、峰谷均价、市场特征标签。
        """
        collection = self._get_chroma()
        try:
            results = collection.query(
                query_texts=[f"{province} {query}"],
                n_results=top_k,
                where={
                    "$and": [
                        {"data_type": "forecast_archive"},
                        {"$or": [
                            {"province": province},
                            {"province": "全国"},
                        ]},
                    ]
                },
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            try:
                results = collection.query(
                    query_texts=[f"{province} {query} 电价预测 LSTM"],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return []

        items = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            items.append({
                "id": doc_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": round(1 - (dists[i] if i < len(dists) else 1), 4),
                "source": "chromadb:forecast_archive",
            })
        return items

    def retrieve_price_history(self, province: str, target_date: str,
                                lookback_days: int = 30, top_k: int = 3) -> list[dict]:
        """
        检索历史电价模式（★ 从 ChromaDB 找相似价格模式的日期）。

        用途: 当 LSTM 预测某个日期电价异常时，从 RAG 中找到历史上类似价格模式的日期，
        并调取该日期的复盘记录，辅助 STRATEGY Agent 给出更精准的建议。

        Args:
            province: 省份
            target_date: 目标日期
            lookback_days: 回溯天数
            top_k: 返回最相似的 K 个历史日期
        """
        collection = self._get_chroma()
        try:
            results = collection.query(
                query_texts=[f"{province} 电价 {target_date} 日前出清"],
                n_results=top_k,
                where={
                    "data_type": "price_history",
                    "province": province,
                },
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        items = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            items.append({
                "id": doc_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": round(1 - (dists[i] if i < len(dists) else 1), 4),
                "source": "chromadb:price_history",
            })
        return items

    def retrieve_market_data(self, province: str, date: str | None = None) -> list[dict]:
        """从 SQLite 检索结构化市场数据。"""
        import sqlite3

        db_path = (
            Path(r"D:\GitHub_work\china-power-market-agent-main\power_market_agent")
            / "knowledge_base"
            / "knowledge_base.db"
        )
        if not db_path.exists():
            return []

        items = []
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # 省份画像
            prov = conn.execute(
                "SELECT * FROM provinces WHERE name=?", (province,)
            ).fetchone()
            if prov:
                items.append({
                    "content": f"省份: {prov['name']} | 区域: {prov['region']} | 能源: {prov['energy_mix']}",
                    "metadata": {"type": "province_profile", "province": province},
                    "score": 1.0,
                    "source": "sqlite:provinces",
                })

            # 因子快照
            if date:
                factors = conn.execute(
                    "SELECT * FROM factor_snapshots WHERE province=? AND date=?",
                    (province, date),
                ).fetchone()
                if factors:
                    items.append({
                        "content": f"因子快照: 方向={factors['overall_direction']} 风险={factors['overall_risk']}",
                        "metadata": {"type": "factor_snapshot", "date": date},
                        "score": 0.9,
                        "source": "sqlite:factor_snapshots",
                    })

            # 政策文档
            policies = conn.execute(
                "SELECT * FROM policy_docs WHERE province=? OR province='全国' LIMIT 5",
                (province,),
            ).fetchall()
            for p in policies:
                items.append({
                    "content": f"[{p['category']}] {p['title']}: {(p['content'] or '')[:300]}",
                    "metadata": {
                        "type": "policy_doc",
                        "province": p["province"],
                        "category": p["category"],
                        "doc_number": p["doc_number"] or "",
                    },
                    "score": 0.8,
                    "source": "sqlite:policy_docs",
                })

            conn.close()
        except Exception:
            pass

        return items

    def _fallback_sqlite_policy(self, query: str, province: str, top_k: int) -> list[dict]:
        """ChromaDB 不可用时的 SQLite 关键字回退。"""
        import sqlite3
        db_path = (
            Path(r"D:\GitHub_work\china-power-market-agent-main\power_market_agent")
            / "knowledge_base"
            / "knowledge_base.db"
        )
        if not db_path.exists():
            return []

        items = []
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            terms = query.split()
            conditions = " OR ".join(
                f"content LIKE '%{t}%' OR title LIKE '%{t}%'" for t in terms if t.strip()
            )
            if conditions:
                rows = conn.execute(
                    f"SELECT * FROM policy_docs WHERE {conditions} LIMIT ?",
                    (top_k,),
                ).fetchall()
                for r in rows:
                    items.append({
                        "id": f"sqlite-policy-{r['id']}",
                        "content": f"[{r['category']}] {r['title']}: {(r['content'] or '')[:300]}",
                        "metadata": {"type": "policy_doc", "province": r['province']},
                        "score": 0.7,
                        "source": "sqlite_fallback",
                    })
            conn.close()
        except Exception:
            pass
        return items


# ═══════════════════════════════════════════════════════════════════
# 2. Augmentation Layer (增强层) — 核心新增
# ═══════════════════════════════════════════════════════════════════


class AugmentationLayer:
    """查询增强: 改写 → 多源融合 → 去噪 → 重排。"""

    def rewrite_query(self, query: str, province: str) -> str:
        """查询改写: 补充省份上下文、专业术语扩展。"""
        # 省份限定
        if province and province not in query:
            query = f"{province} {query}"

        # 术语扩展字典
        term_map = {
            "偏差": "偏差考核 偏差免考 偏差范围",
            "现货": "现货市场 日前出清 实时出清",
            "中长期": "中长期交易 双边协商 集中竞价",
            "分时": "分时电价 峰谷价差 尖峰低谷",
            "储能": "储能套利 充放电 辅助服务",
            "新能源": "风电 光伏 可再生能源 弃电",
        }
        for key, expansion in term_map.items():
            if key in query and expansion not in query:
                query = f"{query} {expansion}"

        return query

    def fuse_contexts(
        self,
        policy_docs: list[dict],
        market_data: list[dict],
        prediction_data: Optional[dict] = None,
        expert_insights: Optional[list[dict]] = None,
        market_snapshots: Optional[list[dict]] = None,
        price_history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        多源融合: RRF 跨源合并，统一排序。

        权重分配（总计 1.0）：
          - 政策文档: 0.25
          - 专家心得: 0.20
          - 市场行情快照: 0.20 (★ 新增: 日级/月级市场数据摘要)
          - 市场数据: 0.15 (SQLite结构化)
          - 预测数据: 0.12
          - 电价历史: 0.08
        """
        fused: dict[str, dict] = {}
        c = 60

        # 政策文档 (权重 0.25)
        for rank, doc in enumerate(policy_docs):
            key = doc.get("id", doc.get("content", "")[:100])
            if key not in fused:
                fused[key] = {**doc, "score": 0.0}
            fused[key]["score"] += 0.25 / (rank + 1 + c)

        # ★ 专家心得 (权重 0.20)
        if expert_insights:
            for rank, doc in enumerate(expert_insights):
                key = doc.get("id", doc.get("content", "")[:100])
                if key not in fused:
                    fused[key] = {**doc, "score": 0.0}
                fused[key]["score"] += 0.20 / (rank + 1 + c)

        # ★ 市场行情快照 (权重 0.20) — 新增: 日级/月级市场数据
        if market_snapshots:
            for rank, doc in enumerate(market_snapshots):
                key = doc.get("id", doc.get("content", "")[:100])
                if key not in fused:
                    fused[key] = {**doc, "score": 0.0}
                fused[key]["score"] += 0.20 / (rank + 1 + c)

        # 市场数据 SQLite (权重 0.15)
        for rank, doc in enumerate(market_data):
            key = doc.get("id", doc.get("content", "")[:100])
            if key not in fused:
                fused[key] = {**doc, "score": 0.0}
            fused[key]["score"] += 0.15 / (rank + 1 + c)

        # 预测数据 (权重 0.12)
        if prediction_data:
            key = f"prediction-{prediction_data.get('target_date', '')}"
            fused[key] = {
                "content": json.dumps(prediction_data, ensure_ascii=False),
                "metadata": {"type": "price_forecast"},
                "score": 0.12,
                "source": "lstm_prediction",
            }

        # 电价历史模式 (权重 0.08)
        if price_history:
            for rank, doc in enumerate(price_history):
                key = doc.get("id", doc.get("content", "")[:100])
                if key not in fused:
                    fused[key] = {**doc, "score": 0.0}
                fused[key]["score"] += 0.08 / (rank + 1 + c)

        return sorted(fused.values(), key=lambda x: x["score"], reverse=True)

    def compress(self, documents: list[dict], max_chars: int = 4000) -> str:
        """上下文压缩: 保留 Top-K 且去噪截断。"""
        parts = []
        total_chars = 0
        for doc in documents[:8]:
            content = doc.get("content", "")[:600]
            source = doc.get("source", "unknown")
            meta = doc.get("metadata", {})
            score = doc.get("score", 0)
            citation = (
                f"[来源: {source}"
                + (f" | {meta.get('province', '')}" if meta.get('province') else "")
                + (f" | {meta.get('doc_number', '')}" if meta.get('doc_number') else "")
                + f" | 相关度: {score:.2f}]"
            )
            chunk = f"{citation}\n{content}"
            if total_chars + len(chunk) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    parts.append(chunk[:remaining])
                break
            parts.append(chunk)
            total_chars += len(chunk)
        return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 3. Generation Layer (生成层) — 核心新增
# ═══════════════════════════════════════════════════════════════════


class GenerationLayer:
    """大模型生成: 增强 prompt → LLM 调用 → Citation 标注 → 事实校验。"""

    def __init__(self) -> None:
        self._llm = None

    def _get_llm(self):
        if self._llm:
            return self._llm
        from langchain_deepseek import ChatDeepSeek

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            env_path = PROJECT_ROOT / "config" / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if "DEEPSEEK_API_KEY" in line:
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not found")
        self._llm = ChatDeepSeek(model="deepseek-chat", temperature=0.15, api_key=api_key)
        return self._llm

    def build_prompt(
        self,
        query: str,
        context: str,
        prediction: Optional[dict] = None,
        task: str = "general",
    ) -> str:
        """构造增强后的完整 Prompt。"""
        task_instructions = {
            "general": "请基于上下文和预测数据，给出专业的电力市场分析回答。",
            "price_forecast": "请重点分析电价走势，结合供需因子和政策规则给出预测逻辑。",
            "policy_search": "请基于政策原文，给出准确的政策解读和适用性分析。",
            "strategy": "请给出可执行的交易策略建议，每项建议标注支撑依据。",
            "client_report": "请以客户视角，给出用电成本优化建议。",
        }
        instruction = task_instructions.get(task, task_instructions["general"])

        pred_block = ""
        if prediction:
            pred_block = f"""
## 预测模型输出
{json.dumps(prediction, ensure_ascii=False, indent=2)}

"""

        return f"""## 系统角色
你是 PowerMind，中国电力市场双引擎 AI 分析系统。你拥有量化预测模型和电力市场专业知识。

## 检索到的参考资料 (带来源标注)
{context}

{pred_block}
## 用户问题
{query}

## 回答要求
{instruction}

请在你的回答中:
1. 引用具体的来源标注 (如 [来源: sqlite:policy_docs | 山东])
2. 数值预测必须标注来自 LSTM 模型
3. 区分确定性结论和不确定性推断
4. 策略建议需落到具体可操作的步骤"""

    def generate(
        self,
        query: str,
        context: str,
        prediction: Optional[dict] = None,
        task: str = "general",
    ) -> str:
        """执行 LLM 生成。"""
        llm = self._get_llm()
        prompt = self.build_prompt(query, context, prediction, task)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def verify_facts(self, response: str, context: str) -> dict:
        """
        事实校验: 将 LLM 输出中的关键数字与检索上下文比对。
        返回校验结果字典。
        """
        import re

        issues = []
        # 提取 LLM 回答中的数字
        numbers = re.findall(r"(\d+\.?\d*)\s*(元/MWh|%|MW|MWh|元)", response)
        # 检查每个数字是否在 context 中出现
        for num, unit in numbers:
            if num not in context:
                issues.append({
                    "value": f"{num}{unit}",
                    "severity": "warning",
                    "detail": f"数值 {num}{unit} 在检索上下文中未找到直接支撑",
                })

        return {
            "total_claims": len(numbers),
            "unverified": len(issues),
            "issues": issues,
            "status": "ok" if len(issues) <= 2 else "needs_review",
        }


# ═══════════════════════════════════════════════════════════════════
# 4. PowerRAG Pipeline (完整三阶段)
# ═══════════════════════════════════════════════════════════════════


class PowerRAGPipeline:
    """完整的 RAG 流水线: Retrieval → Augmentation → Generation。"""

    def __init__(self) -> None:
        self.retrieval = RetrievalLayer()
        self.augmentation = AugmentationLayer()
        self.generation = GenerationLayer()

    def run(
        self,
        query: str,
        province: str = "全国",
        prediction: Optional[dict] = None,
        task: str = "general",
    ) -> dict:
        """
        执行完整 RAG 流程。

        Args:
            query: 用户查询
            province: 省份
            prediction: 可选的 LSTM 预测结果
            task: 任务类型 (general/price_forecast/policy_search/strategy/client_report)

        Returns:
            {
                "query": str,
                "rewritten_query": str,
                "context": str,
                "sources": list[dict],
                "response": str,
                "fact_check": dict,
            }
        """
        # ── ① Retrieval ──
        rewritten = self.augmentation.rewrite_query(query, province)
        policy_docs = self.retrieval.retrieve_policy(rewritten, province)
        market_data = self.retrieval.retrieve_market_data(province)

        # ★ 新增: 检索该省专家实战心得
        expert_insights = self.retrieval.retrieve_expert_insight(
            rewritten, province
        )

        # ★ 新增: 检索市场行情快照 (日级+月级)
        market_snapshots = self.retrieval.retrieve_market_snapshot(
            rewritten, province
        )

        # ★ 新增: 检索历史电价模式（如果有日期）
        price_history = []
        if prediction and prediction.get("target_date"):
            price_history = self.retrieval.retrieve_price_history(
                province, prediction["target_date"]
            )

        # ── ② Augmentation ──
        fused = self.augmentation.fuse_contexts(
            policy_docs, market_data, prediction, expert_insights, market_snapshots, price_history
        )
        context = self.augmentation.compress(fused)

        # ── ③ Generation ──
        response = self.generation.generate(query, context, prediction, task)
        fact_check = self.generation.verify_facts(response, context)

        return {
            "query": query,
            "rewritten_query": rewritten,
            "context": context,
            "sources": fused,
            "response": response,
            "fact_check": fact_check,
        }


# ── 全局单例 ─────────────────────────────────────────────────

_rag_pipeline: Optional[PowerRAGPipeline] = None


def get_rag_pipeline() -> PowerRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = PowerRAGPipeline()
    return _rag_pipeline


# ── CLI 测试 ─────────────────────────────────────────────────

if __name__ == "__main__":
    # 加载 .env
    env_path = PROJECT_ROOT / "config" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    pipeline = PowerRAGPipeline()
    result = pipeline.run(
        query="山东现货交易偏差考核标准是什么？",
        province="山东",
        prediction={"target_date": "2026-05-21", "predicted_price": 50.84},
        task="policy_search",
    )

    print(f"原始查询: {result['query']}")
    print(f"改写查询: {result['rewritten_query']}")
    print(f"上下文长度: {len(result['context'])} chars")
    print(f"检索来源数: {len(result['sources'])}")
    print(f"事实校验: {result['fact_check']}")
    print(f"\n{'='*60}")
    print("LLM 生成结果:")
    print(f"{'='*60}")
    print(result["response"])
