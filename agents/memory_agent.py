"""Memory Agent — Multi-level memory management.

Three memory tiers:
  1. Working Memory  — current conversation context (short-lived, per-session)
  2. Episodic Memory — historical interactions and outcomes (persisted, searchable)
  3. Semantic Memory — persistent rules, policies, and domain knowledge (immutable-ish)

Memory is injected into agent conversation flow at plan time and generation time.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory entry types
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single memory record."""

    memory_id: str
    memory_type: str  # "working" | "episodic" | "semantic"
    content: str
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5  # 0.0 - 1.0
    ttl_seconds: int = 0  # 0 = permanent
    access_count: int = 0


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------


class WorkingMemory:
    """Short-lived, per-session conversation context.

    Holds the current query, intermediate results, and agent state.
    Limited to WORKING_MEMORY_MAX entries (oldest evicted on overflow).
    """

    WORKING_MEMORY_MAX = 20

    def __init__(self):
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()

    def add(self, key: str, content: str, metadata: dict | None = None) -> None:
        if len(self._entries) >= self.WORKING_MEMORY_MAX:
            # Evict oldest
            oldest = next(iter(self._entries))
            del self._entries[oldest]
        entry = MemoryEntry(
            memory_id=key,
            memory_type="working",
            content=content,
            metadata=metadata or {},
        )
        self._entries[key] = entry

    def get(self, key: str) -> MemoryEntry | None:
        entry = self._entries.get(key)
        if entry:
            entry.access_count += 1
        return entry

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def to_context_string(self) -> str:
        """Format working memory for injection into prompts."""
        if not self._entries:
            return ""
        parts = ["## 当前工作记忆"]
        for key, entry in self._entries.items():
            parts.append(f"- [{key}] {entry.content[:200]}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """Historical interaction store with simple relevance-based retrieval.

    Persisted to disk as JSONL for durability.
    """

    def __init__(self, storage_path: str = "data/episodic_memory.jsonl"):
        self._storage_path = Path(storage_path)
        self._entries: list[MemoryEntry] = []
        self._load()

    def record(
        self,
        query: str,
        result_summary: str,
        metadata: dict | None = None,
        importance: float = 0.5,
    ) -> str:
        """Record an interaction for future recall."""
        import uuid

        entry = MemoryEntry(
            memory_id=str(uuid.uuid4())[:8],
            memory_type="episodic",
            content=json.dumps(
                {"query": query, "result": result_summary},
                ensure_ascii=False,
            ),
            metadata=metadata or {},
            importance=importance,
            ttl_seconds=30 * 24 * 3600,  # 30 days default TTL
        )
        self._entries.append(entry)
        self._append_to_file(entry)
        self._prune_expired()
        return entry.memory_id

    def retrieve(
        self, query: str, top_k: int = 5, min_importance: float = 0.3
    ) -> list[MemoryEntry]:
        """Retrieve relevant past interactions by keyword match.

        A production system would use embedding similarity; this implementation
        uses keyword overlap for simplicity and deterministic behaviour.
        """
        query_keywords = set(query.lower().split())
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._entries:
            if entry.importance < min_importance:
                continue
            if self._is_expired(entry):
                continue

            content_lower = entry.content.lower()
            hits = sum(1 for kw in query_keywords if kw in content_lower)
            if hits > 0:
                score = hits / len(query_keywords) * entry.importance
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        entry_list = [e for _, e in scored[:top_k]]
        for e in entry_list:
            e.access_count += 1
        return entry_list

    def to_context_string(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant memories and format for prompt injection."""
        entries = self.retrieve(query, top_k=top_k)
        if not entries:
            return ""
        parts = ["## 历史相关交互"]
        for e in entries:
            parts.append(f"- [{e.memory_id}] {e.content[:200]}")
        return "\n".join(parts)

    # ----- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            for line in self._storage_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                self._entries.append(
                    MemoryEntry(
                        memory_id=data.get("memory_id", ""),
                        memory_type="episodic",
                        content=data.get("content", ""),
                        metadata=data.get("metadata", {}),
                        timestamp=data.get("timestamp", ""),
                        importance=data.get("importance", 0.5),
                        ttl_seconds=data.get("ttl_seconds", 0),
                    )
                )
        except Exception as exc:
            logger.warning("EpisodicMemory: load failed — %s", exc)

    def _append_to_file(self, entry: MemoryEntry) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._storage_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "memory_id": entry.memory_id,
                            "content": entry.content,
                            "metadata": entry.metadata,
                            "timestamp": entry.timestamp,
                            "importance": entry.importance,
                            "ttl_seconds": entry.ttl_seconds,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            logger.warning("EpisodicMemory: write failed — %s", exc)

    def _prune_expired(self) -> None:
        self._entries = [e for e in self._entries if not self._is_expired(e)]

    @staticmethod
    def _is_expired(entry: MemoryEntry) -> bool:
        if entry.ttl_seconds <= 0:
            return False
        try:
            ts = datetime.fromisoformat(entry.timestamp)
            return datetime.now() - ts > timedelta(seconds=entry.ttl_seconds)
        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------


class SemanticMemory:
    """Persistent domain rules and knowledge.

    Stores province-specific market rules, regulatory constraints, and
    operational heuristics. Loaded from YAML configs and optionally
    supplemented via ChromaDB.
    """

    def __init__(self, config_dir: str = "data/configs"):
        self._config_dir = Path(config_dir)
        self._rules: dict[str, list[dict]] = {}  # province -> [rule dicts]
        self._load_from_configs()

    def get_rules(self, province: str) -> list[dict]:
        """Get all semantic rules for a province."""
        province_rules = self._rules.get(province, [])
        national_rules = self._rules.get("全国", [])
        return province_rules + national_rules

    def add_rule(
        self, province: str, rule: dict, importance: float = 0.7
    ) -> None:
        rule["importance"] = importance
        self._rules.setdefault(province, []).append(rule)

    def to_context_string(self, province: str = "") -> str:
        """Format semantic memory for prompt injection."""
        parts = ["## 持久规则与约束"]
        if province:
            rules = self.get_rules(province)
            parts.append(f"### {province} 市场规则")
            for r in rules:
                parts.append(
                    f"- {r.get('name', '')}: {r.get('value', '')} "
                    f"(来源: {r.get('source', '未知')})"
                )
        return "\n".join(parts) if len(parts) > 1 else ""

    def _load_from_configs(self) -> None:
        """Parse province YAML configs into semantic rules."""
        import yaml

        for yf in self._config_dir.glob("*.yaml"):
            try:
                cfg = yaml.safe_load(yf.read_text(encoding="utf-8"))
            except Exception:
                continue

            province_cfg = cfg.get("province", {})
            province = province_cfg.get("name", yf.stem)
            rules: list[dict] = []

            # Spot market rules
            spot = cfg.get("spot_market", {})
            if spot.get("price_cap"):
                rules.append({
                    "name": "现货价格上限",
                    "value": f"{spot['price_cap']} 元/MWh",
                    "source": f"{province}市场规则",
                })
            if spot.get("price_floor") is not None:
                rules.append({
                    "name": "现货价格下限",
                    "value": f"{spot['price_floor']} 元/MWh",
                    "source": f"{province}市场规则",
                })

            # Deviation rules
            dev = cfg.get("deviation", {})
            if dev.get("free_range") is not None:
                rules.append({
                    "name": "偏差免考范围",
                    "value": f"±{dev['free_range']}%",
                    "source": f"{province}偏差考核规则",
                })
            if dev.get("penalty_multiplier"):
                rules.append({
                    "name": "偏差考核惩罚倍数",
                    "value": f"{dev['penalty_multiplier']}倍",
                    "source": f"{province}偏差考核规则",
                })

            # TOU segments
            tou = cfg.get("tou", {})
            for seg in tou.get("segments", []):
                hours = seg.get("hours", [])
                hours_str = ", ".join(hours[:3]) if hours else "未指定时段"
                rules.append({
                    "name": f"分时电价-{seg.get('name', '')}",
                    "value": f"{hours_str} (倍率: {seg.get('multiplier', '?')})",
                    "source": f"{province}分时电价规则",
                })

            # Energy mix
            emix = cfg.get("energy_mix", {})
            if emix:
                rules.append({
                    "name": "能源结构",
                    "value": ", ".join(f"{k}: {v}%" for k, v in emix.items()),
                    "source": f"{province}能源局数据",
                })

            self._rules[province] = rules


# ---------------------------------------------------------------------------
# Memory Agent (composite)
# ---------------------------------------------------------------------------


class MemoryAgent:
    """Composite memory manager that unifies all three memory tiers.

    Usage:
        memory = MemoryAgent()
        memory.working.add("query", "山东明日现货价格预测")
        memory.episodic.record("山东现货价格预测", "均价380元/MWh")
        context = memory.build_context("山东现货价格", province="山东")
    """

    def __init__(
        self,
        working: WorkingMemory | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
    ):
        self.working = working or WorkingMemory()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()

    def build_context(
        self,
        query: str,
        province: str = "",
        include_episodic: bool = True,
        include_semantic: bool = True,
    ) -> str:
        """Build a combined memory context string for prompt injection."""
        parts = []

        # Working memory (always included)
        wm = self.working.to_context_string()
        if wm:
            parts.append(wm)

        # Episodic memory (relevant history)
        if include_episodic:
            em = self.episodic.to_context_string(query)
            if em:
                parts.append(em)

        # Semantic memory (rules)
        if include_semantic and province:
            sm = self.semantic.to_context_string(province)
            if sm:
                parts.append(sm)

        return "\n\n".join(parts)

    def record_interaction(
        self,
        query: str,
        result: str,
        province: str = "",
        importance: float = 0.5,
    ) -> None:
        """Record a completed interaction in episodic memory."""
        self.episodic.record(
            query=query,
            result_summary=result,
            metadata={"province": province} if province else {},
            importance=importance,
        )

    # ── Redis-backed memory (enterprise deployment) ──────────────────

    async def build_context_redis(
        self,
        query: str,
        tenant_id: str = "default",
        province: str = "",
        include_episodic: bool = True,
        include_semantic: bool = True,
    ) -> str:
        """Build memory context from Redis cache (async, enterprise path).

        Falls back gracefully to local WorkingMemory/EpisodicMemory if Redis
        is unavailable. Used by Orchestrator when USE_REDIS=true.
        """
        try:
            from infrastructure.cache import MemoryCache
            mem_cache = MemoryCache()
        except ImportError:
            return self.build_context(query, province, include_episodic, include_semantic)

        parts = []

        # Working memory from Redis
        wm_entries = await mem_cache.wm_get_all(tenant_id)
        if wm_entries:
            parts.append("## 当前工作记忆 (Redis)")
            for entry in wm_entries[:10]:
                parts.append(f"- {entry[:200]}")
        else:
            # Fallback to local
            wm = self.working.to_context_string()
            if wm:
                parts.append(wm)

        # Episodic memory from Redis
        if include_episodic:
            em_results = await mem_cache.em_search(tenant_id, query)
            if em_results:
                parts.append("## 历史相关交互 (Redis)")
                for entry in em_results[:5]:
                    data = entry.get("data", {})
                    parts.append(f"- [{entry.get('memory_id', '?')}] {str(data)[:200]}")
            else:
                em = self.episodic.to_context_string(query)
                if em:
                    parts.append(em)

        # Semantic memory (from local YAML configs, always local)
        if include_semantic and province:
            sm = self.semantic.to_context_string(province)
            if sm:
                parts.append(sm)

        return "\n\n".join(parts)

    async def record_interaction_redis(
        self,
        query: str,
        result: str,
        tenant_id: str = "default",
        province: str = "",
        importance: float = 0.5,
    ) -> None:
        """Record interaction in both Redis episodic memory and local JSONL fallback."""
        # Local fallback (always recorded)
        self.episodic.record(
            query=query,
            result_summary=result,
            metadata={"province": province, "tenant_id": tenant_id},
            importance=importance,
        )

        # Redis cache
        try:
            from infrastructure.cache import MemoryCache
            import uuid
            mem_cache = MemoryCache()
            await mem_cache.em_record(
                tenant_id=tenant_id,
                memory_id=str(uuid.uuid4())[:8],
                data={"query": query, "result": result[:500], "province": province},
                importance=importance,
            )
        except (ImportError, Exception):
            pass  # Redis is optional

    async def wm_push_redis(
        self,
        tenant_id: str,
        entry: str,
    ) -> None:
        """Push to Redis working memory with TTL + FIFO."""
        # Local fallback
        self.working.add(entry[:50], entry, {"tenant_id": tenant_id})

        # Redis cache
        try:
            from infrastructure.cache import MemoryCache
            mem_cache = MemoryCache()
            await mem_cache.wm_push(tenant_id, entry)
        except (ImportError, Exception):
            pass

    async def wm_clear_redis(self, tenant_id: str) -> None:
        """Clear Redis working memory for a tenant."""
        self.working.clear()
        try:
            from infrastructure.cache import MemoryCache
            mem_cache = MemoryCache()
            await mem_cache.wm_clear(tenant_id)
        except (ImportError, Exception):
            pass
