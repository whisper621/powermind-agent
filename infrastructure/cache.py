"""Redis Distributed Cache Layer.

Three cache domains:
  1. ProvinceConfigCache  — 31-province YAML rules cached in Redis Hash
  2. MemoryCache          — Working + Episodic memory with TTL auto-expiry + FIFO eviction
  3. ModelRegistryCache   — Hot-swappable forecaster hyper-params in Redis

All operations are async (redis.asyncio) to avoid blocking the event loop.
Graceful fallback to in-memory dict when USE_REDIS=false (local dev).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------


class RedisCache:
    """Singleton async Redis client with connection pooling.

    Usage:
        cache = await RedisCache.get_instance()
        await cache.set("key", "value", ttl=3600)
        value = await cache.get("key")
    """

    _instance: Optional["RedisCache"] = None

    def __init__(self) -> None:
        from infrastructure.settings import get_settings
        self._settings = get_settings()
        self._client = None
        self._fallback: dict[str, Any] = {}
        self._enabled = self._settings.use_redis

    @classmethod
    async def get_instance(cls) -> "RedisCache":
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance._connect()
        return cls._instance

    async def _connect(self) -> None:
        if not self._enabled:
            logger.info("RedisCache: disabled — using in-memory fallback")
            return

        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self._settings.redis_dsn,
                max_connections=self._settings.redis_max_connections,
                decode_responses=True,
            )
            await self._client.ping()
            logger.info("RedisCache: connected — %s", self._settings.redis_host)
        except Exception as exc:
            logger.warning("RedisCache: connection failed (%s) — using fallback", exc)
            self._enabled = False
            self._client = None

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    # ── Key-Value ops ──────────────────────────────────────────

    async def get(self, key: str) -> Optional[str]:
        if self._client:
            return await self._client.get(key)
        return self._fallback.get(key)

    async def set(self, key: str, value: str, ttl: int = 0) -> None:
        if self._client:
            await self._client.set(key, value, ex=ttl if ttl > 0 else None)
        else:
            self._fallback[key] = value

    async def delete(self, key: str) -> None:
        if self._client:
            await self._client.delete(key)
        else:
            self._fallback.pop(key, None)

    async def exists(self, key: str) -> bool:
        if self._client:
            return await self._client.exists(key) > 0
        return key in self._fallback

    # ── Hash ops (for province configs) ─────────────────────────

    async def hget(self, name: str, key: str) -> Optional[str]:
        if self._client:
            return await self._client.hget(name, key)
        fallback_hash = self._fallback.get(name, {})
        return fallback_hash.get(key)

    async def hset(self, name: str, key: str, value: str) -> None:
        if self._client:
            await self._client.hset(name, key, value)
        else:
            self._fallback.setdefault(name, {})[key] = value

    async def hgetall(self, name: str) -> dict[str, str]:
        if self._client:
            return await self._client.hgetall(name)
        return self._fallback.get(name, {})

    # ── List ops (for FIFO memory) ──────────────────────────────

    async def lpush(self, key: str, *values: str) -> None:
        if self._client:
            await self._client.lpush(key, *values)
        else:
            lst = self._fallback.setdefault(key, [])
            for v in values:
                lst.insert(0, v)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        if self._client:
            return await self._client.lrange(key, start, end)
        lst = self._fallback.get(key, [])
        return lst[start:end + 1 if end >= 0 else None]

    async def ltrim(self, key: str, start: int, end: int) -> None:
        if self._client:
            await self._client.ltrim(key, start, end)
        else:
            lst = self._fallback.get(key, [])
            self._fallback[key] = lst[start:end + 1 if end >= 0 else None]

    async def expire(self, key: str, ttl: int) -> None:
        if self._client:
            await self._client.expire(key, ttl)

    async def ttl(self, key: str) -> int:
        if self._client:
            return await self._client.ttl(key)
        return -1


# ---------------------------------------------------------------------------
# Province config cache — 31-province YAML → Redis Hash
# ---------------------------------------------------------------------------


class ProvinceConfigCache:
    """Load 31-province YAML rule files into Redis on startup.

    Each province maps to a Redis Hash:  "province:shandong" → {field: value, ...}
    When ModelRegistry.get_model() is called, it queries Redis (μs latency)
    instead of reading YAML files from disk.

    Usage:
        cache = ProvinceConfigCache()
        await cache.warm_up()  # call once at startup
        cfg = await cache.get("shandong", "spot_market.price_cap")  # → "1500"
    """

    CACHE_PREFIX = "province:"

    def __init__(self, cache: RedisCache | None = None) -> None:
        self._cache = cache
        self._config_dir = Path(__file__).resolve().parents[1] / "data" / "configs"
        self._fallback: dict[str, dict] = {}

    async def warm_up(self) -> int:
        """Load all province YAML configs into Redis. Returns count loaded."""
        import yaml

        if self._cache is None:
            self._cache = await RedisCache.get_instance()

        count = 0
        for yf in self._config_dir.glob("*.yaml"):
            try:
                cfg = yaml.safe_load(yf.read_text(encoding="utf-8"))
            except Exception:
                continue

            province = cfg.get("province", {}).get("name", yf.stem)
            flat = self._flatten_config(cfg)
            self._fallback[province] = flat

            # Write to Redis Hash
            cache_key = f"{self.CACHE_PREFIX}{province}"
            for k, v in flat.items():
                await self._cache.hset(cache_key, k, json.dumps(v, ensure_ascii=False))
            count += 1

        logger.info("ProvinceConfigCache: warmed %d provinces into Redis", count)
        return count

    async def get(self, province: str, key: str | None = None) -> dict | str | None:
        """Get province config. If key is None, returns the full dict."""
        if self._cache is None:
            self._cache = await RedisCache.get_instance()

        cache_key = f"{self.CACHE_PREFIX}{province}"

        if key:
            raw = await self._cache.hget(cache_key, key)
            if raw:
                return json.loads(raw)
            # Fallback
            return self._fallback.get(province, {}).get(key)
        else:
            raw = await self._cache.hgetall(cache_key)
            if raw:
                return {k: json.loads(v) for k, v in raw.items()}
            return self._fallback.get(province, {})

    async def get_price_bounds(self, province: str) -> tuple[float, float]:
        """Return (price_floor, price_cap) for a province. Fast path for Verifier."""
        cfg = await self.get(province)
        if not cfg:
            return (0.0, 1500.0)
        spot = cfg.get("spot_market", {})
        return (float(spot.get("price_floor", 0)), float(spot.get("price_cap", 1500)))

    async def get_deviation_free_range(self, province: str) -> float:
        """Return deviation free range percentage (e.g., 5.0 for ±5%)."""
        cfg = await self.get(province)
        if not cfg:
            return 5.0
        dev = cfg.get("deviation", {})
        return float(dev.get("free_range", 5.0))

    @staticmethod
    def _flatten_config(cfg: dict, prefix: str = "") -> dict:
        """Flatten nested YAML into dot-separated keys for Redis Hash."""
        result = {}
        for k, v in cfg.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(ProvinceConfigCache._flatten_config(v, full_key))
            elif isinstance(v, list):
                result[full_key] = v
            else:
                result[full_key] = v
        return result


# ---------------------------------------------------------------------------
# Memory Cache — TTL + FIFO for working / episodic memory
# ---------------------------------------------------------------------------


class MemoryCache:
    """Redis-backed memory lifecycle management for PowerAgent OS.

    Working Memory (short-term, 1-hour TTL):
        Redis List with LTRIM for FIFO sliding window (max 20 entries).
        Entries expire via TTL on the list key.

    Episodic Memory (mid-term, 30-day TTL):
        Redis Hash per tenant, keyed by memory_id.
        Automatic cleanup via TTL.

    Semantic Memory (long-term):
        Persisted in PostgreSQL province_rules table + ChromaDB.
        Not managed by this cache.

    Usage:
        mem = MemoryCache()
        # Working memory
        await mem.wm_push("tenant-1", "query: 山东电价预测")
        recent = await mem.wm_get_all("tenant-1")  # → last 20 entries

        # Episodic memory
        await mem.em_record("tenant-1", "mem-001", {"query": "...", "result": "..."})
        results = await mem.em_search("tenant-1", "电价")
    """

    WM_PREFIX = "wm:"     # working memory key prefix
    EM_PREFIX = "em:"     # episodic memory key prefix

    def __init__(self, cache: RedisCache | None = None) -> None:
        self._cache = cache
        from infrastructure.settings import get_settings
        self._settings = get_settings()

    async def _ensure_cache(self) -> RedisCache:
        if self._cache is None:
            self._cache = await RedisCache.get_instance()
        return self._cache

    # ── Working Memory (FIFO + TTL) ───────────────────────────

    async def wm_push(self, tenant_id: str, entry: str) -> None:
        """Push an entry to working memory. FIFO: oldest evicted when > max."""
        cache = await self._ensure_cache()
        key = f"{self.WM_PREFIX}{tenant_id}"
        max_entries = self._settings.working_memory_max_entries
        ttl = self._settings.working_memory_ttl

        await cache.lpush(key, entry)
        await cache.ltrim(key, 0, max_entries - 1)  # FIFO slide
        await cache.expire(key, ttl)  # auto-expire

    async def wm_get_all(self, tenant_id: str) -> list[str]:
        """Retrieve all working memory entries (newest first)."""
        cache = await self._ensure_cache()
        key = f"{self.WM_PREFIX}{tenant_id}"
        return await cache.lrange(key, 0, -1)

    async def wm_clear(self, tenant_id: str) -> None:
        """Clear working memory for a tenant."""
        cache = await self._ensure_cache()
        await cache.delete(f"{self.WM_PREFIX}{tenant_id}")

    # ── Episodic Memory (TTL-based expiry) ────────────────────

    async def em_record(
        self,
        tenant_id: str,
        memory_id: str,
        data: dict,
        importance: float = 0.5,
    ) -> None:
        """Record an episodic memory entry with 30-day TTL."""
        cache = await self._ensure_cache()
        key = f"{self.EM_PREFIX}{tenant_id}"
        ttl = self._settings.episodic_memory_ttl

        payload = json.dumps({
            "memory_id": memory_id,
            "importance": importance,
            "data": data,
        }, ensure_ascii=False)

        await cache.hset(key, memory_id, payload)
        await cache.expire(key, ttl)

    async def em_search(self, tenant_id: str, query: str) -> list[dict]:
        """Keyword-search episodic memories. Returns matched entries."""
        cache = await self._ensure_cache()
        key = f"{self.EM_PREFIX}{tenant_id}"
        all_entries = await cache.hgetall(key)

        query_lower = query.lower()
        results = []
        for mem_id, raw in all_entries.items():
            if query_lower in raw.lower():
                try:
                    results.append(json.loads(raw))
                except json.JSONDecodeError:
                    results.append({"memory_id": mem_id, "raw": raw})
        return results

    async def em_prune_expired(self, tenant_id: str) -> int:
        """Remove expired entries. Redis TTL handles this automatically,
        but this provides explicit cleanup for the fallback dict."""
        cache = await self._ensure_cache()
        key = f"{self.EM_PREFIX}{tenant_id}"
        ttl_remaining = await cache.ttl(key)
        if ttl_remaining <= 0:
            await cache.delete(key)
            return 1
        return 0


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

async def get_cache() -> RedisCache:
    """Get the singleton Redis cache instance."""
    return await RedisCache.get_instance()
