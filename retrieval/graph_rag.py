"""GraphRAG — Power domain entity graph with multi-hop relational retrieval.

Models the power-market domain as a property graph:
  Province -> PolicyDocument -> Rule -> (related entities)

Fallback: when graph DB is unavailable, operates as an in-memory NetworkX graph
loaded from structured YAML/SQLite sources.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from retrieval.base_retriever import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain entity models
# ---------------------------------------------------------------------------


@dataclass
class PowerEntity:
    """Base entity in the power-market knowledge graph."""

    entity_id: str
    entity_type: str = "Entity"  # Province | PolicyDocument | Rule | MarketNode | Company
    name: str = ""
    properties: dict = field(default_factory=dict)


@dataclass
class Province(PowerEntity):
    """Province-level market entity."""

    region: str = ""
    market_type: str = ""  # SPOT / 中长期
    energy_mix: dict = field(default_factory=dict)

    def __post_init__(self):
        self.entity_type = "Province"


@dataclass
class PolicyDocument(PowerEntity):
    """Policy document entity."""

    doc_number: str = ""
    publish_date: str = ""
    effective_date: str = ""
    issuing_body: str = ""
    category: str = ""
    url: str = ""

    def __post_init__(self):
        self.entity_type = "PolicyDocument"


@dataclass
class Rule(PowerEntity):
    """A specific rule extracted from a policy document."""

    rule_type: str = ""  # price_cap | deviation | tou | reserve | emission
    parent_policy_id: str = ""
    applicable_provinces: list[str] = field(default_factory=list)
    value: str = ""
    unit: str = ""

    def __post_init__(self):
        self.entity_type = "Rule"


# ---------------------------------------------------------------------------
# GraphRAG engine
# ---------------------------------------------------------------------------


class GraphRAG(BaseRetriever):
    """In-memory power-domain knowledge graph with multi-hop traversal.

    Operates as a NetworkX DiGraph. Nodes are PowerEntity instances keyed by
    entity_id.  Edges represent typed relationships (PUBLISHED, DERIVED_FROM,
    APPLIES_TO, etc.).
    """

    def __init__(self, config_dir: str = "data/configs"):
        self._graph = nx.DiGraph()
        self._config_dir = Path(config_dir)
        self._loaded = False

    # ----- graph construction ------------------------------------------------

    def build_from_configs(self) -> int:
        """Build the graph from province YAML configs and SQLite policy data."""
        count = 0
        count += self._load_provinces()
        count += self._load_policy_graph()
        self._loaded = True
        logger.info("GraphRAG: built graph with %d nodes", count)
        return count

    def _load_provinces(self) -> int:
        count = 0
        for yf in self._config_dir.glob("*.yaml"):
            try:
                cfg = yaml.safe_load(yf.read_text(encoding="utf-8"))
            except Exception:
                continue

            province_cfg = cfg.get("province", {})
            name = province_cfg.get("name", yf.stem)
            pid = province_cfg.get("code", yf.stem)

            province = Province(
                entity_id=f"province:{pid}",
                name=name,
                region=province_cfg.get("region", ""),
                market_type=province_cfg.get("market_stage", ""),
                energy_mix=cfg.get("energy_mix", {}),
                properties={
                    "spot_cap": cfg.get("spot_market", {}).get("price_cap", 1500),
                    "spot_floor": cfg.get("spot_market", {}).get("price_floor", 0),
                    "deviation_free_pct": cfg.get("deviation", {}).get("free_range", 5),
                    "deviation_penalty_mult": cfg.get("deviation", {}).get("penalty_multiplier", 2),
                    "tou_segments": [
                        {"name": s.get("name"), "hours": s.get("hours", []), "multiplier": s.get("multiplier", 1)}
                        for s in cfg.get("tou", {}).get("segments", [])
                    ],
                    "transmission_corridors": cfg.get("transmission", []),
                },
            )
            self._graph.add_node(province.entity_id, entity=province)
            count += 1
        return count

    def _load_policy_graph(self) -> int:
        """Load policy-doc and rule nodes from SQLite, link to provinces.

        Falls back gracefully when the external SQLite is unavailable.
        """
        db_paths = [
            Path("D:/GitHub_work/china-power-market-agent-main/power_market_agent/knowledge_base/knowledge_base.db"),
            Path("data/knowledge_base.db"),
        ]
        db_path = None
        for p in db_paths:
            if p.exists():
                db_path = p
                break

        if db_path is None:
            logger.warning("GraphRAG: no SQLite policy DB found — graph will only contain province nodes")
            return 0

        count = 0
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # ★ 先检查 policy_docs 表是否存在
            tables = set(
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            )
            if "policy_docs" not in tables:
                conn.close()
                logger.info("GraphRAG: no policy_docs table — skipping policy graph load")
                return 0

            cur = conn.execute(
                "SELECT id, title, doc_number, publish_date, issuing_body, "
                "category, province, url, content FROM policy_docs"
            )
            for row in cur.fetchall():
                doc = PolicyDocument(
                    entity_id=f"policy:{row['id']}",
                    name=row["title"] or "",
                    doc_number=row["doc_number"] or "",
                    publish_date=str(row["publish_date"]) if row["publish_date"] else "",
                    issuing_body=row["issuing_body"] or "",
                    category=row["category"] or "",
                    url=row["url"] or "",
                    properties={"content_snippet": (row["content"] or "")[:500]},
                )
                self._graph.add_node(doc.entity_id, entity=doc)

                # Link document -> province
                prov = row["province"] or ""
                if prov:
                    pid = f"province:{prov}"
                    if pid in self._graph:
                        self._graph.add_edge(doc.entity_id, pid, relation="APPLIES_TO")
                else:
                    # National policy: link to all provinces
                    for nid in list(self._graph.nodes):
                        if nid.startswith("province:"):
                            self._graph.add_edge(doc.entity_id, nid, relation="APPLIES_TO")

                count += 1
            conn.close()
        except Exception as exc:
            logger.warning("GraphRAG: policy-doc load failed — %s", exc)

        return count

    def upsert_entity(self, entity: PowerEntity) -> None:
        """Insert or update a single entity in the graph."""
        self._graph.add_node(entity.entity_id, entity=entity)

    def add_relationship(
        self, from_id: str, to_id: str, relation: str
    ) -> None:
        self._graph.add_edge(from_id, to_id, relation=relation)

    # ----- retrieval ---------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 10, **filters
    ) -> list[RetrievalResult]:
        """Multi-hop retrieval starting from entities matching the query.

        Strategy:
          1. Find seed nodes by name/type substring match.
          2. Traverse 1-2 hops via typed edges.
          3. Rank by graph centrality and return as RetrievalResults.
        """
        if not self._loaded:
            self.build_from_configs()

        province = filters.get("province")
        seeds = self._find_seeds(query, province)
        if not seeds:
            return self._fallback_text_search(query, top_k, province)

        results: list[RetrievalResult] = []

        for seed_id in seeds[:5]:
            entity = self._graph.nodes[seed_id].get("entity")
            if entity is None:
                continue

            # Collect 1-hop neighbours
            neighbours = []
            for _, dst, edge_data in self._graph.out_edges(seed_id, data=True):
                nb_entity = self._graph.nodes[dst].get("entity")
                if nb_entity:
                    neighbours.append((dst, nb_entity, edge_data.get("relation", "")))
            for src, _, edge_data in self._graph.in_edges(seed_id, data=True):
                nb_entity = self._graph.nodes[src].get("entity")
                if nb_entity:
                    neighbours.append((src, nb_entity, edge_data.get("relation", "")))

            # Score neighbours by graph degree (centrality heuristic)
            for nb_id, nb_entity, rel in neighbours:
                degree = self._graph.degree(nb_id)
                score = min(degree / max(1, len(self._graph)), 1.0)

                results.append(
                    RetrievalResult(
                        doc_id=nb_id,
                        content=self._entity_to_text(nb_entity, relation=rel),
                        source="graph",
                        score=round(score, 4),
                        metadata={
                            "entity_type": nb_entity.entity_type,
                            "relation": rel,
                            "seed_id": seed_id,
                        },
                        citations=[nb_entity.name],
                    )
                )

        # Deduplicate by doc_id
        seen: set[str] = set()
        unique: list[RetrievalResult] = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                unique.append(r)
        return unique[:top_k]

    # ----- helpers -----------------------------------------------------------

    def _find_seeds(self, query: str, province: str | None) -> list[str]:
        seeds: list[tuple[str, float]] = []
        q_lower = query.lower()
        for nid in self._graph.nodes:
            entity = self._graph.nodes[nid].get("entity")
            if entity is None:
                continue
            name_lower = entity.name.lower()
            # Simple substring match scoring
            if q_lower in name_lower or any(
                kw in name_lower for kw in q_lower.split()
            ):
                score = len(q_lower) / max(1, len(name_lower))
                seeds.append((nid, score))
            # Also match by entity_type keywords
            etype = entity.entity_type.lower()
            if etype in q_lower:
                seeds.append((nid, 0.3))
        # Province filter
        if province:
            pid = f"province:{province}"
            seeds = [(n, s) for n, s in seeds if pid in n or not n.startswith("province:")]
        seeds.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in seeds[:10]]

    def _fallback_text_search(
        self, query: str, top_k: int, province: str | None
    ) -> list[RetrievalResult]:
        """Fallback: return all entities whose name/text contains query keywords."""
        results: list[RetrievalResult] = []
        keywords = query.lower().split()
        for nid in self._graph.nodes:
            entity = self._graph.nodes[nid].get("entity")
            if entity is None:
                continue
            text = f"{entity.name} {entity.entity_type} {json.dumps(entity.properties, ensure_ascii=False)}".lower()
            if any(kw in text for kw in keywords):
                if province and nid.startswith("province:") and province not in nid:
                    continue
                results.append(
                    RetrievalResult(
                        doc_id=nid,
                        content=self._entity_to_text(entity),
                        source="graph",
                        score=0.5,
                        metadata={"entity_type": entity.entity_type},
                        citations=[entity.name],
                    )
                )
        return results[:top_k]

    @staticmethod
    def _entity_to_text(entity: PowerEntity, relation: str = "") -> str:
        rel_str = f" (关系: {relation})" if relation else ""
        props = json.dumps(entity.properties, ensure_ascii=False, default=str)
        return f"[{entity.entity_type}]{rel_str} {entity.name}\n属性: {props}"

    def health_check(self) -> bool:
        return self._loaded or bool(self._graph.nodes)

    # ----- graph persistence (NEW) -------------------------------------------

    _GRAPH_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "graphdb" / "knowledge_graph.gpickle"

    def save_graph(self, path: str | None = None) -> str:
        """
        序列化知识图谱到磁盘。使用 NetworkX write_gpickle 格式。

        Returns: 保存路径
        """
        import pickle
        target = Path(path) if path else self._GRAPH_CACHE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self._graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("GraphRAG: 图谱已保存 (%d nodes, %d edges) → %s",
                     self._graph.number_of_nodes(), self._graph.number_of_edges(), target)
        return str(target)

    def load_graph(self, path: str | None = None) -> bool:
        """
        从磁盘反序列化知识图谱。

        如果缓存存在且比源数据新，则直接加载缓存（跳过重建）。
        如果缓存过期或不存在，返回 False 表示需要重建。
        """
        import pickle
        target = Path(path) if path else self._GRAPH_CACHE_PATH
        if not target.exists():
            return False

        try:
            # 检查缓存是否过期 (源YAML/SQLite数据比缓存新?)
            cache_mtime = target.stat().st_mtime
            # 检查 configs 目录的修改时间
            config_mtime = max(
                (f.stat().st_mtime for f in (PROJECT_ROOT / "data" / "configs").rglob("*.yaml") if f.is_file()),
                default=0
            )
            if config_mtime > cache_mtime:
                logger.info("GraphRAG: 源数据已更新，缓存过期，将重建图谱")
                return False

            with open(target, "rb") as f:
                self._graph = pickle.load(f)
            self._loaded = True
            logger.info("GraphRAG: 图谱已加载 (%d nodes, %d edges) ← %s",
                         self._graph.number_of_nodes(), self._graph.number_of_edges(), target)
            return True
        except Exception as e:
            logger.warning("GraphRAG: 图谱加载失败 (%s), 将重建", e)
            return False

    def build_or_load(self) -> int:
        """
        智能加载: 如果缓存有效则加载，否则从源数据重建并保存。
        Returns: 当前图谱的节点数
        """
        if self.load_graph():
            return self._graph.number_of_nodes()

        count = self.build_from_configs()
        if count > 0:
            self.save_graph()
        return count
