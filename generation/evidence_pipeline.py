"""Evidence-first generation pipeline.

Enforces a strict three-phase generation protocol:
  1. Evidence Extraction  — extract claims with citations from retrieved context
  2. Answer Generation    — produce answer with inline citation markers
  3. Action Recommendation — optional actionable next steps

All output flows through an audit trail and is verified post-generation.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from generation.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EvidenceCard:
    """A single piece of evidence extracted from retrieved context."""

    claim: str
    source_id: str
    source_name: str
    relevance: str  # "high" | "medium" | "low"
    confidence: float  # 0.0 - 1.0
    raw_quote: str = ""


@dataclass
class GenerationResult:
    """Complete generation output with evidence and audit trail."""

    request_id: str
    query: str
    task_type: str
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    answer: str = ""
    recommendations: list[dict] = field(default_factory=list)
    citations_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generation_time_ms: float = 0.0
    token_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Evidence Pipeline
# ---------------------------------------------------------------------------


class EvidencePipeline:
    """Orchestrates evidence-first generation with LLM enforcement.

    The pipeline:
      1. Extracts structured evidence cards from retrieved context
      2. Generates the final answer with mandatory citation markers
      3. Extracts actionable recommendations
      4. Validates citation integrity
    """

    def __init__(
        self,
        llm=None,
        prompts: PromptTemplates | None = None,
        audit_logger=None,
    ):
        self._llm = llm
        self._prompts = prompts or PromptTemplates()
        self._audit = audit_logger

    def generate(
        self,
        query: str,
        context: str,
        task_type: str = "general",
        province: str = "",
        prediction: str = "",
        tenant_context: str = "",
    ) -> GenerationResult:
        """Run the full evidence-first generation pipeline."""
        request_id = str(uuid.uuid4())[:12]
        t0 = time.perf_counter()

        result = GenerationResult(
            request_id=request_id,
            query=query,
            task_type=task_type,
        )

        # Step 1: Extract evidence cards from context
        result.evidence_cards = self._extract_evidence(query, context)

        # Step 2: Build full prompt and generate
        full_prompt = self._prompts.build_full_prompt(
            task_type=task_type,
            query=query,
            context=context,
            province=province,
            prediction=prediction,
            tenant_context=tenant_context,
        )

        answer_text = self._generate_text(full_prompt)

        # Step 3: Parse answer into sections
        result.answer, result.recommendations = self._parse_answer(answer_text)

        # Step 4: Validate citation integrity
        result.warnings = self._validate_citations(result.answer, result.evidence_cards)
        result.citations_used = self._extract_citation_ids(result.answer)

        result.generation_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Write audit trail
        if self._audit:
            try:
                self._audit.log_agent_step(
                    step="evidence_generation",
                    detail={
                        "request_id": request_id,
                        "evidence_count": len(result.evidence_cards),
                        "citation_count": len(result.citations_used),
                        "warnings": result.warnings,
                        "generation_time_ms": result.generation_time_ms,
                    },
                )
            except Exception:
                pass

        logger.info(
            "EvidencePipeline: generated %d evidence cards, %d citations, %d warnings in %.0fms",
            len(result.evidence_cards),
            len(result.citations_used),
            len(result.warnings),
            result.generation_time_ms,
        )
        return result

    # ----- evidence extraction -----------------------------------------------

    def _extract_evidence(
        self, query: str, context: str
    ) -> list[EvidenceCard]:
        """Extract structured evidence cards from the retrieved context.

        Uses the LLM to parse the context into claim-source pairs. Falls back
        to a regex-based extraction when no LLM is available.
        """
        if self._llm is None:
            return self._fallback_extract(context)

        extract_prompt = f"""从以下检索到的上下文中提取关键证据。对于每条证据，提供：

1. 具体的事实声明（claim）
2. 来源编号（从上下文中的 [来源:X] 标记提取）
3. 来源名称
4. 相关性（high/medium/low）
5. 置信度（0.0-1.0）

只提取与查询直接相关的证据。输出 JSON 数组。

查询: {query}

上下文:
{context[:8000]}

输出格式:
[
  {{
    "claim": "事实声明",
    "source_id": "来源编号",
    "source_name": "来源名称",
    "relevance": "high",
    "confidence": 0.95,
    "raw_quote": "原文引用"
  }}
]

只返回 JSON 数组，不要其他内容。"""

        try:
            resp = self._llm.invoke(extract_prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            data = self._safe_json_parse(text)
            if isinstance(data, list):
                return [
                    EvidenceCard(
                        claim=item.get("claim", ""),
                        source_id=item.get("source_id", ""),
                        source_name=item.get("source_name", ""),
                        relevance=item.get("relevance", "medium"),
                        confidence=float(item.get("confidence", 0.5)),
                        raw_quote=item.get("raw_quote", ""),
                    )
                    for item in data[:20]
                ]
        except Exception as exc:
            logger.warning("Evidence extraction via LLM failed: %s", exc)

        return self._fallback_extract(context)

    def _fallback_extract(self, context: str) -> list[EvidenceCard]:
        """Regex-based fallback evidence extraction."""
        cards: list[EvidenceCard] = []
        # Extract source blocks like [政策 | 山东省发改委] or [来源:X]
        source_pattern = re.compile(r"\[(?:来源|source)[：:\s]*(\d+|[^\]]+)\]", re.IGNORECASE)
        # Extract sentences with numbers (likely factual claims)
        claim_pattern = re.compile(
            r"([^。\n]{10,80}?(?:\d+\.?\d*\s*(?:元/MWh|%|MW|MWh|元|小时|天))[^。\n]{0,40})"
        )

        sources = source_pattern.findall(context)
        claims = claim_pattern.findall(context)

        for i, claim in enumerate(claims[:15]):
            source_id = sources[i] if i < len(sources) else "context"
            cards.append(
                EvidenceCard(
                    claim=claim.strip(),
                    source_id=str(source_id),
                    source_name=str(source_id),
                    relevance="medium",
                    confidence=0.6,
                    raw_quote=claim.strip(),
                )
            )
        return cards

    # ----- text generation ---------------------------------------------------

    def _generate_text(self, full_prompt: str) -> str:
        """Generate text via LLM.

        Fault handling belongs to the unified LangGraph degradation policy.
        This component reports failure instead of silently returning a fixed
        template that could be mistaken for a valid business answer.
        """
        if self._llm is None:
            raise RuntimeError("LLM生成能力未配置")

        try:
            resp = self._llm.invoke(full_prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise RuntimeError(f"LLM生成失败: {exc}") from exc

    # ----- answer parsing ----------------------------------------------------

    def _parse_answer(self, text: str) -> tuple[str, list[dict]]:
        """Parse the three-part answer structure into sections."""
        # Split on section headers
        sections = re.split(r"###\s*[123]\.", text)

        answer = text  # full text as default
        recommendations: list[dict] = []

        if len(sections) >= 3:
            # sections[0] = preamble, [1] = evidence, [2] = answer, [3] = recommendations
            answer = sections[2].strip() if len(sections) > 2 else text
            if len(sections) > 3:
                rec_text = sections[3].strip()
                recommendations = self._parse_recommendations(rec_text)

        return answer, recommendations

    @staticmethod
    def _parse_recommendations(text: str) -> list[dict]:
        """Parse recommendation lines into structured dicts."""
        recs = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("-"):
                continue
            # Parse: "- [风险等级] 建议内容"
            match = re.match(r"-\s*\[([^\]]+)\]\s*(.+)", line)
            if match:
                recs.append({
                    "risk_level": match.group(1).strip(),
                    "recommendation": match.group(2).strip(),
                })
            else:
                recs.append({
                    "risk_level": "未标注",
                    "recommendation": line.lstrip("- "),
                })
        return recs

    # ----- citation validation -----------------------------------------------

    def _validate_citations(
        self, answer: str, evidence_cards: list[EvidenceCard]
    ) -> list[str]:
        """Check that all citation references in the answer map to valid evidence cards."""
        warnings: list[str] = []

        # Find all [来源:X] or [引用:X] markers
        cite_markers = re.findall(r"\[(?:来源|引用|source)[：:\s]*(\d+|[^\]]+)\]", answer)
        valid_ids = {card.source_id for card in evidence_cards}

        for marker in cite_markers:
            if marker not in valid_ids and marker not in ("推断", "待验证", "inferred", "unverified"):
                warnings.append(f"引用 [{marker}] 在证据卡片中未找到对应来源")

        # Check for unsubstantiated claims (no citation after factual statements)
        factual_sentences = re.findall(
            r"([^。\n]{15,}?(?:\d+\.?\d*\s*(?:元/MWh|%|MW|MWh|元))[^。\n]*)",
            answer,
        )
        for sent in factual_sentences:
            if not re.search(r"\[(?:来源|引用|source)", sent, re.IGNORECASE):
                warnings.append(f"未引用声明: {sent[:60]}...")

        return warnings[:10]

    @staticmethod
    def _extract_citation_ids(answer: str) -> list[str]:
        """Extract all unique citation IDs from the answer."""
        markers = re.findall(r"\[(?:来源|引用|source)[：:\s]*(\d+|[^\]]+)\]", answer, re.IGNORECASE)
        return list(dict.fromkeys(markers))

    # ----- utilities ---------------------------------------------------------

    @staticmethod
    def _safe_json_parse(text: str):
        """Parse JSON from LLM output, handling markdown code fences."""
        # Strip markdown code blocks
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.strip()
        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON array
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return []
