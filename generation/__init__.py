"""PowerAgent OS — Generation Layer.

Evidence-first generation pipeline with citation enforcement and audit trail.
"""

from generation.evidence_pipeline import EvidencePipeline, EvidenceCard
from generation.prompt_templates import PromptTemplates, TaskPrompt

__all__ = [
    "EvidencePipeline",
    "EvidenceCard",
    "PromptTemplates",
    "TaskPrompt",
]
