#!/usr/bin/env python3
"""Versioned prompt asset registry used by the demo runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from infrastructure.paths import PROJECT_ROOT, PROMPT_REGISTRY_PATH


class PromptRegistry:
    """Load active Agent, component, scenario and output prompts from YAML."""

    SCENARIO_PROMPTS = {
        "daily_pulse": "daily_pulse_scenario",
        "bidding_advice": "bidding_advice_scenario",
        "policy_search": "policy_search_scenario",
        "long_term_locking": "long_term_locking_scenario",
        "real_time_alert": "real_time_alert_scenario",
    }

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.registry_path = Path(registry_path) if registry_path else PROMPT_REGISTRY_PATH
        self._assets: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        data = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        for metadata in data.get("prompts", []):
            if metadata.get("status") != "active":
                continue
            prompt_id = metadata["prompt_id"]
            path = (PROJECT_ROOT / metadata["file"]).resolve()
            if PROJECT_ROOT.resolve() not in path.parents or not path.is_file():
                raise FileNotFoundError(f"Prompt asset missing or outside project: {metadata['file']}")
            self._assets[prompt_id] = {
                **metadata,
                "path": str(path),
                "content": path.read_text(encoding="utf-8").strip(),
            }

    def get_asset(self, prompt_id: str) -> dict:
        try:
            return dict(self._assets[prompt_id])
        except KeyError as exc:
            raise KeyError(f"Unknown active prompt_id: {prompt_id}") from exc

    def get_prompt(self, task_type: str = "daily_pulse", province: str = "山东") -> str:
        prompt_id = self.SCENARIO_PROMPTS.get(task_type, "supervisor_system")
        content = self.get_asset(prompt_id)["content"]
        return content.replace("{province}", province)

    def validate(self, required_ids: list[str] | None = None) -> dict:
        required = required_ids or list(self._assets)
        missing = [prompt_id for prompt_id in required if prompt_id not in self._assets]
        empty = [prompt_id for prompt_id in required if prompt_id in self._assets and not self._assets[prompt_id]["content"]]
        return {"valid": not missing and not empty, "active_count": len(self._assets), "missing": missing, "empty": empty}


_prompt_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    global _prompt_registry
    if _prompt_registry is None:
        _prompt_registry = PromptRegistry()
    return _prompt_registry
