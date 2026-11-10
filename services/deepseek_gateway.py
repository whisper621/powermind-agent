"""DeepSeek LLM Gateway implementation.

Reads API key from ONE place.  Every agent goes through this gateway.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek

from services.llm_gateway import LLMCallResult, LLMGateway

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeepSeekGateway:
    """LLM Gateway backed by DeepSeek Chat API.

    The gateway owns the API key lifecycle.  Agents never touch it.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.1,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or self._resolve_api_key()
        self._llm: Optional[ChatDeepSeek] = None

    # ---- LLMGateway impl --------------------------------------------------

    def invoke(
        self,
        *,
        agent_name: str,
        messages: list,
        tools: list | None = None,
        task_context: dict | None = None,
    ) -> LLMCallResult:
        t0 = time.perf_counter()
        try:
            llm = self._get_llm(temperature=self._temperature)
            if tools:
                llm = llm.bind_tools(tools)
            resp = llm.invoke(messages)
            content = resp.content if hasattr(resp, "content") else str(resp)
            tool_calls = []
            if hasattr(resp, "tool_calls") and resp.tool_calls:
                tool_calls = list(resp.tool_calls)
            return LLMCallResult(
                content=content,
                model=self._model,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                tool_calls=tool_calls,
            )
        except Exception as exc:
            return LLMCallResult(
                content="",
                model=self._model,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
            )

    def invoke_with_tools(
        self,
        *,
        agent_name: str,
        messages: list,
        tools: list,
        task_context: dict | None = None,
    ) -> LLMCallResult:
        return self.invoke(
            agent_name=agent_name,
            messages=messages,
            tools=tools,
            task_context=task_context,
        )

    def health_check(self) -> bool:
        return bool(self._api_key)

    # ---- internals --------------------------------------------------------

    def _get_llm(self, temperature: float | None = None) -> ChatDeepSeek:
        if self._llm is None:
            self._llm = ChatDeepSeek(
                model=self._model,
                temperature=temperature if temperature is not None else self._temperature,
                api_key=self._api_key,
            )
        return self._llm

    @staticmethod
    def _resolve_api_key() -> str:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if key:
            return key
        env_path = PROJECT_ROOT / "config" / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        raise RuntimeError("DEEPSEEK_API_KEY 未配置 — 请设置环境变量或 config/.env")
