from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHAT_MODEL, LLM_TEMPERATURE


_llm: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    return _llm
