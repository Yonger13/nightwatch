"""LLM 客户端层：把"模型是哪个供应商"这个细节与主循环隔离。"""

from .client import (
    LLMClient, LLMError, LLMResponse, MockLLM, OpenAICompatLLM, build_llm,
)

__all__ = [
    "LLMClient", "LLMError", "LLMResponse", "MockLLM", "OpenAICompatLLM",
    "build_llm",
]
