from typing import Optional

from app.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.mock_provider import MockRuleBasedLLMProvider
from app.llm.openai_provider import OpenAILLMProvider


def get_llm_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    name = (provider_name or settings.LLM_PROVIDER).lower()

    if name == "openai":
        return OpenAILLMProvider()
    elif name == "mock":
        return MockRuleBasedLLMProvider()
    else:
        # Default fallback to mock for reliable offline and dev execution
        return MockRuleBasedLLMProvider()
