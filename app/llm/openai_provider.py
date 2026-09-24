import json
from typing import Any, Dict, Optional, Type, TypeVar
import httpx
from pydantic import BaseModel

from app.config import settings
from app.llm.base import BaseLLMProvider
from app.utils.logging import sanitize_secrets

T = TypeVar("T", bound=BaseModel)


class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY or ""
        self.model = model or settings.LLM_MODEL or "gpt-4o-mini"
        self.base_url = "https://api.openai.com/v1"

    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not configured.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        async with httpx.AsyncClient(timeout=settings.LLM_REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str = "",
        temperature: float = 0.1
    ) -> T:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not configured.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use JSON schema response format
        schema_dict = schema.model_json_schema()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema_dict
                }
            }
        }

        async with httpx.AsyncClient(timeout=settings.LLM_REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw_text)
            return schema.model_validate(parsed)
