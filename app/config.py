import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General Project Info
    PROJECT_NAME: str = "Krishna: Autonomous Agentic AI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # LLM Settings
    LLM_PROVIDER: str = Field(default="mock", description="mock | openai | anthropic | gemini")
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1
    LLM_REQUEST_TIMEOUT: float = 30.0

    # Memory / Vector Database
    VECTOR_STORE_TYPE: str = Field(default="chroma", description="chroma | inmemory")
    CHROMA_PERSIST_DIRECTORY: str = "./data/chromadb"
    CHROMA_COLLECTION_NAME: str = "krishna_memory"
    EMBEDDING_PROVIDER: str = Field(default="chroma_default", description="chroma_default | openai | mock")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Execution Bounds & Safety
    MAX_ITERATIONS: int = 10
    MAX_TOOL_RETRIES: int = 3
    TOOL_TIMEOUT_SECONDS: float = 15.0
    REQUIRE_HUMAN_APPROVAL_HIGH_RISK: bool = True
    ALLOWED_HTTP_DOMAINS: List[str] = [
        "api.github.com",
        "httpbin.org",
        "jsonplaceholder.typicode.com",
        "en.wikipedia.org",
        "dummyjson.com"
    ]

    # Host & Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
