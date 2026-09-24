from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    goal: str = Field(..., description="High-level user goal for autonomous execution.", example="Calculate the compound growth of 120000 at 7.5% over 5 years and store the result.")
    auto_approve_high_risk: bool = Field(default=False, description="Automatically approve high-risk tool steps.")


class AgentRunResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    iterations: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryCreateRequest(BaseModel):
    text: str = Field(..., description="Knowledge snippet or fact to persist in semantic memory.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata tags (source, category, timestamp, etc.).")
    task_id: Optional[str] = Field(default=None, description="Associated task ID if applicable.")
    source: str = Field(default="api_ingestion", description="Source identifier.")
    doc_type: str = Field(default="custom_knowledge", description="Document type.")


class MemoryCreateResponse(BaseModel):
    id: str
    status: str = "stored"
    text: str


class MemorySearchResponse(BaseModel):
    query: str
    count: int
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    llm_provider: str
    vector_store: str
    tools_count: int
