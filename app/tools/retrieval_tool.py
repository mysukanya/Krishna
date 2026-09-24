from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.memory.memory_manager import memory_manager
from app.tools.registry import BaseTool, RiskLevel


class RetrievalInput(BaseModel):
    query: str = Field(..., description="Semantic search query to retrieve stored facts, past results, or domain context.")
    top_k: int = Field(default=3, description="Maximum number of items to retrieve.")


class RetrievalOutput(BaseModel):
    query: str
    count: int
    results: List[Dict[str, Any]]


class RetrievalTool(BaseTool):
    name: str = "retrieval_tool"
    description: str = "Searches the vector database for relevant past task results, background documentation, and semantic memory."
    risk_level: RiskLevel = RiskLevel.LOW_RISK
    input_schema = RetrievalInput
    output_schema = RetrievalOutput

    async def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        records = memory_manager.search(query=query, top_k=top_k)
        formatted = []
        for r in records:
            formatted.append({
                "id": r["id"],
                "text": r["text"],
                "score": round(r["score"], 4),
                "metadata": r.get("metadata", {})
            })

        return {
            "query": query,
            "count": len(formatted),
            "results": formatted
        }
