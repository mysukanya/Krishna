from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.agent.agent import AutonomousAgent
from app.agent.state import AgentState
from app.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    HealthResponse,
    MemoryCreateRequest,
    MemoryCreateResponse,
    MemorySearchResponse,
)
from app.config import settings
from app.memory.memory_manager import memory_manager
from app.tools.registry import registry

router = APIRouter()

# In-memory execution store for tasks
TASK_CACHE: Dict[str, AgentState] = {}


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """Health check endpoint exposing runtime configuration and tool availability."""
    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        llm_provider=settings.LLM_PROVIDER,
        vector_store=settings.VECTOR_STORE_TYPE,
        tools_count=len(registry.list_tools())
    )


@router.post("/agent/run", response_model=AgentRunResponse, tags=["Agent"])
async def run_agent(request: AgentRunRequest):
    """
    Primary endpoint for goal-driven autonomous execution:
    Executes Goal -> Retrieve -> Plan -> Act -> Observe -> Validate -> Replan -> Synthesize
    """
    agent = AutonomousAgent()
    state = await agent.run(
        goal=request.goal,
        auto_approve_high_risk=request.auto_approve_high_risk
    )
    TASK_CACHE[state.task_id] = state

    steps_dump = [s.model_dump() for s in (state.current_plan.steps if state.current_plan else [])]
    tools_dump = [t.model_dump() for t in state.tool_calls]
    errors_dump = [e.to_dict() for e in state.errors]
    timeline_dump = [evt.model_dump() for evt in state.execution_timeline]

    return AgentRunResponse(
        task_id=state.task_id,
        status=state.status.value,
        result=state.final_result,
        steps=steps_dump,
        tool_calls=tools_dump,
        observations=state.observations,
        iterations=state.iteration_count,
        errors=errors_dump,
        timeline=timeline_dump
    )


@router.get("/agent/{task_id}", response_model=AgentRunResponse, tags=["Agent"])
async def get_agent_task(task_id: str):
    """Retrieves full execution trace, timeline, and state of a previous task."""
    state = TASK_CACHE.get(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    steps_dump = [s.model_dump() for s in (state.current_plan.steps if state.current_plan else [])]
    tools_dump = [t.model_dump() for t in state.tool_calls]
    errors_dump = [e.to_dict() for e in state.errors]
    timeline_dump = [evt.model_dump() for evt in state.execution_timeline]

    return AgentRunResponse(
        task_id=state.task_id,
        status=state.status.value,
        result=state.final_result,
        steps=steps_dump,
        tool_calls=tools_dump,
        observations=state.observations,
        iterations=state.iteration_count,
        errors=errors_dump,
        timeline=timeline_dump
    )


@router.post("/memory", response_model=MemoryCreateResponse, tags=["Memory"])
async def create_memory(request: MemoryCreateRequest):
    """Stores high-value domain knowledge or past results into semantic memory."""
    try:
        doc_id = memory_manager.store(
            text=request.text,
            metadata=request.metadata,
            task_id=request.task_id,
            source=request.source,
            doc_type=request.doc_type
        )
        return MemoryCreateResponse(id=doc_id, status="stored", text=request.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/memory/search", response_model=MemorySearchResponse, tags=["Memory"])
async def search_memory(
    q: str = Query(..., description="Query string for semantic search"),
    limit: int = Query(5, ge=1, le=20, description="Max results")
):
    """Semantic vector search across stored agent knowledge."""
    results = memory_manager.search(query=q, top_k=limit)
    return MemorySearchResponse(
        query=q,
        count=len(results),
        results=results
    )


@router.get("/tools", tags=["Tools"])
async def list_tools():
    """Lists all registered tools, risk permissions, and input/output JSON schemas."""
    return {
        "count": len(registry.list_tools()),
        "tools": registry.list_tools()
    }
