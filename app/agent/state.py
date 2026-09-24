import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.utils.errors import AgentError


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    AWAITING_APPROVAL = "awaiting_approval"


class AgentStatus(str, Enum):
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    REPLANNING = "replanning"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanStep(BaseModel):
    id: str
    description: str
    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)
    expected_output: str
    success_condition: str
    status: StepStatus = StepStatus.PENDING
    retry_count: int = 0
    actual_output: Optional[Any] = None
    observation: Optional[str] = None
    error: Optional[AgentError] = None
    execution_time_ms: float = 0.0


class TaskPlan(BaseModel):
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)
    version: int = 1
    reasoning: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class ToolCallRecord(BaseModel):
    call_id: str
    step_id: str
    tool_name: str
    input_args: Dict[str, Any]
    output: Optional[Any] = None
    error: Optional[AgentError] = None
    duration_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class ValidationResult(BaseModel):
    step_id: str
    is_valid: bool
    reason: str
    should_retry: bool = False
    should_replan: bool = False
    details: Optional[Dict[str, Any]] = None


class TimelineEvent(BaseModel):
    event_id: str
    event_type: str  # GOAL, PLAN, ACT, OBSERVE, VALIDATE, REPLAN, APPROVAL, COMPLETE, ERROR
    timestamp: float = Field(default_factory=time.time)
    title: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    task_id: str
    user_goal: str
    status: AgentStatus = AgentStatus.INITIALIZING
    current_plan: Optional[TaskPlan] = None
    completed_steps: List[PlanStep] = Field(default_factory=list)
    current_step: Optional[PlanStep] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    errors: List[AgentError] = Field(default_factory=list)
    retrieved_context: List[Dict[str, Any]] = Field(default_factory=list)
    iteration_count: int = 0
    final_result: Optional[str] = None
    execution_timeline: List[TimelineEvent] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def add_event(self, event_type: str, title: str, description: str, metadata: Optional[Dict[str, Any]] = None):
        event = TimelineEvent(
            event_id=f"evt_{len(self.execution_timeline) + 1}",
            event_type=event_type,
            title=title,
            description=description,
            metadata=metadata or {}
        )
        self.execution_timeline.append(event)
        self.updated_at = time.time()
