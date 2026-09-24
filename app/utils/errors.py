import time
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    PLANNING_ERROR = "PLANNING_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    HTTP_SECURITY_VIOLATION = "HTTP_SECURITY_VIOLATION"
    PYTHON_SECURITY_VIOLATION = "PYTHON_SECURITY_VIOLATION"


class AgentError(BaseModel):
    type: ErrorType
    tool: Optional[str] = None
    step_id: Optional[str] = None
    message: str
    recoverable: bool = True
    details: Optional[Dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "tool": self.tool,
            "step_id": self.step_id,
            "message": self.message,
            "recoverable": self.recoverable,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class AgentException(Exception):
    def __init__(self, error: AgentError):
        super().__init__(error.message)
        self.error = error
