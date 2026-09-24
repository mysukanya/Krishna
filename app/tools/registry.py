import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, Field

from app.utils.errors import AgentError, ErrorType


class RiskLevel(str, Enum):
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"


class ToolResult(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[AgentError] = None
    execution_time_ms: float = 0.0


class BaseTool(ABC):
    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.LOW_RISK
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """Core execution logic of the tool."""
        pass

    async def execute(self, **kwargs) -> ToolResult:
        """Executes the tool with input validation, timeout, and structured error handling."""
        start_time = time.time()
        try:
            # Validate input using Pydantic schema
            validated_input = self.input_schema(**kwargs)
            result_data = await self.run(**validated_input.model_dump())
            elapsed = (time.time() - start_time) * 1000.0

            return ToolResult(
                success=True,
                data=result_data,
                execution_time_ms=elapsed
            )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000.0
            error = AgentError(
                type=ErrorType.TOOL_EXECUTION_ERROR,
                tool=self.name,
                message=f"Error executing {self.name}: {str(e)}",
                recoverable=True,
                details={"input": kwargs, "exception_class": e.__class__.__name__}
            )
            return ToolResult(
                success=False,
                error=error,
                execution_time_ms=elapsed
            )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "input_schema": self.input_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema()
        }


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [tool.get_metadata() for tool in self._tools.values()]

    def is_allowed(self, name: str) -> bool:
        return name in self._tools

    async def execute_tool(self, name: str, input_args: Dict[str, Any], timeout_seconds: float = 15.0) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                error=AgentError(
                    type=ErrorType.TOOL_NOT_FOUND,
                    tool=name,
                    message=f"Tool '{name}' is not registered in the system registry.",
                    recoverable=False
                )
            )

        try:
            return await asyncio.wait_for(tool.execute(**input_args), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=AgentError(
                    type=ErrorType.TIMEOUT_ERROR,
                    tool=name,
                    message=f"Execution of tool '{name}' timed out after {timeout_seconds} seconds.",
                    recoverable=True
                )
            )


# Global tool registry instance
registry = ToolRegistry()
