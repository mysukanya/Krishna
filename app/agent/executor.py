import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from app.agent.state import PlanStep, StepStatus, ToolCallRecord
from app.config import settings
from app.tools.registry import RiskLevel, ToolResult, registry
from app.utils.errors import AgentError, ErrorType


class Executor:
    def __init__(self):
        self.registry = registry

    async def execute_step(
        self,
        step: PlanStep,
        approved: bool = False
    ) -> Tuple[ToolCallRecord, str, Optional[AgentError]]:
        """
        Executes a plan step:
        1. Checks tool existence and permissions.
        2. Checks risk level and approval requirement.
        3. Invokes tool with timeout.
        4. Constructs ToolCallRecord and observation summary.
        """
        tool = self.registry.get(step.tool)
        call_id = f"call_{uuid.uuid4().hex[:10]}"

        if not tool:
            err = AgentError(
                type=ErrorType.TOOL_NOT_FOUND,
                tool=step.tool,
                step_id=step.id,
                message=f"Tool '{step.tool}' not found in registry.",
                recoverable=False
            )
            record = ToolCallRecord(
                call_id=call_id,
                step_id=step.id,
                tool_name=step.tool,
                input_args=step.input,
                error=err
            )
            return record, f"Error: Tool '{step.tool}' not found.", err

        # Risk and approval verification
        if tool.risk_level == RiskLevel.HIGH_RISK and settings.REQUIRE_HUMAN_APPROVAL_HIGH_RISK and not approved:
            err = AgentError(
                type=ErrorType.APPROVAL_REJECTED,
                tool=step.tool,
                step_id=step.id,
                message=f"Tool '{step.tool}' requires human approval because it is marked HIGH_RISK.",
                recoverable=True
            )
            record = ToolCallRecord(
                call_id=call_id,
                step_id=step.id,
                tool_name=step.tool,
                input_args=step.input,
                error=err
            )
            return record, "Step awaiting human approval.", err

        start_time = time.time()
        tool_res: ToolResult = await self.registry.execute_tool(
            name=step.tool,
            input_args=step.input,
            timeout_seconds=settings.TOOL_TIMEOUT_SECONDS
        )
        duration_ms = (time.time() - start_time) * 1000.0

        if not tool_res.success:
            err = tool_res.error or AgentError(
                type=ErrorType.TOOL_EXECUTION_ERROR,
                tool=step.tool,
                step_id=step.id,
                message=f"Execution of '{step.tool}' failed without explicit error.",
                recoverable=True
            )
            record = ToolCallRecord(
                call_id=call_id,
                step_id=step.id,
                tool_name=step.tool,
                input_args=step.input,
                error=err,
                duration_ms=duration_ms
            )
            obs = f"Step {step.id} failed: {err.message}"
            return record, obs, err

        # Execution succeeded
        record = ToolCallRecord(
            call_id=call_id,
            step_id=step.id,
            tool_name=step.tool,
            input_args=step.input,
            output=tool_res.data,
            duration_ms=duration_ms
        )

        # Build clean observation string
        data_str = json.dumps(tool_res.data, default=str) if isinstance(tool_res.data, (dict, list)) else str(tool_res.data)
        if len(data_str) > 1000:
            data_str = data_str[:1000] + "... [truncated]"
        obs = f"Step {step.id} ({step.tool}) produced output: {data_str}"

        return record, obs, None
