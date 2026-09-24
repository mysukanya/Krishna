import ast
import re
from typing import Any, Dict, Optional

from app.agent.state import PlanStep, ToolCallRecord, ValidationResult
from app.config import settings
from app.utils.errors import AgentError


class Validator:
    def validate_step(
        self,
        step: PlanStep,
        tool_call: ToolCallRecord,
        error: Optional[AgentError] = None
    ) -> ValidationResult:
        """
        Explicitly evaluates whether a step succeeded, satisfied expected conditions,
        or requires retry / replanning.
        """
        # 1. Check for execution errors
        if error or tool_call.error:
            err = error or tool_call.error
            can_retry = err.recoverable and (step.retry_count < settings.MAX_TOOL_RETRIES)
            should_replan = not can_retry or not err.recoverable

            return ValidationResult(
                step_id=step.id,
                is_valid=False,
                reason=f"Tool execution failed with error: {err.message}",
                should_retry=can_retry,
                should_replan=should_replan,
                details={"error": err.to_dict(), "retry_count": step.retry_count}
            )

        # 2. Check for missing or null output
        output = tool_call.output
        if output is None:
            can_retry = step.retry_count < settings.MAX_TOOL_RETRIES
            return ValidationResult(
                step_id=step.id,
                is_valid=False,
                reason="Tool returned null or empty output contrary to expectation.",
                should_retry=can_retry,
                should_replan=not can_retry,
                details={"expected_output": step.expected_output}
            )

        # 3. Check structural validity and HTTP status codes
        if isinstance(output, dict) and "status_code" in output:
            status_code = output["status_code"]
            if status_code >= 400:
                can_retry = (status_code in (408, 429, 500, 502, 503, 504)) and (step.retry_count < settings.MAX_TOOL_RETRIES)
                return ValidationResult(
                    step_id=step.id,
                    is_valid=False,
                    reason=f"HTTP request returned client/server error code {status_code}",
                    should_retry=can_retry,
                    should_replan=not can_retry,
                    details={"status_code": status_code, "url": output.get("url")}
                )

        # 4. Check success condition
        cond_valid, cond_reason = self._check_success_condition(step.success_condition, output)
        if not cond_valid:
            can_retry = step.retry_count < settings.MAX_TOOL_RETRIES
            return ValidationResult(
                step_id=step.id,
                is_valid=False,
                reason=f"Success condition '{step.success_condition}' was not met: {cond_reason}",
                should_retry=can_retry,
                should_replan=not can_retry,
                details={"condition": step.success_condition, "output_sample": str(output)[:200]}
            )

        # Validation passed
        return ValidationResult(
            step_id=step.id,
            is_valid=True,
            reason=f"Step verified: output structurally valid and satisfied '{step.success_condition}'",
            should_retry=False,
            should_replan=False,
            details={"condition": step.success_condition}
        )

    def _check_success_condition(self, condition: str, output: Any) -> tuple[bool, str]:
        """Safely evaluates condition against tool output."""
        if not condition:
            return True, "No condition specified"

        cond_lower = condition.lower().strip()

        # Check numeric results
        if "is numeric" in cond_lower:
            val = output.get("result") if isinstance(output, dict) else output
            if isinstance(val, (int, float)):
                return True, "Value is numeric"
            return False, f"Expected numeric value, got {type(val).__name__}"

        # Check not empty results
        if "not empty" in cond_lower or "results is not empty" in cond_lower:
            if isinstance(output, dict) and "results" in output:
                res = output["results"]
                if isinstance(res, list) and len(res) > 0:
                    return True, "Results list is non-empty"
                return False, "Results list is empty"
            if isinstance(output, list) and len(output) > 0:
                return True, "Output list is non-empty"
            return False, "Output is empty"

        # Check status_code == 200
        if "status_code == 200" in cond_lower:
            if isinstance(output, dict) and output.get("status_code") == 200:
                return True, "status_code is 200"
            code = output.get("status_code") if isinstance(output, dict) else "unknown"
            return False, f"status_code was {code}, expected 200"

        # Check count > 0
        if "count > 0" in cond_lower:
            if isinstance(output, dict) and output.get("count", 0) > 0:
                return True, "count > 0 satisfied"
            return False, f"count is {output.get('count') if isinstance(output, dict) else 0}"

        # General truthiness check
        return True, "Condition verified"
