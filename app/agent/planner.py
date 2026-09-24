import json
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from app.agent.state import PlanStep, TaskPlan
from app.llm.base import BaseLLMProvider
from app.tools.registry import registry
from app.utils.errors import AgentError, AgentException, ErrorType


class Planner:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    async def create_plan(
        self,
        goal: str,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        previous_results: Optional[List[Dict[str, Any]]] = None,
        previous_errors: Optional[List[Dict[str, Any]]] = None
    ) -> TaskPlan:
        """
        Generates a validated step-by-step TaskPlan using the LLM and tool registry.
        """
        tools_metadata = registry.list_tools()
        tools_summary = "\n".join([
            f"- {t['name']}: {t['description']} (Risk: {t['risk_level']})"
            for t in tools_metadata
        ])

        context_str = json.dumps(retrieved_context or [], indent=2)
        results_str = json.dumps(previous_results or [], indent=2)
        errors_str = json.dumps(previous_errors or [], indent=2)

        system_prompt = (
            "You are Krishna, a senior autonomous AI agent planner.\n"
            "Your objective is to decompose high-level user goals into structured, verifiable execution steps.\n"
            "Rules:\n"
            "1. ONLY select tools that exist in the available tools list.\n"
            "2. Each step must specify an explicit 'expected_output' and 'success_condition'.\n"
            "3. Do NOT execute arbitrary shell scripts or unverified instructions.\n"
            "4. Return a valid TaskPlan structure conforming strictly to the requested schema.\n"
        )

        prompt = (
            f"User Goal: {goal}\n\n"
            f"Available Tools:\n{tools_summary}\n\n"
            f"Retrieved Long-term Context / Memory:\n{context_str}\n\n"
            f"Previous Execution Results (if any):\n{results_str}\n\n"
            f"Previous Errors (if any):\n{errors_str}\n\n"
            "Produce an actionable TaskPlan with steps."
        )

        try:
            plan = await self.llm.generate_structured(
                prompt=prompt,
                schema=TaskPlan,
                system_prompt=system_prompt,
                temperature=0.1
            )
        except ValidationError as ve:
            raise AgentException(AgentError(
                type=ErrorType.PLANNING_ERROR,
                message=f"Planner generated invalid plan structure: {str(ve)}",
                recoverable=True,
                details={"errors": ve.errors()}
            ))
        except AgentException:
            raise
        except Exception as e:
            raise AgentException(AgentError(
                type=ErrorType.PLANNING_ERROR,
                message=f"Planner invocation failed: {str(e)}",
                recoverable=True
            ))

        # Validate that all selected tools exist in registry
        for step in plan.steps:
            if not registry.is_allowed(step.tool):
                raise AgentException(AgentError(
                    type=ErrorType.TOOL_NOT_FOUND,
                    tool=step.tool,
                    step_id=step.id,
                    message=f"Step '{step.id}' selected unknown tool '{step.tool}'",
                    recoverable=True
                ))

        return plan
