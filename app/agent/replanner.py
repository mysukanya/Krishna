import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agent.state import PlanStep, StepStatus, TaskPlan
from app.llm.base import BaseLLMProvider
from app.tools.registry import registry
from app.utils.errors import AgentError, ErrorType


class Replanner:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    async def replan(
        self,
        goal: str,
        failed_step: PlanStep,
        validation_reason: str,
        completed_steps: List[PlanStep],
        current_plan: TaskPlan
    ) -> TaskPlan:
        """
        Dynamically analyzes step failure and constructs a modified plan.
        """
        tools_summary = "\n".join([
            f"- {t['name']}: {t['description']} (Risk: {t['risk_level']})"
            for t in registry.list_tools()
        ])

        completed_summary = [
            {"id": s.id, "tool": s.tool, "observation": s.observation}
            for s in completed_steps
        ]

        system_prompt = (
            "You are Krishna's dynamic replanning engine.\n"
            "An execution step has failed or failed validation.\n"
            "Inspect the failure details, examine completed progress, and generate an updated, repaired plan.\n"
            "You may substitute tools, adjust arguments, or split steps into safer sub-steps."
        )

        prompt = (
            f"Goal: {goal}\n"
            f"Failed Step ID: {failed_step.id}\n"
            f"Tool Attempted: {failed_step.tool}\n"
            f"Input Parameters: {json.dumps(failed_step.input)}\n"
            f"Failure / Validation Reason: {validation_reason}\n"
            f"Completed Steps: {json.dumps(completed_summary)}\n\n"
            f"Available Tools:\n{tools_summary}\n\n"
            f"Generate a repaired TaskPlan with new or updated remaining steps."
        )

        new_plan = await self.llm.generate_structured(
            prompt=prompt,
            schema=TaskPlan,
            system_prompt=system_prompt,
            temperature=0.1
        )

        new_plan.version = current_plan.version + 1
        return new_plan
