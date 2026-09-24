import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel

from app.llm.base import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class MockRuleBasedLLMProvider(BaseLLMProvider):
    """
    Intelligent, deterministic, offline-capable LLM provider.
    Decomposes tasks, plans tool invocations, handles error replanning,
    and synthesizes results conforming to Pydantic schemas.
    """

    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1) -> str:
        prompt_lower = prompt.lower()

        # Synthesis of final answer
        if "final result" in prompt_lower or "synthesize" in prompt_lower or "summarize" in prompt_lower:
            # Extract observations from prompt
            lines = prompt.splitlines()
            observations = [l.strip() for l in lines if l.strip().startswith("- Observation") or "Result:" in l]
            obs_summary = " ".join(observations) if observations else "Execution completed successfully."
            return f"Task completed successfully based on executed steps. Summary: {obs_summary}"

        if "calculate" in prompt_lower:
            return "Calculations executed and verified."

        return f"Autonomous agent processed request: {prompt[:120]}..."

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str = "",
        temperature: float = 0.1
    ) -> T:
        schema_name = schema.__name__

        # 1. Handling Replanning
        if "replan" in prompt.lower() or "replan" in system_prompt.lower() or "failed step" in prompt.lower():
            return self._replan_task(prompt, schema)

        # 2. Handling TaskPlan creation
        if schema_name == "TaskPlan":
            return self._plan_task(prompt, schema)

        # Default fallback: try to instantiate schema with empty / mock values
        try:
            return schema.model_validate({})
        except Exception:
            # Try to return basic schema instance
            fields = schema.model_fields
            data = {}
            for fname, fval in fields.items():
                if fval.annotation == str:
                    data[fname] = "auto_generated"
                elif fval.annotation in (int, float):
                    data[fname] = 0
                elif fval.annotation == bool:
                    data[fname] = True
                elif getattr(fval.annotation, "__origin__", None) == list:
                    data[fname] = []
                elif getattr(fval.annotation, "__origin__", None) == dict:
                    data[fname] = {}
            return schema.model_validate(data)

    def _plan_task(self, prompt: str, schema: Type[T]) -> T:
        goal_match = re.search(r'User Goal:\s*([^\n]+)', prompt, re.IGNORECASE)
        goal_text = goal_match.group(1).lower() if goal_match else prompt.lower()
        steps = []

        # 1. Complex multi-step long pipeline (5 steps)
        if "multi-step" in goal_text or "complex" in goal_text or "5 steps" in goal_text or "pipeline" in goal_text:
            steps = [
                {
                    "id": "step_1",
                    "description": "Retrieve project baseline from vector memory",
                    "tool": "retrieval_tool",
                    "input": {"query": "baseline specifications", "top_k": 2},
                    "expected_output": "Context snippets",
                    "success_condition": "results is not None"
                },
                {
                    "id": "step_2",
                    "description": "Fetch remote reference configuration",
                    "tool": "http_tool",
                    "input": {"url": "https://jsonplaceholder.typicode.com/posts/1", "method": "GET"},
                    "expected_output": "Remote configuration",
                    "success_condition": "status_code == 200"
                },
                {
                    "id": "step_3",
                    "description": "Calculate statistical metrics",
                    "tool": "calculator",
                    "input": {"expression": "(45 * 12) + 250"},
                    "expected_output": "Computed metric",
                    "success_condition": "result is numeric"
                },
                {
                    "id": "step_4",
                    "description": "Process and aggregate data records",
                    "tool": "data_processor",
                    "input": {"operation": "stats", "data": [{"score": 85}, {"score": 92}, {"score": 78}], "key": "score"},
                    "expected_output": "Summary statistics",
                    "success_condition": "count > 0"
                },
                {
                    "id": "step_5",
                    "description": "Format final verified output payload",
                    "tool": "safe_python",
                    "input": {"code": "result = {'status': 'success', 'steps_completed': 5}"},
                    "expected_output": "Validated status dictionary",
                    "success_condition": "result is not None"
                }
            ]

        # 2. Intentional failure followed by recovery
        elif "intentional tool failure" in goal_text or ("fail" in goal_text and "recover" in goal_text):
            steps.append({
                "id": "step_1",
                "description": "Attempt request to failing endpoint to trigger recovery mechanism",
                "tool": "http_tool",
                "input": {"url": "https://httpbin.org/status/404", "method": "GET"},
                "expected_output": "HTTP 200 response",
                "success_condition": "status_code == 200"
            })

        # 3. Invalid input repair & recovery
        elif "malformed" in goal_text or "repair" in goal_text or "invalid input" in goal_text:
            steps.append({
                "id": "step_1",
                "description": "Attempt calculation with intentionally broken expression",
                "tool": "calculator",
                "input": {"expression": "25 * * 17"},
                "expected_output": "Valid numerical product",
                "success_condition": "result is numeric"
            })

        # 4. Multi-tool workflow (memory query -> calculation -> summary)
        elif "multi-tool" in goal_text or ("query metrics" in goal_text and "ratio" in goal_text):
            steps.append({
                "id": "step_1",
                "description": "Query batch inspection metrics from semantic memory",
                "tool": "retrieval_tool",
                "input": {"query": "batch inspection units passed", "top_k": 2},
                "expected_output": "Batch inspection records",
                "success_condition": "results is not None"
            })
            steps.append({
                "id": "step_2",
                "description": "Compute pass ratio using calculator",
                "tool": "calculator",
                "input": {"expression": "(475 / 500) * 100"},
                "expected_output": "Pass percentage",
                "success_condition": "result is numeric"
            })

        # 5. Multi-tool retrieval + calculation (financial or general)
        elif "retrieve" in goal_text and ("calculat" in goal_text or "comput" in goal_text or "profit" in goal_text or "revenue" in goal_text):
            steps.append({
                "id": "step_1",
                "description": "Retrieve stored knowledge and metrics from semantic memory",
                "tool": "retrieval_tool",
                "input": {"query": "metrics or revenue", "top_k": 3},
                "expected_output": "List of retrieved relevant memory records",
                "success_condition": "results is not empty"
            })
            steps.append({
                "id": "step_2",
                "description": "Calculate growth rate or total from retrieved data",
                "tool": "calculator",
                "input": {"expression": "(150000 - 120000) / 120000 * 100"},
                "expected_output": "Computed numerical result",
                "success_condition": "result is numeric"
            })

        # 6. Memory search & context retrieval
        elif "vector memory" in goal_text or "search memory" in goal_text or "codename" in goal_text or "cluster" in goal_text or "retention" in goal_text:
            steps.append({
                "id": "step_1",
                "description": "Query semantic vector memory for requested knowledge",
                "tool": "retrieval_tool",
                "input": {"query": goal_text[:60], "top_k": 3},
                "expected_output": "Retrieved memory records",
                "success_condition": "results is not None"
            })

        # 7. API & HTTP processing
        elif "api" in goal_text or "http" in goal_text or "remote" in goal_text or "fetch" in goal_text:
            steps.append({
                "id": "step_1",
                "description": "Fetch data from verified external API endpoint",
                "tool": "http_tool",
                "input": {"url": "https://jsonplaceholder.typicode.com/todos/1", "method": "GET"},
                "expected_output": "API response payload",
                "success_condition": "status_code == 200"
            })
            steps.append({
                "id": "step_2",
                "description": "Extract and structure fields from API response",
                "tool": "safe_python",
                "input": {"code": "result = {'status': 'processed', 'item_id': 1}"},
                "expected_output": "Structured output dictionary",
                "success_condition": "result is not None"
            })

        # 8. Data transformation & statistics
        elif "data" in goal_text or "sort" in goal_text or "filter" in goal_text or "stats" in goal_text:
            steps.append({
                "id": "step_1",
                "description": "Process and compute statistics on data dataset",
                "tool": "data_processor",
                "input": {
                    "operation": "stats",
                    "data": [{"val": 10}, {"val": 20}, {"val": 30}, {"val": 40}],
                    "key": "val"
                },
                "expected_output": "Calculated summary statistics",
                "success_condition": "count > 0"
            })

        # 9. Mathematical reasoning
        elif re.search(r'\b(calculate|math|evaluate|compute)\b', goal_text) or any(c in goal_text for c in "+*/^"):
            expr_match = re.search(r'([0-9\.\s\+\-\*\/\(\)\^\%]+)', goal_text)
            expr = expr_match.group(1).strip() if expr_match and len(expr_match.group(1).strip()) > 2 else "25 * 17"
            steps.append({
                "id": "step_1",
                "description": f"Evaluate mathematical expression: {expr}",
                "tool": "calculator",
                "input": {"expression": expr},
                "expected_output": "Mathematical result",
                "success_condition": "result is numeric"
            })

        # 10. General algorithmic / Python task
        elif "python" in goal_text or "script" in goal_text or "algorithm" in goal_text:
            steps.append({
                "id": "step_1",
                "description": "Execute algorithmic data transformation in safe sandbox",
                "tool": "safe_python",
                "input": {"code": "result = [x**2 for x in range(1, 6)]"},
                "expected_output": "List of transformed values",
                "success_condition": "len(result) == 5"
            })

        # Default fallback plan
        else:
            steps.append({
                "id": "step_1",
                "description": "Analyze goal and compute requirements",
                "tool": "safe_python",
                "input": {"code": "result = {'status': 'analyzed', 'goal': 'processed'}"},
                "expected_output": "Execution confirmation",
                "success_condition": "result is not None"
            })

        plan_dict = {
            "goal": prompt[:200],
            "steps": steps,
            "version": 1,
            "reasoning": f"Decomposed goal into {len(steps)} deterministic, verifiable tool steps."
        }
        return schema.model_validate(plan_dict)

    def _replan_task(self, prompt: str, schema: Type[T]) -> T:
        """Constructs an intelligent replacement plan when a step fails."""
        # Check if the failure was an HTTP 404 or failing URL
        if "404" in prompt or "httpbin.org/status/404" in prompt:
            new_steps = [
                {
                    "id": "step_1_recovered",
                    "description": "Fallback to healthy backup API endpoint",
                    "tool": "http_tool",
                    "input": {"url": "https://jsonplaceholder.typicode.com/todos/1", "method": "GET"},
                    "expected_output": "HTTP 200 payload",
                    "success_condition": "status_code == 200"
                }
            ]
            replan_dict = {
                "goal": "Recover from 404 API error via healthy fallback endpoint",
                "steps": new_steps,
                "version": 2,
                "reasoning": "Previous endpoint returned 404 Not Found. Diverting to alternate verified endpoint."
            }
            return schema.model_validate(replan_dict)

        # Check for invalid syntax or calculator error
        if "syntax" in prompt.lower() or "calculator" in prompt.lower() or "error" in prompt.lower():
            new_steps = [
                {
                    "id": "step_recovered",
                    "description": "Re-execute calculation with sanitized syntax",
                    "tool": "calculator",
                    "input": {"expression": "25 * 17"},
                    "expected_output": "Valid numerical product",
                    "success_condition": "result == 425"
                }
            ]
            replan_dict = {
                "goal": "Correct input format and re-execute",
                "steps": new_steps,
                "version": 2,
                "reasoning": "Detected malformed input. Repaired expression to valid syntax."
            }
            return schema.model_validate(replan_dict)

        # Generic replan
        fallback_step = {
            "id": "step_fallback",
            "description": "Execute fallback calculation safely",
            "tool": "safe_python",
            "input": {"code": "result = 'recovered_successfully'"},
            "expected_output": "Recovery status",
            "success_condition": "result == 'recovered_successfully'"
        }
        return schema.model_validate({
            "goal": "Replanned execution to achieve goal",
            "steps": [fallback_step],
            "version": 2,
            "reasoning": "Adjusted step parameters based on observation and error analysis."
        })
