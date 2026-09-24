import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from app.agent.executor import Executor
from app.agent.planner import Planner
from app.agent.replanner import Replanner
from app.agent.state import (
    AgentState,
    AgentStatus,
    PlanStep,
    StepStatus,
    TaskPlan,
    ValidationResult,
)
from app.agent.validator import Validator
from app.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.provider import get_llm_provider
from app.memory.memory_manager import memory_manager
from app.utils.errors import AgentError, AgentException, ErrorType
from app.utils.logging import AgentLogger


class AutonomousAgent:
    """
    Krishna Autonomous Agent:
    Implements the complete goal-driven execution loop:
    Goal -> Retrieve Memory -> Plan -> Tool -> Observe -> Validate -> Replan -> Synthesize -> Store Memory
    """

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()
        self.planner = Planner(self.llm)
        self.executor = Executor()
        self.validator = Validator()
        self.replanner = Replanner(self.llm)
        self.memory = memory_manager

    async def run(
        self,
        goal: str,
        task_id: Optional[str] = None,
        auto_approve_high_risk: bool = False
    ) -> AgentState:
        """
        Executes the autonomous agent loop until the goal is achieved,
        or max iterations / non-recoverable error is reached.
        """
        tid = task_id or f"task_{uuid.uuid4().hex[:10]}"
        state = AgentState(task_id=tid, user_goal=goal)
        logger = AgentLogger(tid)

        start_time = time.time()
        logger.log_goal_received(goal)
        state.add_event("GOAL", "User Goal Received", f"Initiated execution for goal: {goal}")

        # 1. Retrieve Relevant Long-Term Memory
        try:
            retrieved = self.memory.retrieve_context(query=goal, top_k=3)
            state.retrieved_context = retrieved
            if retrieved:
                state.add_event(
                    "MEMORY",
                    "Context Retrieved",
                    f"Retrieved {len(retrieved)} relevant knowledge item(s) from vector memory.",
                    {"context_count": len(retrieved)}
                )
        except Exception as e:
            state.errors.append(AgentError(
                type=ErrorType.MEMORY_ERROR,
                message=f"Failed to query semantic memory: {str(e)}",
                recoverable=True
            ))

        # 2. Planning Phase
        state.status = AgentStatus.PLANNING
        try:
            plan = await self.planner.create_plan(
                goal=goal,
                retrieved_context=state.retrieved_context
            )
            state.current_plan = plan
            logger.log_planner_decision(plan.version, len(plan.steps), plan.reasoning)
            state.add_event(
                "PLAN",
                f"Task Plan Generated (v{plan.version})",
                f"Decomposed goal into {len(plan.steps)} executable step(s).",
                {"steps": [s.model_dump() for s in plan.steps]}
            )
        except AgentException as ae:
            state.status = AgentStatus.FAILED
            state.errors.append(ae.error)
            state.final_result = f"Planning failed: {ae.error.message}"
            return state
        except AgentError as ae:
            state.status = AgentStatus.FAILED
            state.errors.append(ae)
            state.final_result = f"Planning failed: {ae.message}"
            return state

        # 3. Execution, Observation, Validation, and Replanning Loop
        state.status = AgentStatus.EXECUTING
        current_step_index = 0

        while current_step_index < len(state.current_plan.steps):
            state.iteration_count += 1

            # Guard against infinite loops
            if state.iteration_count > settings.MAX_ITERATIONS:
                err = AgentError(
                    type=ErrorType.MAX_ITERATIONS_REACHED,
                    message=f"Maximum allowed iterations ({settings.MAX_ITERATIONS}) reached.",
                    recoverable=False
                )
                state.errors.append(err)
                state.status = AgentStatus.FAILED
                state.final_result = "Execution halted: Exceeded maximum iteration limit."
                break

            step = state.current_plan.steps[current_step_index]
            state.current_step = step
            step.status = StepStatus.RUNNING

            logger.log_tool_selected(step.id, step.tool, step.input)
            state.add_event(
                "ACT",
                f"Executing Step {step.id} ({step.tool})",
                f"Description: {step.description}",
                {"step_id": step.id, "tool": step.tool, "input": step.input}
            )

            # Execute tool
            tool_record, observation, exec_error = await self.executor.execute_step(
                step=step,
                approved=auto_approve_high_risk
            )
            state.tool_calls.append(tool_record)
            state.observations.append(observation)
            step.actual_output = tool_record.output
            step.observation = observation
            step.execution_time_ms = tool_record.duration_ms

            if exec_error:
                step.error = exec_error
                state.errors.append(exec_error)

            logger.log_tool_executed(step.id, step.tool, tool_record.duration_ms, observation)
            state.add_event(
                "OBSERVE",
                f"Observed Output from Step {step.id}",
                observation[:200] + ("..." if len(observation) > 200 else ""),
                {"step_id": step.id, "tool": step.tool, "duration_ms": tool_record.duration_ms}
            )

            # Validation
            state.status = AgentStatus.VALIDATING
            val_res: ValidationResult = self.validator.validate_step(
                step=step,
                tool_call=tool_record,
                error=exec_error
            )

            logger.log_validation(
                step.id,
                val_res.is_valid,
                val_res.reason,
                val_res.should_retry,
                val_res.should_replan
            )

            state.add_event(
                "VALIDATE",
                f"Validation for Step {step.id}: {'PASSED' if val_res.is_valid else 'FAILED'}",
                val_res.reason,
                {"step_id": step.id, "is_valid": val_res.is_valid, "should_replan": val_res.should_replan}
            )

            # Handle Validation Result
            if val_res.is_valid:
                step.status = StepStatus.COMPLETED
                state.completed_steps.append(step)
                current_step_index += 1
                state.status = AgentStatus.EXECUTING
            else:
                # Failure branch
                if val_res.should_retry:
                    step.retry_count += 1
                    state.add_event(
                        "RETRY",
                        f"Retrying Step {step.id}",
                        f"Attempt {step.retry_count} of {settings.MAX_TOOL_RETRIES}"
                    )
                    # Loop re-executes same step
                    continue

                elif val_res.should_replan:
                    step.status = StepStatus.FAILED
                    state.status = AgentStatus.REPLANNING
                    logger.log_replan(step.id, val_res.reason, state.current_plan.version + 1)
                    state.add_event(
                        "REPLAN",
                        f"Replanning Triggered at Step {step.id}",
                        f"Reason: {val_res.reason}"
                    )

                    try:
                        repaired_plan = await self.replanner.replan(
                            goal=goal,
                            failed_step=step,
                            validation_reason=val_res.reason,
                            completed_steps=state.completed_steps,
                            current_plan=state.current_plan
                        )
                        state.current_plan = repaired_plan
                        current_step_index = 0  # Start executing remaining repaired steps
                        state.status = AgentStatus.EXECUTING
                        continue
                    except Exception as re_err:
                        err = AgentError(
                            type=ErrorType.PLANNING_ERROR,
                            message=f"Dynamic replanning failed: {str(re_err)}",
                            recoverable=False
                        )
                        state.errors.append(err)
                        state.status = AgentStatus.FAILED
                        state.final_result = f"Execution failed during replanning: {err.message}"
                        break
                else:
                    # Non-recoverable failure without replan
                    step.status = StepStatus.FAILED
                    state.status = AgentStatus.FAILED
                    state.final_result = f"Step {step.id} failed without recovery option: {val_res.reason}"
                    break

        # 4. Final Result Generation & Synthesis
        if state.status != AgentStatus.FAILED:
            state.status = AgentStatus.COMPLETED
            synth_prompt = (
                f"Goal: {state.user_goal}\n"
                f"Executed steps and observations:\n"
                + "\n".join([f"- {s.id} ({s.tool}): {s.observation}" for s in state.completed_steps])
                + "\nSynthesize a clear, accurate, and comprehensive final response for the user."
            )
            final_ans = await self.llm.generate(
                prompt=synth_prompt,
                system_prompt="Synthesize the autonomous execution observations into a definitive answer."
            )
            state.final_result = final_ans

            # 5. Store Valuable Result in Long-Term Semantic Memory
            try:
                self.memory.store(
                    text=f"Goal: {goal}\nResult: {final_ans}",
                    metadata={"goal": goal[:100], "iterations": state.iteration_count},
                    task_id=tid,
                    source="autonomous_execution",
                    doc_type="task_outcome"
                )
            except Exception:
                pass

        total_duration = time.time() - start_time
        logger.log_completion(state.status.value, state.iteration_count, total_duration)
        state.add_event(
            "COMPLETE",
            f"Execution Finished: {state.status.value.upper()}",
            f"Completed in {total_duration:.2f}s across {state.iteration_count} iterations.",
            {"status": state.status.value, "duration_seconds": total_duration}
        )

        return state
