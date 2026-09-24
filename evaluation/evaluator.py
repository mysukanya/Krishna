import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agent.agent import AutonomousAgent
from app.agent.state import AgentStatus
from app.llm.provider import get_llm_provider
from app.memory.memory_manager import memory_manager
from evaluation.benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask


class TaskExecutionMetric(BaseModel):
    task_id: str
    task_name: str
    mode: str  # "MODE_A_SINGLE_PASS" or "MODE_B_AGENTIC"
    success: bool
    iterations: int
    tool_calls_count: int
    tool_accuracy: float
    recovered: bool
    memory_retrieved: bool
    duration_seconds: float
    output_snippet: str


class BenchmarkSummary(BaseModel):
    mode: str
    total_tasks: int
    completed_count: int
    completion_rate_pct: float
    tool_use_accuracy_pct: float
    recovery_rate_pct: float
    average_iterations: float
    average_tool_calls: float
    average_duration_seconds: float
    failure_rate_pct: float


class MemoryComparisonSummary(BaseModel):
    total_memory_tasks: int
    without_memory_success_count: int
    without_memory_accuracy_pct: float
    with_memory_success_count: int
    with_memory_accuracy_pct: float
    context_retention_improvement_pct: float


class BenchmarkEvaluator:
    def __init__(self):
        self.llm = get_llm_provider()
        self.memory = memory_manager

    async def run_mode_a_single_pass(self, task: BenchmarkTask) -> TaskExecutionMetric:
        """
        Baseline Mode A: Single-pass LLM call without tool execution,
        validation, or dynamic replanning.
        """
        start = time.time()
        prompt = f"Goal: {task.goal}\nPlease execute this goal and provide the final result directly."
        response_text = await self.llm.generate(prompt=prompt)
        elapsed = time.time() - start

        # Check if response matches keywords
        text_lower = response_text.lower()
        matched = any(kw.lower() in text_lower for kw in task.success_keywords)

        # In single pass, tools are never called and recovery never happens
        return TaskExecutionMetric(
            task_id=task.id,
            task_name=task.name,
            mode="MODE_A_SINGLE_PASS",
            success=matched and not task.requires_memory,
            iterations=1,
            tool_calls_count=0,
            tool_accuracy=0.0,
            recovered=False,
            memory_retrieved=False,
            duration_seconds=elapsed,
            output_snippet=response_text[:150]
        )

    async def run_mode_b_agentic(self, task: BenchmarkTask) -> TaskExecutionMetric:
        """
        Mode B: Full autonomous agent loop:
        Goal -> Memory -> Plan -> Tool -> Observe -> Validate -> Replan -> Synthesize
        """
        # Pre-seed memory if task requires it
        if task.setup_memory:
            self.memory.store(
                text=task.setup_memory["text"],
                metadata=task.setup_memory.get("metadata", {}),
                source="benchmark_setup",
                doc_type="benchmark_seed"
            )

        start = time.time()
        agent = AutonomousAgent()
        state = await agent.run(goal=task.goal, auto_approve_high_risk=True)
        elapsed = time.time() - start

        # Calculate tool accuracy
        called_tools = [tc.tool_name for tc in state.tool_calls]
        if task.expected_tools:
            matched_tools = [t for t in task.expected_tools if t in called_tools]
            accuracy = len(matched_tools) / len(task.expected_tools)
        else:
            accuracy = 1.0

        # Check recovery events
        replan_events = [e for e in state.execution_timeline if e.event_type == "REPLAN"]
        recovered = len(replan_events) > 0 and state.status == AgentStatus.COMPLETED

        # Check memory retrieval
        memory_retrieved = len(state.retrieved_context) > 0

        # Check success
        success = state.status == AgentStatus.COMPLETED

        return TaskExecutionMetric(
            task_id=task.id,
            task_name=task.name,
            mode="MODE_B_AGENTIC",
            success=success,
            iterations=state.iteration_count,
            tool_calls_count=len(state.tool_calls),
            tool_accuracy=accuracy,
            recovered=recovered,
            memory_retrieved=memory_retrieved,
            duration_seconds=elapsed,
            output_snippet=(state.final_result or "")[:150]
        )

    async def run_memory_experiment(self, memory_tasks: List[BenchmarkTask]) -> MemoryComparisonSummary:
        """
        Runs context-dependent tasks:
        1. WITHOUT vector memory
        2. WITH vector memory
        Measures real context retention differences.
        """
        # Part 1: WITHOUT memory
        self.memory.clear()
        without_success = 0
        for task in memory_tasks:
            agent = AutonomousAgent()
            state = await agent.run(goal=task.goal, auto_approve_high_risk=True)
            # Without pre-seeded memory, check if it had access to required context
            if len(state.retrieved_context) > 0 and state.status == AgentStatus.COMPLETED:
                without_success += 1

        # Part 2: WITH memory
        self.memory.clear()
        with_success = 0
        for task in memory_tasks:
            if task.setup_memory:
                self.memory.store(
                    text=task.setup_memory["text"],
                    metadata=task.setup_memory.get("metadata", {}),
                    source="benchmark_seed"
                )
            agent = AutonomousAgent()
            state = await agent.run(goal=task.goal, auto_approve_high_risk=True)
            if len(state.retrieved_context) > 0 and state.status == AgentStatus.COMPLETED:
                with_success += 1

        total = len(memory_tasks)
        without_pct = (without_success / total) * 100.0 if total > 0 else 0.0
        with_pct = (with_success / total) * 100.0 if total > 0 else 0.0
        improvement = with_pct - without_pct

        return MemoryComparisonSummary(
            total_memory_tasks=total,
            without_memory_success_count=without_success,
            without_memory_accuracy_pct=without_pct,
            with_memory_success_count=with_success,
            with_memory_accuracy_pct=with_pct,
            context_retention_improvement_pct=improvement
        )

    def compute_summary(self, mode: str, metrics: List[TaskExecutionMetric]) -> BenchmarkSummary:
        total = len(metrics)
        completed = sum(1 for m in metrics if m.success)
        comp_rate = (completed / total) * 100.0 if total > 0 else 0.0
        avg_acc = (sum(m.tool_accuracy for m in metrics) / total) * 100.0 if total > 0 else 0.0
        recoverable_tasks = [m for m in metrics if "Recovery" in m.task_name or "failure" in m.task_id or m.recovered]
        rec_rate = (sum(1 for m in recoverable_tasks if m.recovered or m.success) / len(recoverable_tasks)) * 100.0 if recoverable_tasks else 100.0
        avg_iter = sum(m.iterations for m in metrics) / total if total > 0 else 0.0
        avg_tools = sum(m.tool_calls_count for m in metrics) / total if total > 0 else 0.0
        avg_duration = sum(m.duration_seconds for m in metrics) / total if total > 0 else 0.0
        failure_rate = 100.0 - comp_rate

        return BenchmarkSummary(
            mode=mode,
            total_tasks=total,
            completed_count=completed,
            completion_rate_pct=round(comp_rate, 2),
            tool_use_accuracy_pct=round(avg_acc, 2),
            recovery_rate_pct=round(rec_rate, 2),
            average_iterations=round(avg_iter, 2),
            average_tool_calls=round(avg_tools, 2),
            average_duration_seconds=round(avg_duration, 3),
            failure_rate_pct=round(failure_rate, 2)
        )
