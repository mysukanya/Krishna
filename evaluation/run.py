import asyncio
import json
import os
import sys
import time

from evaluation.benchmark_tasks import BENCHMARK_TASKS
from evaluation.evaluator import BenchmarkEvaluator


async def main():
    print("=" * 80)
    print("        KRISHNA AUTONOMOUS AGENT - BENCHMARK & EVALUATION SUITE")
    print("=" * 80)
    print(f"Total Multi-Step Benchmark Tasks: {len(BENCHMARK_TASKS)}")
    print("Executing Real Experiments: Mode A (Single-Pass) vs Mode B (Autonomous Loop)...\n")

    evaluator = BenchmarkEvaluator()

    # 1. Run Mode A (Baseline Single-Pass)
    print("[1/3] Running Mode A: Single-Pass Direct LLM Call...")
    mode_a_metrics = []
    for idx, task in enumerate(BENCHMARK_TASKS, 1):
        metric = await evaluator.run_mode_a_single_pass(task)
        mode_a_metrics.append(metric)
        status = "PASSED" if metric.success else "FAILED"
        print(f"    [{idx:02d}/{len(BENCHMARK_TASKS)}] {task.name:<40} -> {status} ({metric.duration_seconds:.3f}s)")

    summary_a = evaluator.compute_summary("MODE_A_SINGLE_PASS", mode_a_metrics)

    # 2. Run Mode B (Agentic Loop)
    print("\n[2/3] Running Mode B: Autonomous Agentic Plan -> Act -> Observe -> Validate -> Replan...")
    mode_b_metrics = []
    for idx, task in enumerate(BENCHMARK_TASKS, 1):
        metric = await evaluator.run_mode_b_agentic(task)
        mode_b_metrics.append(metric)
        status = "PASSED" if metric.success else "FAILED"
        replan_badge = " [REPLAN RECOVERED]" if metric.recovered else ""
        print(f"    [{idx:02d}/{len(BENCHMARK_TASKS)}] {task.name:<40} -> {status}{replan_badge} ({metric.duration_seconds:.3f}s, {metric.iterations} iter, {metric.tool_calls_count} tools)")

    summary_b = evaluator.compute_summary("MODE_B_AGENTIC", mode_b_metrics)

    # 3. Run Memory Experiment
    print("\n[3/3] Running Vector Memory Retention Experiment on Context Tasks...")
    memory_tasks = [t for t in BENCHMARK_TASKS if t.requires_memory]
    mem_summary = await evaluator.run_memory_experiment(memory_tasks)
    print(f"    - Evaluated {mem_summary.total_memory_tasks} context-dependent tasks.")
    print(f"    - Without Vector Memory: {mem_summary.without_memory_success_count}/{mem_summary.total_memory_tasks} ({mem_summary.without_memory_accuracy_pct:.1f}%)")
    print(f"    - With Vector Memory:    {mem_summary.with_memory_success_count}/{mem_summary.total_memory_tasks} ({mem_summary.with_memory_accuracy_pct:.1f}%)")
    print(f"    - Context Retention Lift: +{mem_summary.context_retention_improvement_pct:.1f}%\n")

    # 4. Print Comparison Table
    print("=" * 80)
    print("                   ACTUAL MEASURED BENCHMARK COMPARISON REPORT")
    print("=" * 80)
    col_w = 34
    print(f"{'METRIC':<36} | {'MODE A (SINGLE-PASS)':<20} | {'MODE B (KRISHNA AGENTIC)':<20}")
    print("-" * 80)
    print(f"{'Task Completion Rate':<36} | {f'{summary_a.completion_rate_pct}%':<20} | {f'{summary_b.completion_rate_pct}%':<20}")
    print(f"{'Tool Use Accuracy':<36} | {f'{summary_a.tool_use_accuracy_pct}%':<20} | {f'{summary_b.tool_use_accuracy_pct}%':<20}")
    print(f"{'Failure Recovery Rate':<36} | {f'{summary_a.recovery_rate_pct}%':<20} | {f'{summary_b.recovery_rate_pct}%':<20}")
    print(f"{'Average Steps / Iterations':<36} | {f'{summary_a.average_iterations}':<20} | {f'{summary_b.average_iterations}':<20}")
    print(f"{'Average Tool Calls':<36} | {f'{summary_a.average_tool_calls}':<20} | {f'{summary_b.average_tool_calls}':<20}")
    print(f"{'Average Execution Time':<36} | {f'{summary_a.average_duration_seconds}s':<20} | {f'{summary_b.average_duration_seconds}s':<20}")
    print(f"{'Overall Failure Rate':<36} | {f'{summary_a.failure_rate_pct}%':<20} | {f'{summary_b.failure_rate_pct}%':<20}")
    print("=" * 80)

    # 5. Persist Clean JSON Report
    report_data = {
        "timestamp": time.time(),
        "total_tasks": len(BENCHMARK_TASKS),
        "mode_a_summary": summary_a.model_dump(),
        "mode_b_summary": summary_b.model_dump(),
        "memory_experiment": mem_summary.model_dump(),
        "task_breakdown_mode_a": [m.model_dump() for m in mode_a_metrics],
        "task_breakdown_mode_b": [m.model_dump() for m in mode_b_metrics]
    }
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[OK] Full evaluation report successfully saved to: {report_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
