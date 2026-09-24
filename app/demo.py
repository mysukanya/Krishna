import asyncio
import sys
import time

from app.agent.agent import AutonomousAgent
from app.memory.memory_manager import memory_manager


async def run_demo():
    print("=" * 70)
    print("      KRISHNA AUTONOMOUS AGENT SYSTEM - LIVE DEMONSTRATION")
    print("=" * 70)
    print("Architecture: GOAL -> RETRIEVE -> PLAN -> ACT -> OBSERVE -> VALIDATE -> REPLAN -> RESULT\n")

    # Step A: Seed memory with historical context
    print("[1] PRE-SEEDED MEMORY INGESTION:")
    mem_id = memory_manager.store(
        text="Fiscal Year 2025 Revenue: $120,000 USD. Operational Expenses: $45,000 USD. Tax Rate: 15%.",
        metadata={"category": "finance", "year": 2025},
        source="annual_report",
        doc_type="financial_record"
    )
    print(f"    - Ingested financial report to vector memory (Record ID: {mem_id})")

    # Step B: Define High-Level Goal
    goal = "Retrieve the FY2025 revenue from memory, calculate net profit after 15% tax, and produce a verified financial summary."
    print(f"\n[2] USER SUBMITS GOAL:\n    \"{goal}\"\n")

    # Step C: Initialize Agent
    agent = AutonomousAgent()
    print("[3] STARTING AUTONOMOUS AGENT EXECUTION LOOP...")
    start_time = time.time()

    state = await agent.run(goal=goal)
    elapsed = time.time() - start_time

    # Step D: Display Execution Trace
    print("\n" + "=" * 70)
    print("                     EXECUTION TIMELINE TRACE")
    print("=" * 70)
    for idx, event in enumerate(state.execution_timeline, 1):
        print(f"[{idx:02d}] {event.event_type:<10} | {event.title}")
        print(f"     Details: {event.description}")
        if event.metadata:
            print(f"     Metadata: {event.metadata}")
        print("-" * 70)

    # Step E: Display Final Outcome
    print("\n" + "=" * 70)
    print("                       FINAL SUMMARY")
    print("=" * 70)
    print(f"Task ID:          {state.task_id}")
    print(f"Status:           {state.status.value.upper()}")
    print(f"Iterations:       {state.iteration_count}")
    print(f"Steps Completed:  {len(state.completed_steps)}")
    print(f"Total Time:       {elapsed:.2f} seconds")
    print(f"\nSynthesized Result:\n{state.final_result}\n")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_demo())
