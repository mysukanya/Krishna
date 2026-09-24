from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    id: str
    name: str
    category: str
    goal: str
    requires_memory: bool = False
    setup_memory: Optional[Dict[str, Any]] = None
    expected_tools: List[str] = Field(default_factory=list)
    success_keywords: List[str] = Field(default_factory=list)


BENCHMARK_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        id="task_01",
        name="Mathematical Multi-Step Reasoning",
        category="math",
        goal="Calculate (45 * 12) + (250 / 5) - 30 and return verified result.",
        requires_memory=False,
        expected_tools=["calculator"],
        success_keywords=["560", "result", "numeric", "completed"]
    ),
    BenchmarkTask(
        id="task_02",
        name="Data Transformation & Statistics",
        category="data",
        goal="Filter and compute summary statistics on data items [10, 20, 30, 40].",
        requires_memory=False,
        expected_tools=["data_processor"],
        success_keywords=["stats", "mean", "25", "count"]
    ),
    BenchmarkTask(
        id="task_03",
        name="Retrieval + Calculation Workflow",
        category="hybrid",
        goal="Retrieve FY2025 revenue from memory, calculate profit margin after expenses.",
        requires_memory=True,
        setup_memory={"text": "Fiscal Year 2025 Revenue: $120,000 USD. Operational Expenses: $45,000 USD.", "metadata": {"year": 2025}},
        expected_tools=["retrieval_tool", "calculator"],
        success_keywords=["revenue", "profit", "120000", "45000", "completed"]
    ),
    BenchmarkTask(
        id="task_04",
        name="API Ingestion & Processing",
        category="api",
        goal="Fetch remote verified API endpoint and structure data payload.",
        requires_memory=False,
        expected_tools=["http_tool", "safe_python"],
        success_keywords=["status_code", "200", "completed"]
    ),
    BenchmarkTask(
        id="task_05",
        name="Intentional Tool Failure & Dynamic Recovery",
        category="recovery",
        goal="Intentional tool failure followed by recovery and backup execution.",
        requires_memory=False,
        expected_tools=["http_tool"],
        success_keywords=["completed", "recovered", "fallback", "200"]
    ),
    BenchmarkTask(
        id="task_06",
        name="Memory-Dependent Context Retrieval",
        category="memory",
        goal="Query vector memory for confidential project codename and explain its purpose.",
        requires_memory=True,
        setup_memory={"text": "Project Codename: KRISHNA-ORBITAL. Purpose: Autonomous planetary satellite constellation.", "metadata": {"class": "secret"}},
        expected_tools=["retrieval_tool"],
        success_keywords=["krishna-orbital", "satellite", "constellation"]
    ),
    BenchmarkTask(
        id="task_07",
        name="Multi-Tool Chained Workflow",
        category="workflow",
        goal="Multi-tool workflow: Query metrics from memory, compute ratio, and format structured summary.",
        requires_memory=True,
        setup_memory={"text": "Production batch 88: 500 units inspected, 475 passed quality assurance.", "metadata": {"batch": 88}},
        expected_tools=["retrieval_tool", "calculator"],
        success_keywords=["units", "passed", "ratio", "quality"]
    ),
    BenchmarkTask(
        id="task_08",
        name="Invalid Input Repair & Recovery",
        category="recovery",
        goal="Execute calculation with initially malformed syntax then repair to valid syntax.",
        requires_memory=False,
        expected_tools=["calculator"],
        success_keywords=["425", "completed", "result"]
    ),
    BenchmarkTask(
        id="task_09",
        name="Context Retention Across Execution",
        category="memory",
        goal="Search memory for server cluster configurations and verify active host count.",
        requires_memory=True,
        setup_memory={"text": "Cluster us-east-prod has 16 active bare-metal nodes with NVMe RAID.", "metadata": {"cluster": "us-east-prod"}},
        expected_tools=["retrieval_tool"],
        success_keywords=["us-east-prod", "nodes", "16", "active"]
    ),
    BenchmarkTask(
        id="task_10",
        name="Long Multi-Step Execution (5 Steps)",
        category="complex",
        goal="Execute complex multi-step pipeline covering memory retrieval, HTTP check, math evaluation, data stats, and output formatting.",
        requires_memory=True,
        setup_memory={"text": "Baseline configuration: 100 max concurrency, 50ms latency target.", "metadata": {"type": "baseline"}},
        expected_tools=["retrieval_tool", "http_tool", "calculator", "data_processor", "safe_python"],
        success_keywords=["status", "success", "completed"]
    )
]
