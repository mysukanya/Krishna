import pytest
from app.tools.calculator import CalculatorTool
from app.tools.python_tool import SafePythonTool
from app.tools.data_tool import DataProcessorTool
from app.tools.http_tool import HttpTool
from app.tools.registry import registry, RiskLevel


@pytest.mark.asyncio
async def test_calculator_tool_valid_math():
    tool = CalculatorTool()
    res = await tool.execute(expression="25 * 17")
    assert res.success is True
    assert res.data["result"] == 425

    res2 = await tool.execute(expression="sqrt(144) + 10")
    assert res2.success is True
    assert res2.data["result"] == 22


@pytest.mark.asyncio
async def test_calculator_blocks_arbitrary_code():
    tool = CalculatorTool()
    res = await tool.execute(expression="__import__('os').system('ls')")
    assert res.success is False
    assert "allowed" in res.error.message.lower() or "syntax" in res.error.message.lower()


@pytest.mark.asyncio
async def test_safe_python_tool_execution():
    tool = SafePythonTool()
    code = """
items = [10, 20, 30, 40]
result = sum(items) / len(items)
"""
    res = await tool.execute(code=code)
    assert res.success is True
    assert res.data["result"] == 25.0


@pytest.mark.asyncio
async def test_safe_python_tool_blocks_imports_and_os():
    tool = SafePythonTool()
    code = "import os\nresult = os.listdir('.')"
    res = await tool.execute(code=code)
    assert res.success is False
    assert "imports are forbidden" in res.error.message.lower()


@pytest.mark.asyncio
async def test_data_processor_tool_stats():
    tool = DataProcessorTool()
    data = [{"score": 80}, {"score": 90}, {"score": 100}]
    res = await tool.execute(operation="stats", data=data, key="score")
    assert res.success is True
    assert res.data["result"]["mean"] == 90.0
    assert res.data["result"]["min"] == 80
    assert res.data["result"]["max"] == 100


@pytest.mark.asyncio
async def test_http_tool_blocks_private_ip():
    tool = HttpTool()
    res = await tool.execute(url="http://127.0.0.1:8000/secret")
    assert res.success is False
    assert "forbidden" in res.error.message.lower()


@pytest.mark.asyncio
async def test_tool_registry_management():
    assert registry.is_allowed("calculator")
    assert registry.is_allowed("safe_python")
    tool_list = registry.list_tools()
    assert len(tool_list) >= 4
    names = [t["name"] for t in tool_list]
    assert "calculator" in names
