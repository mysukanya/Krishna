from app.tools.registry import registry, BaseTool, RiskLevel, ToolResult
from app.tools.calculator import CalculatorTool
from app.tools.python_tool import SafePythonTool
from app.tools.http_tool import HttpTool
from app.tools.retrieval_tool import RetrievalTool
from app.tools.data_tool import DataProcessorTool

# Register all built-in tools
calculator_tool = CalculatorTool()
python_tool = SafePythonTool()
http_tool = HttpTool()
retrieval_tool = RetrievalTool()
data_tool = DataProcessorTool()

registry.register(calculator_tool)
registry.register(python_tool)
registry.register(http_tool)
registry.register(retrieval_tool)
registry.register(data_tool)

__all__ = [
    "registry",
    "BaseTool",
    "RiskLevel",
    "ToolResult",
    "calculator_tool",
    "python_tool",
    "http_tool",
    "retrieval_tool",
    "data_tool"
]
