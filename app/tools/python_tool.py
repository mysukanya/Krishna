import ast
import asyncio
import math
import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.tools.registry import BaseTool, RiskLevel
from app.utils.errors import AgentError, ErrorType


class PythonToolInput(BaseModel):
    code: str = Field(..., description="Python code block to execute safely. Assign output to variable `result` or end with an expression.")
    context_vars: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Variables passed to the execution environment.")


class PythonToolOutput(BaseModel):
    result: Any
    variables: Dict[str, Any]


FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "open", "input", "compile",
    "globals", "locals", "vars", "breakpoint", "exit", "quit",
    "system", "popen", "spawn", "fork", "kill", "remove", "rmdir"
}

FORBIDDEN_ATTRIBUTES = {
    "__subclasses__", "__class__", "__bases__", "__globals__",
    "__code__", "__closure__", "__builtins__", "__import__",
    "__mro__", "__dict__"
}


class SecurityValidator(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import):
        raise ValueError("Imports are forbidden in safe python environment")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        raise ValueError("Imports are forbidden in safe python environment")

    def visit_Name(self, node: ast.Name):
        if node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Use of restricted identifier '{node.id}' is prohibited")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in FORBIDDEN_ATTRIBUTES:
            raise ValueError(f"Access to internal attribute '{node.attr}' is prohibited")
        self.generic_visit(node)


class SafePythonTool(BaseTool):
    name: str = "safe_python"
    description: str = "Executes safe algorithmic Python scripts for data transformation, math, and filtering with strict AST sandboxing."
    risk_level: RiskLevel = RiskLevel.MEDIUM_RISK
    input_schema = PythonToolInput
    output_schema = PythonToolOutput

    async def run(self, code: str, context_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context_vars = context_vars or {}

        # 1. Parse and validate AST
        try:
            tree = ast.parse(code)
        except SyntaxError as se:
            raise ValueError(f"Syntax error in Python code: {se.msg} (line {se.lineno})")

        validator = SecurityValidator()
        validator.visit(tree)

        # 2. Build safe isolated environment
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "int": int, "isinstance": isinstance,
            "iter": iter, "len": len, "list": list, "map": map, "max": max,
            "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
            "range": range, "repr": repr, "reversed": reversed, "round": round,
            "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
            "zip": zip,
        }

        env = {
            "__builtins__": safe_builtins,
            "math": math,
            "json": json,
            **context_vars
        }

        # 3. Run safely in a thread to prevent event loop blocking
        def _execute():
            compiled = compile(tree, filename="<agent_sandbox>", mode="exec")
            exec(compiled, env)
            
            result = env.get("result", None)
            # Collect serializable user-defined variables
            serializable_vars = {}
            for k, v in env.items():
                if k not in ["__builtins__", "math", "json"] and not k.startswith("_"):
                    try:
                        json.dumps(v)
                        serializable_vars[k] = v
                    except (TypeError, OverflowError):
                        serializable_vars[k] = str(v)

            if result is None and len(serializable_vars) == 1:
                result = list(serializable_vars.values())[0]

            return {"result": result, "variables": serializable_vars}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _execute)
