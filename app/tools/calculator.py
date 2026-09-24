import ast
import math
import operator
from typing import Any, Dict, Union
from pydantic import BaseModel, Field

from app.tools.registry import BaseTool, RiskLevel


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate, e.g. '(120 * 45) / 2' or 'sqrt(144) + 10'")


class CalculatorOutput(BaseModel):
    result: Union[int, float]
    expression: str


class SafeEvaluator(ast.NodeVisitor):
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_FUNCTIONS = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "pow": pow,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        "sum": sum,
        "max": max,
        "min": min,
    }

    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def visit(self, node: ast.AST) -> Any:
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported syntax or token: {node.__class__.__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Constant value '{node.value}' of type {type(node.value)} is not allowed")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.ALLOWED_CONSTANTS:
            return self.ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Name '{node.id}' is not an allowed constant")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # Guard against huge power operations causing DoS
        if op_type is ast.Pow:
            if isinstance(right, (int, float)) and right > 1000:
                raise ValueError("Exponent too large (maximum allowed is 1000)")
        return self.ALLOWED_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        if op_type not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Unary operator {op_type.__name__} is not allowed")
        operand = self.visit(node.operand)
        return self.ALLOWED_OPERATORS[op_type](operand)

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        func_name = node.func.id
        if func_name not in self.ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not in the allowed math function list")
        args = [self.visit(arg) for arg in node.args]
        return self.ALLOWED_FUNCTIONS[func_name](*args)

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Safely evaluates mathematical expressions (arithmetic, powers, roots, trigonometry, statistics) without unrestricted code execution."
    risk_level: RiskLevel = RiskLevel.LOW_RISK
    input_schema = CalculatorInput
    output_schema = CalculatorOutput

    async def run(self, expression: str) -> Dict[str, Any]:
        expr = expression.strip()
        parsed = ast.parse(expr, mode='eval')
        evaluator = SafeEvaluator()
        result = evaluator.visit(parsed)
        
        # Round floating point inaccuracies if clean integer
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return {"result": result, "expression": expr}
