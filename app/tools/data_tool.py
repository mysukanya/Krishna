import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from app.tools.registry import BaseTool, RiskLevel


class DataProcessorInput(BaseModel):
    operation: str = Field(..., description="Operation to perform: 'filter', 'sort', 'stats', 'aggregate_sum', 'extract_keys', 'json_parse'")
    data: Union[List[Any], Dict[str, Any], str] = Field(..., description="Data to process (list, dict, or JSON string)")
    key: Optional[str] = Field(default=None, description="Key/field to sort, filter, or aggregate by")
    filter_value: Optional[Any] = Field(default=None, description="Value to match for filter operation")
    reverse: bool = Field(default=False, description="Sort descending if true")


class DataProcessorOutput(BaseModel):
    operation: str
    result: Any
    count: int


class DataProcessorTool(BaseTool):
    name: str = "data_processor"
    description: str = "Performs structured data operations: JSON parsing, filtering lists, sorting, computing statistics (min, max, mean), and aggregating."
    risk_level: RiskLevel = RiskLevel.LOW_RISK
    input_schema = DataProcessorInput
    output_schema = DataProcessorOutput

    async def run(
        self,
        operation: str,
        data: Union[List[Any], Dict[str, Any], str],
        key: Optional[str] = None,
        filter_value: Optional[Any] = None,
        reverse: bool = False
    ) -> Dict[str, Any]:
        # Parse JSON string if given as string
        parsed_data = data
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except Exception:
                parsed_data = data

        op = operation.lower().strip()

        if op == "json_parse":
            return {"operation": op, "result": parsed_data, "count": len(parsed_data) if isinstance(parsed_data, (list, dict)) else 1}

        if op == "filter":
            if not isinstance(parsed_data, list):
                raise ValueError("Filter operation requires a list of items.")
            if key:
                filtered = [item for item in parsed_data if isinstance(item, dict) and item.get(key) == filter_value]
            else:
                filtered = [item for item in parsed_data if item == filter_value]
            return {"operation": op, "result": filtered, "count": len(filtered)}

        if op == "sort":
            if not isinstance(parsed_data, list):
                raise ValueError("Sort operation requires a list of items.")
            if key:
                sorted_list = sorted(parsed_data, key=lambda x: (x.get(key) is None, x.get(key, 0)), reverse=reverse)
            else:
                sorted_list = sorted(parsed_data, reverse=reverse)
            return {"operation": op, "result": sorted_list, "count": len(sorted_list)}

        if op == "stats":
            values = []
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    val = item.get(key) if (isinstance(item, dict) and key) else item
                    if isinstance(val, (int, float)):
                        values.append(val)
            if not values:
                raise ValueError("No numeric values found to calculate statistics.")
            stats = {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 4)
            }
            return {"operation": op, "result": stats, "count": len(values)}

        if op == "aggregate_sum":
            total = 0.0
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    val = item.get(key) if (isinstance(item, dict) and key) else item
                    if isinstance(val, (int, float)):
                        total += val
            return {"operation": op, "result": round(total, 4), "count": 1}

        if op == "extract_keys":
            if isinstance(parsed_data, dict):
                keys = list(parsed_data.keys())
            elif isinstance(parsed_data, list) and parsed_data and isinstance(parsed_data[0], dict):
                keys = list(parsed_data[0].keys())
            else:
                keys = []
            return {"operation": op, "result": keys, "count": len(keys)}

        raise ValueError(f"Unknown data operation '{operation}'. Supported: filter, sort, stats, aggregate_sum, extract_keys, json_parse")
