import json
import logging
import re
import sys
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from app.config import settings

SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|token|secret|password|auth|authorization)["\']?\s*[:=]\s*["\']?([^"\'\s&]+)'),
    re.compile(r'sk-[a-zA-Z0-9_-]{20,}'),
    re.compile(r'github_pat_[a-zA-Z0-9_]{20,}'),
    re.compile(r'nfp_[a-zA-Z0-9_]{20,}'),
]


def sanitize_secrets(text: str) -> str:
    """Mask sensitive tokens, credentials, and API keys from log content."""
    if not isinstance(text, str):
        text = str(text)
    sanitized = text
    # Pattern with key name
    sanitized = SECRET_PATTERNS[0].sub(r'\1: [REDACTED]', sanitized)
    # Token-only patterns
    for pattern in SECRET_PATTERNS[1:]:
        sanitized = pattern.sub('[REDACTED]', sanitized)
    return sanitized


class StructuredJsonFormatter(logging.Formatter):
    """Formats logs into clean, structured JSON format with secret masking."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_secrets(record.getMessage()),
        }
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            # Sanitize any structured payloads
            clean_data = {}
            for k, v in record.structured_data.items():
                if isinstance(v, str):
                    clean_data[k] = sanitize_secrets(v)
                elif isinstance(v, (dict, list)):
                    clean_data[k] = json.loads(sanitize_secrets(json.dumps(v)))
                else:
                    clean_data[k] = v
            payload["context"] = clean_data

        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)
        logger.propagate = False
    return logger


class AgentLogger:
    """High-level logger specifically tracking autonomous agent lifecycle events."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._logger = get_logger("krishna.agent")

    def _log_event(self, event_type: str, message: str, extra: Optional[Dict[str, Any]] = None):
        structured = {
            "task_id": self.task_id,
            "event_type": event_type,
            **(extra or {})
        }
        self._logger.info(message, extra={"structured_data": structured})

    def log_goal_received(self, goal: str):
        self._log_event("GOAL_RECEIVED", f"Goal received: {goal[:100]}...", {"user_goal": goal})

    def log_planner_decision(self, plan_version: int, step_count: int, reasoning: Optional[str] = None):
        self._log_event("PLANNER_DECISION", f"Plan generated (v{plan_version}) with {step_count} steps", {
            "plan_version": plan_version,
            "step_count": step_count,
            "reasoning": reasoning
        })

    def log_tool_selected(self, step_id: str, tool_name: str, tool_input: Dict[str, Any]):
        self._log_event("TOOL_SELECTED", f"Step {step_id}: Selected tool '{tool_name}'", {
            "step_id": step_id,
            "tool_name": tool_name,
            "tool_input": tool_input
        })

    def log_tool_executed(self, step_id: str, tool_name: str, duration_ms: float, output_summary: str):
        self._log_event("TOOL_EXECUTED", f"Step {step_id}: Tool '{tool_name}' executed in {duration_ms:.1f}ms", {
            "step_id": step_id,
            "tool_name": tool_name,
            "duration_ms": duration_ms,
            "output_summary": output_summary[:300]
        })

    def log_validation(self, step_id: str, is_valid: bool, reason: str, should_retry: bool, should_replan: bool):
        self._log_event("VALIDATION", f"Step {step_id}: Validation {'PASSED' if is_valid else 'FAILED'} - {reason}", {
            "step_id": step_id,
            "is_valid": is_valid,
            "reason": reason,
            "should_retry": should_retry,
            "should_replan": should_replan
        })

    def log_replan(self, trigger_step: str, reason: str, new_version: int):
        self._log_event("REPLAN", f"Replanning triggered at step {trigger_step}: {reason}", {
            "trigger_step": trigger_step,
            "reason": reason,
            "new_plan_version": new_version
        })

    def log_completion(self, status: str, total_iterations: int, duration_seconds: float):
        self._log_event("COMPLETION", f"Task finished with status '{status}' after {total_iterations} iterations", {
            "status": status,
            "total_iterations": total_iterations,
            "duration_seconds": duration_seconds
        })
