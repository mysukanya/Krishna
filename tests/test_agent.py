import pytest
from app.agent.agent import AutonomousAgent
from app.agent.state import AgentStatus, PlanStep, StepStatus, ToolCallRecord
from app.agent.validator import Validator
from app.llm.mock_provider import MockRuleBasedLLMProvider
from app.utils.errors import AgentError, ErrorType


@pytest.mark.asyncio
async def test_agent_successful_loop_execution():
    agent = AutonomousAgent(llm_provider=MockRuleBasedLLMProvider())
    state = await agent.run(goal="Calculate 50 * 20 and return the result")

    assert state.status == AgentStatus.COMPLETED
    assert len(state.completed_steps) >= 1
    assert len(state.tool_calls) >= 1
    assert len(state.observations) >= 1
    assert state.final_result is not None
    assert len(state.execution_timeline) >= 4


@pytest.mark.asyncio
async def test_validator_logic():
    validator = Validator()
    step = PlanStep(
        id="s1",
        description="Calculate",
        tool="calculator",
        input={"expression": "10 * 10"},
        expected_output="100",
        success_condition="result is numeric"
    )

    # Valid outcome
    call_record = ToolCallRecord(
        call_id="c1",
        step_id="s1",
        tool_name="calculator",
        input_args={"expression": "10 * 10"},
        output={"result": 100}
    )
    val = validator.validate_step(step, call_record)
    assert val.is_valid is True
    assert val.should_replan is False

    # Failure outcome with recoverable error
    err = AgentError(
        type=ErrorType.TOOL_EXECUTION_ERROR,
        tool="calculator",
        step_id="s1",
        message="Division by zero",
        recoverable=True
    )
    val_fail = validator.validate_step(step, call_record, error=err)
    assert val_fail.is_valid is False
    assert val_fail.should_retry is True


@pytest.mark.asyncio
async def test_agent_dynamic_replanning_on_failure():
    agent = AutonomousAgent(llm_provider=MockRuleBasedLLMProvider())
    # Task with intentional failure triggering recovery
    state = await agent.run(goal="Intentional tool failure followed by recovery")

    assert state.status == AgentStatus.COMPLETED
    # Should have triggered replan
    replan_events = [e for e in state.execution_timeline if e.event_type == "REPLAN"]
    assert len(replan_events) >= 1
    assert state.current_plan.version >= 2
