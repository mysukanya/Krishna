import pytest
from app.agent.planner import Planner
from app.agent.state import TaskPlan, PlanStep
from app.llm.mock_provider import MockRuleBasedLLMProvider


@pytest.mark.asyncio
async def test_planner_creates_valid_pydantic_plan():
    llm = MockRuleBasedLLMProvider()
    planner = Planner(llm)

    goal = "Calculate 25 * 17 and return the result"
    plan = await planner.create_plan(goal=goal)

    assert isinstance(plan, TaskPlan)
    assert plan.version == 1
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool in ["calculator", "safe_python"]
    assert plan.steps[0].expected_output != ""
    assert plan.steps[0].success_condition != ""


@pytest.mark.asyncio
async def test_planner_rejects_unregistered_tool():
    class RogueLLM(MockRuleBasedLLMProvider):
        async def generate_structured(self, prompt, schema, system_prompt="", temperature=0.1):
            return schema.model_validate({
                "goal": "rogue",
                "steps": [{
                    "id": "step_bad",
                    "description": "malicious step",
                    "tool": "non_existent_rogue_tool",
                    "input": {},
                    "expected_output": "something",
                    "success_condition": "done"
                }],
                "version": 1
            })

    planner = Planner(RogueLLM())
    with pytest.raises(Exception) as excinfo:
        await planner.create_plan(goal="test invalid tool")
    assert "unknown tool" in str(excinfo.value).lower()
