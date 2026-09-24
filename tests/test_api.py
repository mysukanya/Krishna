import pytest
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "tools_count" in data
    assert data["tools_count"] >= 4


def test_list_tools_endpoint():
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert data["count"] >= 4


def test_memory_crud_endpoints():
    # Store memory
    res = client.post("/memory", json={
        "text": "Antigravity IDE runs high performance autonomous agent loops.",
        "metadata": {"category": "test"},
        "source": "api_test"
    })
    assert res.status_code == 200
    mem_data = res.json()
    assert "id" in mem_data
    assert mem_data["status"] == "stored"

    # Search memory
    search_res = client.get("/memory/search", params={"q": "autonomous agent", "limit": 2})
    assert search_res.status_code == 200
    s_data = search_res.json()
    assert s_data["count"] >= 1
    assert "autonomous" in s_data["results"][0]["text"].lower()


def test_agent_run_and_status_endpoints():
    run_res = client.post("/agent/run", json={
        "goal": "Calculate 30 * 40 and explain result"
    })
    assert run_res.status_code == 200
    data = run_res.json()
    assert "task_id" in data
    assert data["status"] == "completed"
    assert len(data["steps"]) >= 1
    assert data["iterations"] >= 1

    task_id = data["task_id"]
    get_res = client.get(f"/agent/{task_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["task_id"] == task_id
    assert get_data["status"] == "completed"
