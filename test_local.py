from server.app import app
from fastapi.testclient import TestClient
import json

client = TestClient(app)

def test_api():
    print("Testing /tasks")
    res = client.get("/tasks")
    assert res.status_code == 200
    data = res.json()
    assert "tasks" in data
    assert "action_schema" in data
    
    print("Testing /reset")
    res = client.post("/reset", json={"task_id": "easy"})
    assert res.status_code == 200
    
    print("Testing /state")
    res = client.get("/state")
    assert res.status_code == 200
    
    print("Testing /step")
    res = client.post("/step", json={"action": "move_email", "email_id": "e1", "destination_folder": "HR"})
    assert res.status_code == 200
    data = res.json()
    assert "observation" in data
    assert "reward" in data
    
    print("Testing /grader")
    res = client.get("/grader")
    assert res.status_code == 200
    
    print("All basic endpoints passed!")

if __name__ == "__main__":
    test_api()
