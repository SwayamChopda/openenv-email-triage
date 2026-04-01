from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json

from models import Action, Observation, State, StepResponse
from environment import EmailEnv
from tasks import TASKS

app = FastAPI(title="OpenEnv: Email Triage Agent")
env = EmailEnv()

@app.get("/")
@app.head("/")
def read_root():
    return {"status": "ok", "message": "Welcome to OpenEnv Email Triage. Check out /docs for schema and endpoints."}

class ResetRequest(BaseModel):
    task_id: str = "easy"

@app.get("/reset", response_model=Observation)
@app.post("/reset", response_model=Observation)
def reset_env(req: Optional[ResetRequest] = None):
    task_id = req.task_id if req else "easy"
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail="Invalid task_id")
    obs = env.reset(task_id=task_id)
    return obs

@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    step_res = env.step(action=action)
    return step_res

@app.get("/state", response_model=State)
def get_state():
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return env.state

@app.get("/grader")
def get_grader():
    if env.state is None:
        return {"score": 0.0}
    return {"score": env._get_reward()}

@app.get("/tasks")
def get_tasks():
    # Return tasks and Action JSON schema
    # Pydantic v2 usage
    from models import Action
    # Pydantic >= 2.0 uses model_json_schema() instead of schema() on Union directly? Let's use TypeAdapter
    try:
        from pydantic import TypeAdapter
        action_schema = TypeAdapter(Action).json_schema()
    except Exception:
        action_schema = {}
        
    task_list = [{"id": k, "description": v["description"]} for k, v in TASKS.items()]
    return {
        "tasks": task_list,
        "action_schema": action_schema
    }

@app.get("/baseline")
@app.post("/baseline")
def run_baseline_agent():
    # Trigger inference script and returns baseline score for all 3 tasks
    from inference import evaluate_all
    try:
        results = evaluate_all()
        return {"baseline_scores": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)
