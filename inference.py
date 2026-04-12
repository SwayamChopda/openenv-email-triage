import os
import json
import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def safe_request(method, url, **kwargs):
    try:
        res = requests.request(method, url, timeout=10, **kwargs)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"Network error on {method} {url}: {e}")
        return None

def run_agent(task_id: str, max_steps=15) -> float:
    print(f"[START] task={task_id}", flush=True)
    
    # 1. Safely initialize OpenAI client
    try:
        API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
        HF_TOKEN = os.getenv("HF_TOKEN")
        
        # Verify valid fallback even if set to empty string
        if not API_BASE_URL:
            API_BASE_URL = "https://api.openai.com/v1"
        if not MODEL_NAME:
            MODEL_NAME = "gpt-4o"
            
        api_key = os.getenv("API_KEY") or HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy")
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    except Exception as e:
        # print(f"Unhandled exception initializing OpenAI client: {e}")
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0

    # 2. Reset the Environment
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    
    obs_data = safe_request("POST", f"{env_url}/reset", json={"task_id": task_id})
    if not obs_data:
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0

    # We assume 'tasks' endpoint tells us what the task is. Optional, but helpful.
    task_description = f"Complete task '{task_id}'."
    action_schema_str = ""
    tasks_info = safe_request("GET", f"{env_url}/tasks")
    if tasks_info:
        if "tasks" in tasks_info:
            for t in tasks_info["tasks"]:
                if t["id"] == task_id:
                    task_description = t["description"]
                    break
        if "action_schema" in tasks_info:
            action_schema_str = json.dumps(tasks_info["action_schema"], indent=2)

    system_prompt = f"""
    You are an AI Email Triage Agent.
    Task Objective: {task_description}
    
    You have access to a simulated inbox via a remote API. You can manage emails, read them, move them around, reply, or delete.
    Once you believe you have accomplished the objective, call 'submit_task'.
    Try to be efficient and check the objective.
    You must output a JSON object representing your exact action and nothing else.
    Ensure your output strictly matches the following Action JSON schema:
    {action_schema_str}
    """
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    score = 0.0
    step_count = 0
    for step in range(max_steps):
        messages.append({
            "role": "user",
            "content": f"Current Observation: {json.dumps(obs_data)}"
        })
        
        # 3. Request action from LLM safely
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"}
            )
            action_content = response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API Error at step {step}: {e}")
            break
            
        # 4. Parse JSON safely
        try:
            action_dict = json.loads(action_content)
        except Exception as e:
            messages.append({"role": "user", "content": f"Invalid JSON: {e}"})
            continue

        messages.append({
            "role": "assistant",
            "content": action_content
        })
        
        # 5. Send action to environment safely
        step_count += 1
        print(f"[STEP] step={step_count} reward={score}", flush=True)
        step_res = safe_request("POST", f"{env_url}/step", json=action_dict)
        if not step_res:
            break
            
        obs_data = step_res.get("observation", obs_data)
        score = step_res.get("reward", score)
        
        if step_res.get("done", False):
            break
            
    # Attempt to fetch final score
    grader_data = safe_request("GET", f"{env_url}/grader")
    if grader_data and "score" in grader_data:
        score = grader_data["score"]
        
    print(f"[END] task={task_id} score={score} steps={step_count}", flush=True)
    return score

def evaluate_all():
    scores = {}
    tasks = ["easy", "medium", "hard"]
    
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    tasks_info = safe_request("GET", f"{env_url}/tasks")
    if tasks_info and "tasks" in tasks_info:
        tasks = [t["id"] for t in tasks_info["tasks"]]
    
    for task_id in tasks:
        try:
            score = run_agent(task_id)
            scores[task_id] = score
        except Exception as e:
            scores[task_id] = 0.0

    return scores

if __name__ == "__main__":
    try:
        evaluate_all()
    except Exception as e:
        exit(1)
