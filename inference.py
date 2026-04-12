import os
import sys
import json
import requests
from openai import OpenAI

def safe_request(method, url, **kwargs):
    try:
        res = requests.request(method, url, timeout=30, **kwargs)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"Network error on {method} {url}: {e}", flush=True)
        return None

def run_agent(task_id: str, max_steps=15) -> float:
    print(f"[START] task={task_id}", flush=True)
    
    # 1. Initialize OpenAI client using validator-injected env vars
    try:
        api_base = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")
        
        print(f"Using API_BASE_URL={api_base}", flush=True)
        print(f"Using MODEL_NAME={model_name}", flush=True)
        print(f"API_KEY is set: {bool(api_key)}", flush=True)
        
        client = OpenAI(base_url=api_base, api_key=api_key)
    except KeyError as e:
        print(f"Missing required env var: {e}", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0

    # 2. Reset the Environment
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    print(f"Using ENV_URL={env_url}", flush=True)
    
    obs_data = safe_request("POST", f"{env_url}/reset", json={"task_id": task_id})
    if not obs_data:
        print(f"Failed to reset environment for task {task_id}", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0

    # Get task info
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
        
        # 3. Request action from LLM
        try:
            print(f"Calling LLM (step {step})...", flush=True)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            action_content = response.choices[0].message.content
            print(f"LLM responded successfully", flush=True)
        except Exception as e:
            print(f"OpenAI API Error at step {step}: {e}", flush=True)
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
        
        # 5. Send action to environment
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
            print(f"Exception running task {task_id}: {e}", flush=True)
            scores[task_id] = 0.0

    return scores

if __name__ == "__main__":
    # Print all relevant env vars for debugging
    print(f"API_BASE_URL set: {'API_BASE_URL' in os.environ}", flush=True)
    print(f"API_KEY set: {'API_KEY' in os.environ}", flush=True)
    print(f"ENV_URL set: {'ENV_URL' in os.environ}", flush=True)
    print(f"MODEL_NAME set: {'MODEL_NAME' in os.environ}", flush=True)
    
    try:
        evaluate_all()
    except Exception as e:
        print(f"Fatal error: {e}", flush=True)
        sys.exit(1)
