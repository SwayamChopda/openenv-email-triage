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
    print(f"--- Starting run_agent for task: {task_id} ---")
    
    # 1. Safely initialize OpenAI client
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        groq_api_key = os.environ.get("GROQ_API_KEY")
        hf_token = os.environ.get("HF_TOKEN")
        api_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if hf_token:
            client = OpenAI(base_url=api_base_url, api_key=hf_token)
            model_name = "default-model"
        elif groq_api_key:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)
            model_name = "llama-3.3-70b-versatile"
        elif api_key:
            client = OpenAI(api_key=api_key)
            model_name = "gpt-4o"
        else:
            print("No valid API key found. Returning 0.0")
            return 0.0
    except Exception as e:
        print(f"Unhandled exception initializing OpenAI client: {e}")
        return 0.0

    # 2. Reset the Environment
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    print(f"Connecting to environment at {env_url}...")
    
    obs_data = safe_request("POST", f"{env_url}/reset", json={"task_id": task_id})
    if not obs_data:
        print("Failed to reset environment. Returning 0.0")
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
    for step in range(max_steps):
        messages.append({
            "role": "user",
            "content": f"Current Observation: {json.dumps(obs_data)}"
        })
        
        # 3. Request action from LLM safely
        try:
            response = client.chat.completions.create(
                model=model_name,
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
            print(f"JSON parsing error at step {step}: {e}. Output was: {action_content}")
            messages.append({"role": "user", "content": f"Invalid JSON: {e}"})
            continue

        messages.append({
            "role": "assistant",
            "content": action_content
        })
        
        # 5. Send action to environment safely
        step_res = safe_request("POST", f"{env_url}/step", json=action_dict)
        if not step_res:
            print(f"Failed to execute step ({action_dict}). Stop trying.")
            break
            
        obs_data = step_res.get("observation", obs_data)
        score = step_res.get("reward", score)
        
        if step_res.get("done", False):
            print("Task marked as done.")
            break
            
    # Attempt to fetch final score
    grader_data = safe_request("GET", f"{env_url}/grader")
    if grader_data and "score" in grader_data:
        score = grader_data["score"]
        
    return score

def evaluate_all():
    print("Starting Evaluate All...")
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
            print(f"Unhandled exception inside run_agent for task {task_id}: {e}")
            scores[task_id] = 0.0

    return scores

if __name__ == "__main__":
    try:
        results = evaluate_all()
        print("Final Baseline Results:", results)
    except Exception as e:
        print(f"Unhandled exception in evaluate_all: {e}")
        exit(1)
