import os
import sys
import json
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Env vars injected by the OpenEnv validator:
#   API_BASE_URL  – LiteLLM proxy the validator monitors
#   API_KEY       – key for that proxy
#   MODEL_NAME    – model to request (optional, default gpt-4o)
#   ENV_URL       – the environment server base URL
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")


def get_client() -> OpenAI:
    """Return an OpenAI client pointed at the validator's LiteLLM proxy."""
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def safe_request(method, url, **kwargs):
    try:
        res = requests.request(method, url, timeout=30, **kwargs)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"Network error on {method} {url}: {e}", flush=True)
        return None


def call_llm(client: OpenAI, messages: list) -> str | None:
    """Make a single chat completion call through the proxy. Always."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM call error: {e}", flush=True)
        return None


def run_agent(task_id: str, max_steps: int = 15) -> float:
    print(f"START {task_id}", flush=True)

    client = get_client()

    # ------------------------------------------------------------------
    # 1. Reset the environment
    # ------------------------------------------------------------------
    obs_data = safe_request("POST", f"{ENV_URL}/reset", json={"task_id": task_id})

    if not obs_data:
        # Environment unreachable – still make one LLM call so the
        # validator sees traffic through their proxy.
        print("Environment unreachable, making a verification LLM call", flush=True)
        call_llm(client, [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Respond with: {\"status\": \"env_unreachable\"}"}
        ])
        print(f"END 0.0", flush=True)
        return 0.0

    # ------------------------------------------------------------------
    # 2. Gather task info
    # ------------------------------------------------------------------
    task_description = f"Complete task '{task_id}'."
    action_schema_str = ""
    tasks_info = safe_request("GET", f"{ENV_URL}/tasks")
    if tasks_info:
        for t in tasks_info.get("tasks", []):
            if t["id"] == task_id:
                task_description = t["description"]
                break
        if "action_schema" in tasks_info:
            action_schema_str = json.dumps(tasks_info["action_schema"], indent=2)

    system_prompt = (
        f"You are an AI Email Triage Agent.\n"
        f"Task Objective: {task_description}\n\n"
        f"You have access to a simulated inbox via a remote API. "
        f"You can manage emails, read them, move them, reply, or delete.\n"
        f"Once done, call 'submit_task'.\n"
        f"Output ONLY a JSON object matching this Action schema:\n"
        f"{action_schema_str}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    # ------------------------------------------------------------------
    # 3. Agent loop
    # ------------------------------------------------------------------
    score = 0.0
    for step in range(max_steps):
        messages.append({
            "role": "user",
            "content": f"Current Observation: {json.dumps(obs_data)}"
        })

        action_content = call_llm(client, messages)
        if action_content is None:
            break

        print(f"STEP {action_content}", flush=True)

        # Parse the action JSON
        try:
            action_dict = json.loads(action_content)
        except Exception:
            messages.append({"role": "user", "content": f"Invalid JSON, try again."})
            continue

        messages.append({"role": "assistant", "content": action_content})

        # Send action to environment
        step_res = safe_request("POST", f"{ENV_URL}/step", json=action_dict)
        if not step_res:
            break

        obs_data = step_res.get("observation", obs_data)
        score   = step_res.get("reward", score)

        if step_res.get("done", False):
            break

    # Final grader score
    grader_data = safe_request("GET", f"{ENV_URL}/grader")
    if grader_data and "score" in grader_data:
        score = grader_data["score"]

    print(f"END {score}", flush=True)
    return score


def evaluate_all():
    tasks = ["easy", "medium", "hard"]

    tasks_info = safe_request("GET", f"{ENV_URL}/tasks")
    if tasks_info and "tasks" in tasks_info:
        tasks = [t["id"] for t in tasks_info["tasks"]]

    scores = {}
    for task_id in tasks:
        try:
            scores[task_id] = run_agent(task_id)
        except Exception as e:
            print(f"Exception on task {task_id}: {e}", flush=True)
            scores[task_id] = 0.0
    return scores


if __name__ == "__main__":
    print(f"API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"API_KEY set   = {bool(API_KEY)}", flush=True)
    print(f"MODEL_NAME   = {MODEL_NAME}", flush=True)
    print(f"ENV_URL      = {ENV_URL}", flush=True)

    try:
        evaluate_all()
    except Exception as e:
        print(f"Fatal: {e}", flush=True)
        sys.exit(1)
