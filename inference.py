import os
import json
from openai import OpenAI
from pydantic import TypeAdapter
from environment import EmailEnv
from tasks import TASKS
from models import Action

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def run_agent(task_id: str, max_steps=15) -> float:
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("HF_TOKEN not set. Baseline cannot run. Returning 0.0")
        return 0.0
        
    client = OpenAI(base_url=api_base_url, api_key=hf_token)
        
    env = EmailEnv()
    obs = env.reset(task_id=task_id)
    
    # We define tools dynamically based on Action derived classes
    action_schema = TypeAdapter(Action).json_schema()
    
    system_prompt = f"""
    You are an AI Email Triage Agent.
    Task Objective: {TASKS[task_id]['description']}
    
    You have access to a simulated inbox. You can manage emails, read them, move them around, reply, or delete.
    Once you believe you have accomplished the objective, call 'submit_task'.
    Try to be efficient and check the objective.
    You must output a JSON object representing your exact action and nothing else.
    """
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    score = 0.0
    for _ in range(max_steps):
        # We prompt the model with current observation
        messages.append({
            "role": "user",
            "content": f"Current Observation: {obs.model_dump_json()}"
        })
        
        try:
            response = client.chat.completions.create(
                model=model_name, # dynamic selection
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # OpenAI output should be a JSON matching our Action schema.
            # We'll ask it to output a JSON object containing the action.
        except Exception as e:
            # Maybe model not available or API key issue
            print(f"OpenAI API Error: {e}")
            break
            
        action_dict = {}
        try:
            action_content = response.choices[0].message.content
            action_dict = json.loads(action_content)
            
            # Fix up the action: OpenAI might nest it or output directly.
            # Let's assume the JSON is directly the Action dict.
            action_obj = TypeAdapter(Action).validate_python(action_dict)
            
            messages.append({
                "role": "assistant",
                "content": json.dumps(action_obj.model_dump())    
            })
            
            step_res = env.step(action_obj)
            obs = step_res.observation
            score = step_res.reward
            
            if step_res.done:
                break
                
        except Exception as e:
            messages.append({"role": "user", "content": f"Invalid action format or environment error: {e}. Output must match action schema."})
            
    return env._get_reward()
    

def evaluate_all():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return {"error": "HF_TOKEN not set"}
        
    scores = {}
    for task_id in TASKS.keys():
        print(f"Running baseline for task: {task_id}")
        score = run_agent(task_id)
        scores[task_id] = score
        print(f"Task: {task_id} - Score: {score}")
    return scores

if __name__ == "__main__":
    results = evaluate_all()
    print("Final Baseline Results:", results)
