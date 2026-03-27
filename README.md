# OpenEnv: Email Triage Agent

This repository contains a full OpenEnv-compliant real-world environment for evaluating Autonomous AI Agents on complex, multi-step textual tasks.

## Environment Description

**Domain:** Executive Assistant / Email Triage
**Motivation:** Most AI agents are evaluated on simplistic toy problems (e.g. Tic-Tac-Toe, Blocksworld) or rigid web automation. Email triage requires reading comprehension, synthesis, logic, and multi-step orchestration (e.g., categorizing, replying, summarizing) which heavily taxes an LLM's reasoning and memory management skills.

## Tasks & Difficulty

1. **Easy:** Sort 5 incoming HR emails into the `HR` folder. (Tests basic looping and action execution)
2. **Medium:** Categorize support emails into `Tech` and `Spam` folders. Send template replies to users asking for `Refunds` and move them to a processed folder. (Tests semantic understanding and multi-action composition)
3. **Hard:** A chaotic inbox containing spam, newsletters, and 3 specific emails regarding "Project Alpha". The agent must delete spam, file newsletters, and—crucially—synthesize the project status, blockers, and deployment date into a direct reply to the angry manager. (Tests advanced reasoning, memory retrieval, synthesis, and precise execution)

## Setup Instructions

1. **Clone the repository.**
2. **Docker Build:**
   ```bash
   docker build -t openenv-email-triage .
   docker run -p 7860:7860 openenv-email-triage
   ```
3. **Local Dev (Python 3.11+):**
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --port 7860
   ```

## Running the Baseline

A baseline agent relying on `gpt-4o` structure parsing is included.
To run it, expose your OpenAI API key and trigger the `/baseline` endpoint or script:

```bash
export OPENAI_API_KEY="sk-..."
python baseline.py
```

## Action Space

The agent interacts via a discriminated JSON union `Action` with the following strict endpoints:
- `move_email` (email_id, destination_folder)
- `read_email` (email_id) -> updates Observation
- `reply_email` (email_id, reply_body)
- `delete_email` (email_id)
- `change_folder` (folder_name) -> navigates virtual folders
- `submit_task` -> signals completion

## Observation Space

At each step, the model receives an `Observation`:
- `current_folder`: Which directory is actively viewed
- `emails_in_folder`: A list of summary headers
- `current_email`: Null unless an email was explicitly opened
- `folders`: A list of available directories built over time

### OpenEnv Spec Compliance
- Endpoints: `/step`, `/reset`, `/state`, `/tasks`, `/grader`, `/baseline`
- Strict Pydantic models for type safety.
- Deploys as a standard containerized Hugging Face Space.
