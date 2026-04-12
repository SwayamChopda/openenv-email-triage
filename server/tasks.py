from typing import Dict, Any, Callable
from models import Email, State

def setup_easy_task() -> State:
    emails = {
        "e1": Email(id="e1", sender="hr@company.com", subject="Policy Update", body="Please review the new PTO policy."),
        "e2": Email(id="e2", sender="alice@company.com", subject="Lunch?", body="Want to grab lunch?"),
        "e3": Email(id="e3", sender="hr@company.com", subject="Training", body="Mandatory security training link >here<."),
        "e4": Email(id="e4", sender="bob@company.com", subject="Project XYZ", body="Can you review the PR?"),
        "e5": Email(id="e5", sender="hr@company.com", subject="Benefits Enrollment", body="Open enrollment begins today.")
    }
    return State(
        task_id="easy",
        emails=emails,
        sent_replies={},
        folders=["inbox", "HR"],
        current_folder="inbox",
        current_email_id=None,
        step_count=0,
        done=False
    )

def grade_easy_task(state: State) -> float:
    # Easy Task: Move all hr@company.com emails to 'HR' folder, others stay in 'inbox'
    score = 0.0
    hr_emails = [e for e in state.emails.values() if e.sender == "hr@company.com"]
    other_emails = [e for e in state.emails.values() if e.sender != "hr@company.com"]
    
    # 0.5 points for correctly moving HR emails
    correct_hr = sum(1 for e in hr_emails if e.folder == "HR")
    score += (correct_hr / len(hr_emails)) * 0.5
    
    # 0.5 points for NOT moving other emails
    correct_other = sum(1 for e in other_emails if e.folder == "inbox")
    score += (correct_other / len(other_emails)) * 0.5
    
    return max(0.0, min(1.0, score))


def setup_medium_task() -> State:
    emails = {
        "m1": Email(id="m1", sender="customer1@gmail.com", subject="Refund Request", body="My item arrived broken. I want a refund please."),
        "m2": Email(id="m2", sender="customer2@yahoo.com", subject="App is crashing", body="Every time I open the app it crashes on the login screen."),
        "m3": Email(id="m3", sender="customer3@proton.me", subject="Late delivery refund", body="The package took 2 weeks. I'm requesting a refund."),
        "m4": Email(id="m4", sender="customer4@hotmail.com", subject="Forgot password", body="How do I reset my password?"),
        "m5": Email(id="m5", sender="newsletter@marketing.com", subject="Weekly deals!", body="Buy our new stuff!")
    }
    return State(
        task_id="medium",
        emails=emails,
        sent_replies={},
        folders=["inbox", "Tech", "Spam/Marketing", "Refunds_Processed"],
        current_folder="inbox",
        current_email_id=None,
        step_count=0,
        done=False
    )

def grade_medium_task(state: State) -> float:
    score = 0.0
    
    # Refunds
    refund_score = 0
    for e_id in ["m1", "m3"]:
        if state.emails[e_id].folder == "Refunds_Processed":
            refund_score += 0.25
        if e_id in state.sent_replies and "refund" in state.sent_replies[e_id].lower():
            refund_score += 0.25
    score += refund_score * 0.4
    
    # Tech
    tech_score = 0
    for e_id in ["m2", "m4"]:
        if state.emails[e_id].folder == "Tech":
            tech_score += 0.5
    score += tech_score * 0.4
    
    # Marketing
    if state.emails["m5"].folder == "Spam/Marketing":
        score += 0.2
        
    return max(0.0, min(1.0, score))

def setup_hard_task() -> State:
    emails = {
        "h1": Email(id="h1", sender="scam@phish.net", subject="URGENT: Password Reset", body="Click here to reset your password immediately or lose access."),
        "h2": Email(id="h2", sender="manager@company.com", subject="Status on Project Alpha?", body="I need an update on Project Alpha ASAP. What are the blockers? Is it deploying this week?"),
        "h3": Email(id="h3", sender="dev1@company.com", subject="Alpha update", body="Just finished the backend API for Alpha. Ready for integration."),
        "h4": Email(id="h4", sender="dev2@company.com", subject="Blocker on Alpha", body="Waiting on the design team for the frontend assets before I can proceed."),
        "h5": Email(id="h5", sender="dev3@company.com", subject="Alpha deployment", body="Deployment is scheduled for Friday. No issues on ops side."),
        "h6": Email(id="h6", sender="prince@nigeria.gov", subject="Inheritance", body="You have won 1 million dollars."),
        "h7": Email(id="h7", sender="daily@news.com", subject="Tech News", body="Today's top tech stories..."),
    }
    return State(
        task_id="hard",
        emails=emails,
        sent_replies={},
        folders=["inbox", "Trash", "Newsletters"],
        current_folder="inbox",
        current_email_id=None,
        step_count=0,
        done=False
    )

def grade_hard_task(state: State) -> float:
    score = 0.0
    
    if state.emails["h1"].folder == "Trash" and state.emails["h6"].folder == "Trash":
        score += 0.2
    elif state.emails["h1"].folder == "Trash" or state.emails["h6"].folder == "Trash":
        score += 0.1
        
    if state.emails["h7"].folder == "Newsletters":
        score += 0.1
        
    if "h2" in state.sent_replies:
        reply = state.sent_replies["h2"].lower()
        synthesis_points = 0.0
        if "backend" in reply or "api" in reply or "ready" in reply or "finished" in reply:
            synthesis_points += 0.23
        if "design" in reply or "frontend" in reply or "blocker" in reply or "waiting" in reply:
            synthesis_points += 0.23
        if "friday" in reply or "deploy" in reply:
            synthesis_points += 0.24
            
        score += synthesis_points
        
    return max(0.0, min(1.0, score))

TASKS = {
    "easy": {
        "setup": setup_easy_task,
        "grade": grade_easy_task,
        "description": "Move all HR emails to the 'HR' folder. Leave general emails in the 'inbox'."
    },
    "medium": {
        "setup": setup_medium_task,
        "grade": grade_medium_task,
        "description": "Categorize support emails into 'Tech' and 'Spam/Marketing'. Reply to refund requests and move them to 'Refunds_Processed'."
    },
    "hard": {
        "setup": setup_hard_task,
        "grade": grade_hard_task,
        "description": "Delete spam to 'Trash'. Categorize newsletter to 'Newsletters'. MOST IMPORTANTLY: Reply to the manager (h2) synthesizing the 3 dev emails about Project Alpha (status, blocker, and deployment day)."
    }
}
