from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Dict

# Environment Entities
class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    folder: str = "inbox"
    read: bool = False

# OBSERVATION
class EmailSummary(BaseModel):
    id: str
    sender: str
    subject: str
    read: bool

class Observation(BaseModel):
    current_folder: str
    emails_in_folder: List[EmailSummary]
    current_email: Optional[Email]
    folders: List[str]
    notification: str = ""

# STATE (Internal representation, returned by /state endpoint)
class State(BaseModel):
    task_id: str
    emails: Dict[str, Email]
    sent_replies: Dict[str, str] # email_id -> reply_body
    folders: List[str]
    current_folder: str
    current_email_id: Optional[str]
    step_count: int
    done: bool

# ACTIONS (Union of specific actions)
class MoveEmailAction(BaseModel):
    action: Literal["move_email"] = "move_email"
    email_id: str
    destination_folder: str

class ReadEmailAction(BaseModel):
    action: Literal["read_email"] = "read_email"
    email_id: str

class ReplyEmailAction(BaseModel):
    action: Literal["reply_email"] = "reply_email"
    email_id: str
    reply_body: str

class DeleteEmailAction(BaseModel):
    action: Literal["delete_email"] = "delete_email"
    email_id: str

class ChangeFolderAction(BaseModel):
    action: Literal["change_folder"] = "change_folder"
    folder_name: str

class SubmitTaskAction(BaseModel):
    action: Literal["submit_task"] = "submit_task"

Action = Union[
    MoveEmailAction,
    ReadEmailAction,
    ReplyEmailAction,
    DeleteEmailAction,
    ChangeFolderAction,
    SubmitTaskAction
]

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
