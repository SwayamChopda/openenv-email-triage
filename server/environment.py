from server.models import Action, Observation, State, EmailSummary, StepResponse
from server.tasks import TASKS
import copy

class EmailEnv:
    def __init__(self):
        self.state: State = None
        self.max_steps = 30
    
    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in TASKS:
            task_id = "easy"
        self.state = TASKS[task_id]["setup"]()
        return self._get_observation()
        
    def step(self, action: Action) -> StepResponse:
        info = {}
        reward = 0.0
        
        if self.state is None:
            self.reset()
            
        if self.state.done:
            return StepResponse(
                observation=self._get_observation(),
                reward=self._get_reward(),
                done=True,
                info={"error": "Episode already done. Call reset()."}
            )

        self.state.step_count += 1
        
        # Apply action
        try:
            if action.action == "move_email":
                if action.email_id in self.state.emails:
                    if action.destination_folder not in self.state.folders:
                        self.state.folders.append(action.destination_folder)
                    self.state.emails[action.email_id].folder = action.destination_folder
            
            elif action.action == "read_email":
                if action.email_id in self.state.emails:
                    self.state.current_email_id = action.email_id
                    self.state.emails[action.email_id].read = True
                    
            elif action.action == "reply_email":
                if action.email_id in self.state.emails:
                    self.state.sent_replies[action.email_id] = action.reply_body
                    
            elif action.action == "delete_email":
                if action.email_id in self.state.emails:
                    if "Trash" not in self.state.folders:
                        self.state.folders.append("Trash")
                    self.state.emails[action.email_id].folder = "Trash"
                    self.state.current_email_id = None
                    
            elif action.action == "change_folder":
                if action.folder_name in self.state.folders:
                    self.state.current_folder = action.folder_name
                    self.state.current_email_id = None
                else:
                    self.state.folders.append(action.folder_name)
                    self.state.current_folder = action.folder_name
                    self.state.current_email_id = None
                    
            elif action.action == "submit_task":
                self.state.done = True
                
        except Exception as e:
            info["error"] = str(e)
            
        if self.state.step_count >= self.max_steps:
            self.state.done = True
            info["timeout"] = True
            
        reward = self._get_reward()
        
        return StepResponse(
            observation=self._get_observation(),
            reward=reward,
            done=self.state.done,
            info=info
        )
        
    def _get_observation(self) -> Observation:
        # Build observation from state
        current_folder = self.state.current_folder
        emails_in_folder = []
        for e in self.state.emails.values():
            if e.folder == current_folder:
                emails_in_folder.append(EmailSummary(
                    id=e.id,
                    sender=e.sender,
                    subject=e.subject,
                    read=e.read
                ))
                
        current_email = None
        if self.state.current_email_id and self.state.current_email_id in self.state.emails:
            current_email = self.state.emails[self.state.current_email_id]
            
        return Observation(
            current_folder=current_folder,
            emails_in_folder=emails_in_folder,
            current_email=current_email,
            folders=self.state.folders,
            notification=f"Step {self.state.step_count}/{self.max_steps}"
        )
        
    def _get_reward(self) -> float:
        if self.state is None:
            return 0.0
        grader = TASKS[self.state.task_id]["grade"]
        return grader(self.state)
