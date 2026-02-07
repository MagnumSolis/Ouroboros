
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from ..memory.agent_log import AgentLog

class MasterContext(BaseModel):
    """
    Shared context object passed between agents.
    Represents the "Blackboard" state for a single interaction.
    """
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Input
    user_input: str
    modality: str = "text" # text, audio
    language: str = "hi"
    emotion: Optional[str] = None
    
    # Planner State
    execution_plan: Optional[str] = None # XML/Markdown blueprint
    current_step: int = 0
    
    # Agent Results
    perception_result: Dict[str, Any] = Field(default_factory=dict)
    retrieval_result: Dict[str, Any] = Field(default_factory=dict)
    fraud_result: Dict[str, Any] = Field(default_factory=dict)
    critic_result: Dict[str, Any] = Field(default_factory=dict)
    final_response: Optional[str] = None
    
    # Trace
    logs: List[AgentLog] = Field(default_factory=list)
    
    def add_log(self, log: AgentLog):
        """Add a log entry to the trace"""
        self.logs.append(log)
        
    def to_markdown(self) -> str:
        """Export context as markdown for dashboard/viewing"""
        md = f"# Interaction: {self.interaction_id}\n"
        md += f"**Time**: {self.timestamp}\n"
        md += f"**Input**: {self.user_input} ({self.modality}, {self.language})\n"
        if self.emotion:
            md += f"**Emotion**: {self.emotion}\n"
            
        if self.execution_plan:
            md += f"\n## 📋 Execution Plan\n```xml\n{self.execution_plan}\n```\n"
            
        md += "\n## 🧠 Agent Reasoning Trace\n"
        for log in self.logs:
            md += f"### {log.agent_id} ({log.action})\n"
            md += f"- **Input**: {log.input_summary}\n"
            md += f"- **Reasoning**: {log.reasoning}\n"
            md += f"- **Output**: {log.output_summary}\n"
            md += "---\n"
            
        return md
