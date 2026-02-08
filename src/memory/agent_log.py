
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class AgentLog(BaseModel):
    """
    Structured log entry for agent handoffs and reasoning.
    Stored in 'working_memory' collection.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interaction_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    agent_id: str
    action: str  # PLAN, RETRIEVE, DETECT_FRAUD, CRITIQUE, RESPOND
    
    input_summary: str
    output_summary: str
    reasoning: str
    confidence: float
    
    next_agent: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
