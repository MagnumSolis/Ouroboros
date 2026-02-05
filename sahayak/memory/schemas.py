"""
Memory Schemas - Pydantic models for Blackboard entries
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class MemoryType(str, Enum):
    """Types of memory entries in the Blackboard"""
    USER_INPUT = "user_input"          # User queries (text, audio, image)
    AGENT_FINDING = "agent_finding"    # Agent discoveries/results
    TOOL_OUTPUT = "tool_output"        # Tool execution results
    PLAN_STEP = "plan_step"            # Orchestrator plan steps
    ALERT = "alert"                    # Fraud alerts, warnings
    CONTEXT = "context"                # Retrieved context
    FEEDBACK = "feedback"              # Critic feedback


class MemoryEntry(BaseModel):
    """Universal schema for all Blackboard entries"""
    
    # Core identifiers
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interaction_id: str              # Groups entries in same interaction
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Source information
    agent_id: str                    # Which agent wrote this
    memory_type: MemoryType
    
    # Content
    content: str                     # Human-readable content
    
    # Metadata for filtering and retrieval
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "confidence_score": 0.0,
        "source_modality": "text",   # text, audio, image
        "action_required": False,
        "severity": 0,               # 0-10 for alerts
        "language": "en",
        "region": None,
        "tags": [],
    })
    
    class Config:
        use_enum_values = True


class FraudPattern(BaseModel):
    """Schema for fraud pattern entries"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str                # phishing, vishing, smishing, etc.
    description: str
    indicators: List[str]            # Key phrases/patterns
    severity: int = Field(ge=0, le=10)
    source: Optional[str] = None     # Where this pattern was documented
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocument(BaseModel):
    """Schema for knowledge base documents"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    document_type: str               # policy, scheme, guideline, faq
    source_url: Optional[str] = None
    language: str = "en"
    
    # For chunked documents
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    parent_id: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkingMemoryEntry(BaseModel):
    """Schema for working memory (current task state)"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interaction_id: str
    task_type: str                   # check_fraud, retrieve_info, etc.
    status: str = "pending"          # pending, in_progress, completed, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Task data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    assigned_agent: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
