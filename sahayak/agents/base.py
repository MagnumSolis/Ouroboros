"""
Base Agent - Foundation for all Sahayak agents
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from loguru import logger

from ..adapters import LLMAdapter, ChatMessage
from ..memory import MemoryManager, MemoryEntry, MemoryType


class AgentState(str, Enum):
    """Agent execution states"""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"      # Waiting for other agents
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentContext:
    """Context passed between agents"""
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str = ""
    language: str = "en"
    modality: str = "text"  # text, audio, image
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Standard result from agent execution"""
    success: bool
    content: str
    agent_id: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    next_action: Optional[str] = None  # Suggested next step


class BaseAgent(ABC):
    """
    Base class for all Sahayak agents
    
    All agents:
    - Have a unique ID and role
    - Read/write to the Qdrant Blackboard
    - Use LLM for reasoning
    - Log their activities
    """
    
    def __init__(
        self,
        agent_id: str,
        role: str,
        llm: LLMAdapter,
        memory: MemoryManager,
        system_prompt: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.llm = llm
        self.memory = memory
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.state = AgentState.IDLE
        
        logger.info(f"Initialized agent: {agent_id} ({role})")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent"""
        return f"""You are {self.role}, part of Sahayak - a vernacular financial assistant.
Your role is to help users with financial queries, fraud detection, and scheme information.
Be helpful, accurate, and culturally aware. Support Hindi and English.
Always cite sources when providing financial advice."""
    
    @abstractmethod
    async def process(self, context: AgentContext) -> AgentResult:
        """
        Main processing logic - must be implemented by subclasses
        
        Args:
            context: Current interaction context
            
        Returns:
            AgentResult with outcome
        """
        pass
    
    async def think(
        self,
        prompt: str,
        context: Optional[AgentContext] = None,
        **kwargs
    ) -> str:
        """
        Use LLM for reasoning
        
        Args:
            prompt: The thinking prompt
            context: Optional context for history
            **kwargs: Additional LLM parameters
        """
        messages = [ChatMessage(role="system", content=self.system_prompt)]
        
        # Add history if available
        if context and context.history:
            for entry in context.history[-5:]:  # Last 5 entries
                messages.append(ChatMessage(
                    role=entry.get("role", "user"),
                    content=entry.get("content", "")
                ))
        
        messages.append(ChatMessage(role="user", content=prompt))
        
        response = await self.llm.chat(messages, **kwargs)
        return response.content
    
    async def log_to_memory(
        self,
        content: str,
        context: AgentContext,
        memory_type: MemoryType = MemoryType.AGENT_FINDING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log agent activity to the Blackboard
        
        Args:
            content: What to log
            context: Current context
            memory_type: Type of memory entry
            metadata: Additional metadata
            
        Returns:
            ID of stored entry
        """
        entry = MemoryEntry(
            interaction_id=context.interaction_id,
            agent_id=self.agent_id,
            memory_type=memory_type,
            content=content,
            metadata={
                "language": context.language,
                "source_modality": context.modality,
                **(metadata or {})
            }
        )
        
        entry_id = await self.memory.store_memory(entry)
        logger.debug(f"[{self.agent_id}] Logged: {content[:50]}...")
        
        return entry_id
    
    async def read_context(
        self,
        context: AgentContext,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Read recent context from the Blackboard"""
        return await self.memory.get_interaction_history(
            context.interaction_id,
            limit=limit
        )
    
    def set_state(self, state: AgentState):
        """Update agent state"""
        old_state = self.state
        self.state = state
        logger.debug(f"[{self.agent_id}] State: {old_state} -> {state}")
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} state={self.state}>"
