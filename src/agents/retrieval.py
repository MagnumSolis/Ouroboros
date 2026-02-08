"""
Retrieval Agent - Knowledge base and context search
"""

from typing import List, Dict, Any, Optional

from loguru import logger

from .base import BaseAgent, AgentContext, AgentResult, AgentState
from ..adapters import LLMAdapter
from ..memory import MemoryManager, MemoryType


RETRIEVAL_PROMPT = """You are the Retrieval Agent for Sahayak.

Your job is to find relevant information from:
- Government schemes (PM Jan Dhan, Mudra, etc.)
- RBI policies and guidelines
- Financial literacy FAQs
- Past user interactions

When retrieving information:
1. Focus on accuracy and relevance
2. Cite sources when available
3. Summarize in user's language (Hindi or English)
4. Highlight key eligibility criteria and benefits"""


class RetrievalAgent(BaseAgent):
    """
    Knowledge retrieval agent
    
    Searches:
    - knowledge_base: Policies, schemes, FAQs
    - episodic_memory: Past interactions
    """
    
    def __init__(
        self,
        llm: LLMAdapter,
        memory: MemoryManager
    ):
        super().__init__(
            agent_id="retrieval",
            role="Retrieval Agent",
            llm=llm,
            memory=memory,
            system_prompt=RETRIEVAL_PROMPT
        )
    
    async def process(self, context: AgentContext) -> AgentResult:
        """
        Search knowledge base and synthesize relevant information
        """
        self.set_state(AgentState.PROCESSING)
        
        try:
            # Search knowledge base (no language filter - allows all ingested docs to be found)
            knowledge_results = await self.memory.retrieve_knowledge(
                query=context.user_input,
                language=None,  # Don't filter by language; ingested docs may lack this metadata
                limit=5
            )
            
            # Search episodic memory for context
            episodic_results = await self.memory.search(
                collection="episodic_memory",
                query=context.user_input,
                limit=3
            )
            
            # Combine and summarize
            all_results = knowledge_results + episodic_results
            
            if not all_results:
                self.set_state(AgentState.COMPLETED)
                return AgentResult(
                    success=True,
                    content="No specific information found for this query.",
                    agent_id=self.agent_id,
                    confidence=0.3
                )
            
            # Use LLM to synthesize
            context_text = "\n\n".join([
                f"[{r.get('source', r.get('title', 'Document'))}] (score: {r.get('score', 0):.2f})\n{r.get('content', '')[:500]}"
                for r in all_results[:5]
            ])
            
            summary = await self._summarize_results(context, context_text)
            
            # Log finding
            await self.log_to_memory(
                content=f"Retrieved {len(all_results)} relevant documents",
                context=context,
                metadata={"result_count": len(all_results)}
            )
            
            self.set_state(AgentState.COMPLETED)
            
            return AgentResult(
                success=True,
                content=summary,
                agent_id=self.agent_id,
                confidence=max(r.get("score", 0) for r in all_results) if all_results else 0,
                metadata={"sources": [r.get("source", r.get("title", "Unknown")) for r in all_results[:3]]}
            )
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            self.set_state(AgentState.ERROR)
            
            return AgentResult(
                success=False,
                content="Could not search knowledge base.",
                agent_id=self.agent_id,
                metadata={"error": str(e)}
            )
    
    async def _summarize_results(
        self,
        context: AgentContext,
        context_text: str
    ) -> str:
        """Summarize retrieved results"""
        
        prompt = f"""Based on the following retrieved information, provide a helpful summary for the user.

User Query: {context.user_input}
Language: {context.language}

Retrieved Information:
{context_text}

Provide a concise, accurate summary. Include:
- Key relevant points
- Eligibility criteria (if applicable)
- How to apply/proceed (if applicable)
- Source citations"""

        return await self.think(prompt, context, temperature=0.5)
    
    async def search_schemes(
        self,
        query: str,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Search specifically for government schemes"""
        return await self.memory.retrieve_knowledge(
            query=query,
            doc_type="scheme",
            language=language
        )
    
    async def search_faqs(
        self,
        query: str,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Search FAQs"""
        return await self.memory.retrieve_knowledge(
            query=query,
            doc_type="faq",
            language=language
        )
