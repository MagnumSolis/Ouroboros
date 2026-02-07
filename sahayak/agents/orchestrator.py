"""
Orchestrator Agent - Plans and coordinates other agents

The "Brain" of Sahayak as per research paper:
- Decomposes high-level queries into actionable steps
- Dispatches tasks to specialized agents
- Collects results and synthesizes responses
- Uses Critic agent for validation before final output
"""

from typing import List, Dict, Any, Optional
import json

from loguru import logger

from .base import BaseAgent, AgentContext, AgentResult, AgentState
from ..adapters import LLMAdapter
from ..memory import MemoryManager, MemoryType
from ..context.master_context import MasterContext
from ..memory.agent_log import AgentLog


ORCHESTRATOR_PROMPT = """You are the Orchestrator Agent for Sahayak - The Vernacular Financial Sentinel.

Your job is to:
1. Analyze user queries (from text, audio, or image)
2. Determine which agents/actions are needed
3. Create an execution plan
4. Coordinate the response

Available actions:
- PROCESS_INPUT: Process multimodal input (audio/image) through Perception agent
- CHECK_FRAUD: Analyze for potential fraud/scam indicators (prioritize for safety)
- RETRIEVE_KNOWLEDGE: Search knowledge base for schemes, policies, FAQs
- RETRIEVE_CONTEXT: Get relevant past interactions
- VALIDATE_RESPONSE: Use Critic agent to verify accuracy (recommended for financial advice)

Output your plan as JSON:
{
    "intent": "brief description of user intent",
    "language": "detected language (hi/en/mixed)",
    "urgency": "low/medium/high",
    "plan": [
        {"action": "ACTION_NAME", "reason": "why this action"},
        ...
    ],
    "requires_fraud_check": true/false,
    "requires_validation": true/false
}

IMPORTANT:
- Always CHECK_FRAUD if query mentions: calls, messages, OTP, money requests, KYC, urgent, prize
- Always RETRIEVE_KNOWLEDGE for: schemes, loans, subsidies, RBI, bank, eligibility
- Always VALIDATE_RESPONSE for: specific financial figures, interest rates, eligibility criteria
- For audio/image input, first PROCESS_INPUT to get text content"""


class OrchestratorAgent(BaseAgent):
    """
    Central planning agent that coordinates the workflow
    
    Follows Blackboard Pattern:
    - Writes plans to working_memory
    - Reads results from other agents via memory
    - Synthesizes final response with validation
    """
    
    def __init__(
        self,
        llm: LLMAdapter,
        memory: MemoryManager
    ):
        super().__init__(
            agent_id="orchestrator",
            role="Orchestrator Agent",
            llm=llm,
            memory=memory,
            system_prompt=ORCHESTRATOR_PROMPT
        )
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register a specialized agent"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    def register_agents(self, agents: List[BaseAgent]):
        """Register multiple agents"""
        for agent in agents:
            self.register_agent(agent)
    
    async def process(self, context: AgentContext) -> AgentResult:
        """
        Orchestrate the agent workflow using MasterContext and XML Planning
        """
        self.set_state(AgentState.PROCESSING)

        try:
            # 0. Handle multimodal input FIRST (transcribe audio, extract text from images)
            processed_context = await self._handle_multimodal(context)
            
            # 1. Initialize Master Context with PROCESSED input
            master_context = MasterContext(
                interaction_id=processed_context.interaction_id,
                user_input=processed_context.user_input,
                language=processed_context.language,
                modality=processed_context.modality,
                emotion=processed_context.metadata.get("perception_analysis", {}).get("emotion")
            )

            # 3. Create Execution Plan (XML Blueprint)
            plan_xml = await self._create_execution_plan(master_context)
            master_context.execution_plan = plan_xml
            
            # Log Planning Step
            log = AgentLog(
                interaction_id=master_context.interaction_id,
                agent_id=self.agent_id,
                action="PLAN",
                input_summary=master_context.user_input[:50],
                output_summary="XML Plan Generated",
                reasoning="Generated plan based on user intent and emotion",
                confidence=0.9,
                metadata={"plan": plan_xml}
            )
            await self.memory.log_agent_action(log)
            master_context.add_log(log)

            # 4. Execute Plan (Dynamic Routing)
            # Parse XML (Simple parsing for demo)
            steps = []
            if 'agent="fraud"' in plan_xml: steps.append("fraud")
            if 'agent="retrieval"' in plan_xml: steps.append("retrieval")
            if 'agent="critic"' in plan_xml: steps.append("critic")
            
            # Default fallback if plan fails
            if not steps: steps = ["retrieval", "critic"]
            
            response_text = ""
            is_fraud = False
            
            for step in steps:
                curr_agent = self.agents.get(step)
                if not curr_agent: continue
                
                # Execute Agent
                # Note: In a full implementation, we'd pass MasterContext directly
                # For now, we adapt it to AgentContext to keep existing agents working
                step_context = AgentContext(
                    interaction_id=master_context.interaction_id,
                    user_input=master_context.user_input,
                    language=master_context.language,
                    metadata={"master_context": master_context.model_dump()}
                )
                
                result = await curr_agent.process(step_context)
                
                # Log Agent Result
                if result.success:
                    if step == "fraud":
                        if result.metadata.get("is_fraud"):
                            is_fraud = True
                        master_context.fraud_result = result.metadata
                    elif step == "retrieval":
                        response_text = result.content
                        master_context.retrieval_result = {"content": result.content}
                    elif step == "critic":
                        # If critic changes response
                        if result.content != "VALID" and len(result.content) > 10:
                            response_text = result.content
                        master_context.critic_result = {"validation": result.content}

            # 5. Final Response Construction
            if is_fraud:
                 final_response = "⚠️ WARNING: This appears to be a fraud attempt. " + response_text
            else:
                 final_response = response_text

            master_context.final_response = final_response
            
            self.set_state(AgentState.COMPLETED)

            return AgentResult(
                success=True,
                content=final_response,
                agent_id=self.agent_id,
                metadata={
                    "plan": plan_xml,
                    "master_context": master_context.model_dump(), 
                    "is_fraud": is_fraud
                }
            )

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            self.set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                content=f"Error in processing: {e}",
                agent_id=self.agent_id,
                metadata={}
            )

    async def _create_execution_plan(self, context: MasterContext) -> str:
        """Generate XML blueprint for execution"""
        prompt = f"""
        You are the Orchestrator for Sahayak. Create a multi-agent execution plan in XML.
        
        User Input: "{context.user_input}"
        Language: {context.language}
        Emotion: {context.emotion or 'Neutral'}
        
        Available Agents:
        - perception: (Already run) Handles audio/emotion
        - fraud: Detects scams, OTP requests, money demands
        - retrieval: Searches schemes/policies database
        - critic: Validates answers
        
        Rules:
        1. If input involves MONEY, OTP, KYC, BANK, or FEAR/URGENCY -> Invoke 'fraud' agent FIRST.
        2. If asking about SCHEMES, LOANS, RULES -> Invoke 'retrieval' agent.
        3. ALWAYS end with 'critic' agent for safety.
        
        Output format (XML ONLY):
        <plan>
            <step order="1" agent="fraud" reason="Check for financial risk"/>
            <step order="2" agent="retrieval" reason="Fetch scheme details"/>
            <step order="3" agent="critic" reason="Verify info"/>
        </plan>
        """
        
        # We need to adapt MasterContext to AgentContext for the LLM call
        temp_context = AgentContext(
            interaction_id=context.interaction_id,
            user_input=context.user_input
        )
        
        # LLMAdapter.chat() expects List[ChatMessage]
        from sahayak.adapters.llm import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]
        
        llm_response = await self.llm.chat(messages, temperature=0.1)
        response = llm_response.content
        # naive cleanup
        xml = response.replace("```xml", "").replace("```", "").strip()
        return xml
    
    async def _handle_multimodal(self, context: AgentContext) -> AgentContext:
        """Process audio/image input through Perception agent"""
        
        if context.modality == "text":
            return context
        
        if "perception" not in self.agents:
            logger.warning("Perception agent not registered, skipping multimodal processing")
            return context
        
        perception_result = await self.agents["perception"].process(context)
        
        if perception_result.success:
            # Create new context with transcribed/extracted text
            return AgentContext(
                interaction_id=context.interaction_id,
                user_input=perception_result.content,
                language=perception_result.metadata.get("language", context.language),
                modality=context.modality,  # Keep original modality for logging
                metadata={
                    **context.metadata,
                    "original_input": context.user_input,
                    "perception_analysis": perception_result.metadata
                },
                history=context.history
            )
        
        return context
    
    async def _create_plan(self, context: AgentContext) -> Dict[str, Any]:
        """Use LLM to create execution plan"""
        
        prompt = f"""Analyze this user query and create an execution plan:

User Query: {context.user_input}
Detected Language: {context.language}
Input Type: {context.modality}
Additional Context: {json.dumps(context.metadata.get('perception_analysis', {}), default=str)[:300]}
Emotion Analysis: {context.metadata.get('perception_analysis', {}).get('emotion', 'unknown')} (conf: {context.metadata.get('perception_analysis', {}).get('emotion_confidence', 0):.2f})

Create a JSON plan as specified. Be strategic about which agents to invoke.
If emotion is 'fear' or 'angry', prioritize CHECK_FRAUD and raise urgency."""

        response = await self.think(prompt, context, temperature=0.3)
        
        # Parse JSON from response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                plan = json.loads(response[start:end])
                # Ensure required fields
                plan.setdefault("intent", "general query")
                plan.setdefault("language", context.language)
                plan.setdefault("urgency", "low")
                plan.setdefault("plan", [])
                plan.setdefault("requires_fraud_check", False)
                plan.setdefault("requires_validation", False)
            else:
                plan = self._default_plan(context)
        except json.JSONDecodeError:
            logger.warning("Failed to parse plan JSON, using default")
            plan = self._default_plan(context)
        
        return plan
    
    def _default_plan(self, context: AgentContext) -> Dict[str, Any]:
        """Create default plan when LLM parsing fails"""
        return {
            "intent": "general query",
            "language": context.language,
            "urgency": "low",
            "plan": [
                {"action": "CHECK_FRAUD", "reason": "safety check"},
                {"action": "RETRIEVE_KNOWLEDGE", "reason": "default search"}
            ],
            "requires_fraud_check": True,
            "requires_validation": False
        }
    
    async def _execute_plan(
        self,
        plan: Dict[str, Any],
        context: AgentContext
    ) -> List[Dict[str, Any]]:
        """Execute each step in the plan"""
        
        results = []
        
        # Always run fraud check if flagged
        if plan.get("requires_fraud_check", False) and "fraud" in self.agents:
            if not any(s.get("action") == "CHECK_FRAUD" for s in plan.get("plan", [])):
                plan["plan"].insert(0, {"action": "CHECK_FRAUD", "reason": "security priority"})
        
        for step in plan.get("plan", []):
            action = step.get("action")
            
            try:
                if action == "CHECK_FRAUD" and "fraud" in self.agents:
                    result = await self.agents["fraud"].process(context)
                    results.append({"action": action, "result": result, "success": result.success})
                    
                elif action == "RETRIEVE_KNOWLEDGE" and "retrieval" in self.agents:
                    result = await self.agents["retrieval"].process(context)
                    results.append({"action": action, "result": result, "success": result.success})
                    
                elif action == "PROCESS_INPUT" and "perception" in self.agents:
                    # Already handled in _handle_multimodal, skip
                    pass
                    
                elif action == "RETRIEVE_CONTEXT":
                    history = await self.read_context(context)
                    results.append({"action": action, "result": history, "success": True})
                    
            except Exception as e:
                logger.error(f"Error executing {action}: {e}")
                results.append({"action": action, "error": str(e), "success": False})
        
        return results
    
    async def _synthesize_response(
        self,
        context: AgentContext,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> str:
        """Combine results into final response"""
        
        # Build context from results
        fraud_alert = None
        knowledge_context = []
        
        for item in results:
            action = item.get("action")
            result = item.get("result")
            
            if action == "CHECK_FRAUD" and isinstance(result, AgentResult):
                if result.metadata.get("is_fraud"):
                    fraud_alert = result.content
                    
            elif action == "RETRIEVE_KNOWLEDGE" and isinstance(result, AgentResult):
                knowledge_context.append(result.content)
                
            elif action == "RETRIEVE_CONTEXT" and isinstance(result, list):
                context_summary = "\n".join([
                    f"- {r.get('content', '')[:100]}" 
                    for r in result[:3]
                ])
                if context_summary:
                    knowledge_context.append(f"Past context:\n{context_summary}")
        
        # Generate final response
        prompt = f"""Generate a helpful response for the user.

User Query: {context.user_input}
Language to respond in: {plan.get('language', 'en')}
Urgency: {plan.get('urgency', 'low')}

{"⚠️ FRAUD ALERT: " + fraud_alert if fraud_alert else ""}

Relevant Information:
{chr(10).join(knowledge_context) if knowledge_context else "No specific information found."}

Instructions:
- If there's a fraud alert, PRIORITIZE warning the user clearly
- Use the appropriate language (Hindi/English/mixed)
- Be culturally appropriate and respectful
- For financial info, mention that user should verify with official sources
- Be concise but complete"""

        response = await self.think(prompt, context, temperature=0.7)
        
        # Prepend fraud warning if detected
        if fraud_alert:
            response = f"⚠️ **सावधान / Warning**: {fraud_alert}\n\n{response}"
        
        return response
    
    async def _validate_response(
        self,
        response: str,
        results: List[Dict[str, Any]],
        context: AgentContext
    ) -> tuple[str, Dict[str, Any]]:
        """Use Critic agent to validate response"""
        
        # Gather evidence from retrieval results
        evidence = []
        for item in results:
            if item.get("action") == "RETRIEVE_KNOWLEDGE":
                result = item.get("result")
                if isinstance(result, AgentResult) and result.metadata:
                    evidence.append(result.metadata)
        
        # Create validation context
        validation_context = AgentContext(
            interaction_id=context.interaction_id,
            user_input=context.user_input,
            language=context.language,
            metadata={
                "proposed_response": response,
                "retrieved_evidence": evidence
            }
        )
        
        validation_result = await self.agents["critic"].process(validation_context)
        
        if validation_result.success:
            validation = validation_result.metadata
            
            # If validation fails with corrections, apply them
            if not validation.get("approved", True):
                suggestions = validation.get("suggestions", [])
                if suggestions:
                    response += f"\n\n*(Note: {suggestions[0]})*"
            
            return response, validation
        
        return response, {"approved": True, "note": "Validation skipped"}

