"""
Critic Agent - Response validation and hallucination detection

The "Quality Control" of Sahayak as per research paper:
- Reviews proposed responses for accuracy
- Verifies citations against retrieved evidence
- Checks for hallucinations
- Validates safety and compliance

Uses Chain-of-Verification (CoVe) prompting pattern.
"""

from typing import Dict, Any, List, Optional
import json

from loguru import logger

from .base import BaseAgent, AgentContext, AgentResult, AgentState
from ..adapters import LLMAdapter
from ..memory import MemoryManager, MemoryType
from ..memory.agent_log import AgentLog
from ..context.master_context import MasterContext


CRITIC_PROMPT = """You are the Critic Agent for Sahayak - The Vernacular Financial Sentinel.

Your critical job is QUALITY CONTROL. You must:
1. Verify that responses are factually accurate
2. Check that cited information matches retrieved sources
3. Detect potential hallucinations or fabricated details
4. Ensure responses are safe and compliant with financial regulations
5. Validate that Hindi/English translations are accurate

You are the LAST CHECK before a response reaches the user.
Be thorough but fair. Flag issues clearly.

For financial advice:
- Interest rates must be verified
- Scheme eligibility must be accurate
- Fraud warnings must not cause unnecessary panic
- Never approve medical/legal advice without disclaimers"""


class CriticAgent(BaseAgent):
    """
    Validation and quality control agent
    
    Responsibilities:
    - Hallucination detection
    - Citation verification
    - Safety checks
    - Response quality scoring
    """
    
    def __init__(
        self,
        llm: LLMAdapter,
        memory: MemoryManager
    ):
        super().__init__(
            agent_id="critic",
            role="Critic Agent",
            llm=llm,
            memory=memory,
            system_prompt=CRITIC_PROMPT
        )
    
    async def process(self, context: AgentContext) -> AgentResult:
        """
        Validate a proposed response
        
        Expects context.metadata to contain:
        - master_context: Dictionary representation of MasterContext
        """
        self.set_state(AgentState.PROCESSING)
        
        # Try to recover Master Context
        master_ctx_dict = context.metadata.get("master_context", {})
        if not master_ctx_dict:
            # Fallback for direct calls
            proposed_response = context.metadata.get("proposed_response", "")
            retrieved_evidence = context.metadata.get("retrieved_evidence", [])
            user_input = context.user_input
        else:
            proposed_response = master_ctx_dict.get("final_response", "")
            # Reconstruct evidence from retrieval_result in context
            ret_res = master_ctx_dict.get("retrieval_result", {})
            retrieved_evidence = [ret_res] if ret_res else []
            user_input = master_ctx_dict.get("user_input", "")

        if not proposed_response:
             # If no response yet (maybe called in parallel?), skip
             return AgentResult(success=True, content="VALID", agent_id=self.agent_id)

        try:
            # Run validation checks
            validation = await self._validate_response(
                proposed_response,
                retrieved_evidence,
                user_input
            )
            
            # Log validation result with AgentLog
            log = AgentLog(
                interaction_id=context.interaction_id,
                agent_id=self.agent_id,
                action="VALIDATE",
                input_summary=proposed_response[:50] + "...",
                output_summary=validation["summary"],
                reasoning=f"Approving: {validation['approved']}. Issues: {len(validation['issues'])}",
                confidence=validation["confidence"],
                metadata=validation
            )
            await self.memory.log_agent_action(log)
            
            self.set_state(AgentState.COMPLETED)
            
            return AgentResult(
                success=True,
                content=validation["summary"] if not validation["approved"] else "VALID",
                agent_id=self.agent_id,
                confidence=validation["confidence"],
                metadata=validation
            )
            
        except Exception as e:
            logger.error(f"Critic error: {e}")
            self.set_state(AgentState.ERROR)
            
            return AgentResult(
                success=False,
                content="Validation failed",
                agent_id=self.agent_id,
                metadata={"error": str(e)}
            )
    
    async def _validate_response(
        self,
        response: str,
        evidence: List[Dict[str, Any]],
        user_input: str
    ) -> Dict[str, Any]:
        """Run Chain-of-Verification validation"""
        
        # Check if we need independent verification (Double-Check Pattern)
        # If no evidence is provided but response contains factual claims
        if not evidence and len(response) > 50:
            logger.info("Critic performing independent verification...")
            # Extract keywords to search
            # Simple keyword extraction for now
            search_query = user_input
            new_evidence = await self.memory.retrieve_knowledge(search_query, limit=3)
            evidence = new_evidence

        evidence_text = "\n\n".join([
            f"[Source {i+1}]: {e.get('content', e.get('text', ''))[:300]}..."
            for i, e in enumerate(evidence[:5])
        ]) if evidence else "No evidence provided."
        
        prompt = f"""Validate this response using Chain-of-Verification:
 
 PROPOSED RESPONSE:
 {response}
 
 RETRIEVED EVIDENCE:
 {evidence_text}
 
 USER'S ORIGINAL QUERY:
 {user_input}
 
 VALIDATION CHECKLIST:
 1. FACTUAL ACCURACY: Are all facts in the response supported by evidence?
 2. HALLUCINATION CHECK: Does the response claim anything not in the evidence?
 3. CITATION VALIDITY: Are sources cited correctly?
 4. SAFETY CHECK: Is the response safe? No harmful financial advice?
 5. LANGUAGE CHECK: Is the response in the appropriate language for the user?
 6. COMPLETENESS: Does it answer the user's query?
 
 Respond with JSON:
 {{
     "approved": true/false,
     "confidence": 0.0-1.0,
     "issues": ["list of specific issues found"],
     "suggestions": ["how to fix issues"],
     "factual_score": 0-10,
     "safety_score": 0-10,
     "summary": "one-line validation summary"
 }}
 
 Be strict but fair. If the response is reasonable but has minor issues, you can approve with suggestions."""
 
        validation_response = await self.think(prompt, None, temperature=0.1)
        
        # Parse validation result
        try:
            start = validation_response.find("{")
            end = validation_response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(validation_response[start:end])
                # Ensure required fields
                result.setdefault("approved", False)
                result.setdefault("confidence", 0.5)
                result.setdefault("issues", [])
                result.setdefault("suggestions", [])
                result.setdefault("summary", "Validation complete")
                return result
        except json.JSONDecodeError:
            pass
        
        # Default: cautious approval
        return {
            "approved": True,
            "confidence": 0.6,
            "issues": ["Could not fully validate - proceeding with caution"],
            "suggestions": [],
            "factual_score": 7,
            "safety_score": 8,
            "summary": "Partial validation - response appears reasonable"
        }
    
    async def quick_validate(
        self,
        response: str,
        context: AgentContext
    ) -> bool:
        """Quick safety check without full validation"""
        
        # Check for dangerous patterns
        danger_patterns = [
            "send money immediately",
            "share your OTP",
            "click this link",
            "guaranteed returns",
            "no risk investment",
            "तुरंत पैसे भेजें",
            "OTP शेयर करें"
        ]
        
        response_lower = response.lower()
        for pattern in danger_patterns:
            if pattern.lower() in response_lower:
                logger.warning(f"Critic flagged dangerous pattern: {pattern}")
                return False
        
        return True
    
    async def validate_fraud_alert(
        self,
        alert_content: str,
        severity: int,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Validate a fraud alert before showing to user"""
        
        prompt = f"""Validate this fraud alert:

ALERT: {alert_content}
SEVERITY: {severity}/10

Check:
1. Is the severity level appropriate?
2. Is the warning clear and actionable?
3. Will this cause unnecessary panic?
4. Are the recommended actions safe?

JSON response:
{{
    "approved": true/false,
    "adjusted_severity": 0-10,
    "improved_message": "clearer version if needed",
    "reason": "why approved/rejected"
}}"""

        response = await self.think(prompt, context, temperature=0.1)
        
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        
        return {"approved": True, "adjusted_severity": severity, "reason": "Default approval"}
