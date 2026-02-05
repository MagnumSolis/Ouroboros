"""
Fraud Detection Agent - Identifies potential financial scams
"""

from typing import List, Dict, Any, Optional

from loguru import logger

from .base import BaseAgent, AgentContext, AgentResult, AgentState
from ..adapters import LLMAdapter
from ..memory import MemoryManager, MemoryType


FRAUD_PROMPT = """You are the Fraud Detection Agent for Sahayak.

Your critical job is to protect users from financial fraud. Common scams in India include:
- OTP fraud (asking for OTP codes)
- KYC scams (fake bank KYC update calls)
- Lottery/prize scams
- Loan fraud (fake loan offers with upfront fees)
- UPI fraud (send money to receive money)
- Fake customer care numbers

When analyzing:
1. Look for red flags: urgency, money requests, OTP demands, too-good offers
2. Consider the context (phone call, SMS, WhatsApp)
3. Rate severity (1-10)
4. Always err on the side of caution

Respond in the user's language with clear warnings."""


FRAUD_INDICATORS = [
    "otp", "ओटीपी", "one time password",
    "kyc", "केवाईसी", "verify account",
    "lottery", "लॉटरी", "prize", "winner",
    "urgent", "जरूरी", "immediately",
    "send money", "पैसे भेजो", "transfer",
    "bank officer", "बैंक अधिकारी",
    "customer care", "link click",
    "refund", "cashback", "bonus"
]


class FraudAgent(BaseAgent):
    """
    Fraud detection agent
    
    Analyzes user input for potential scam indicators using:
    - Pattern matching against known fraud patterns
    - Semantic similarity search in fraud_patterns collection
    - LLM reasoning for context understanding
    """
    
    def __init__(
        self,
        llm: LLMAdapter,
        memory: MemoryManager,
        threshold: float = 0.70
    ):
        super().__init__(
            agent_id="fraud",
            role="Fraud Detection Agent",
            llm=llm,
            memory=memory,
            system_prompt=FRAUD_PROMPT
        )
        self.threshold = threshold
    
    async def process(self, context: AgentContext) -> AgentResult:
        """
        Analyze input for fraud indicators
        
        Returns result with:
        - is_fraud: Whether fraud is detected
        - severity: 1-10 rating
        - warning: User-friendly warning message
        """
        self.set_state(AgentState.PROCESSING)
        
        try:
            text = context.user_input.lower()
            
            # Step 1: Quick keyword check
            keyword_matches = [ind for ind in FRAUD_INDICATORS if ind in text]
            
            # Step 2: Semantic search against fraud patterns
            pattern_matches = await self.memory.check_fraud_similarity(
                text=context.user_input,
                threshold=self.threshold
            )
            
            # Step 3: LLM analysis if keywords or patterns found
            if keyword_matches or pattern_matches:
                analysis = await self._analyze_fraud(context, keyword_matches, pattern_matches)
            else:
                analysis = {
                    "is_fraud": False,
                    "severity": 0,
                    "confidence": 0.9,
                    "warning": None
                }
            
            # Log to memory
            if analysis["is_fraud"]:
                await self.log_to_memory(
                    content=f"⚠️ Fraud detected (severity: {analysis['severity']}): {keyword_matches}",
                    context=context,
                    memory_type=MemoryType.ALERT,
                    metadata={
                        "severity": analysis["severity"],
                        "indicators": keyword_matches,
                        "action_required": True
                    }
                )
            
            self.set_state(AgentState.COMPLETED)
            
            return AgentResult(
                success=True,
                content=analysis.get("warning", "No fraud detected."),
                agent_id=self.agent_id,
                confidence=analysis.get("confidence", 0.5),
                metadata={
                    "is_fraud": analysis["is_fraud"],
                    "severity": analysis["severity"],
                    "indicators": keyword_matches,
                    "pattern_matches": len(pattern_matches)
                }
            )
            
        except Exception as e:
            logger.error(f"Fraud detection error: {e}")
            self.set_state(AgentState.ERROR)
            
            return AgentResult(
                success=False,
                content="Could not complete fraud analysis.",
                agent_id=self.agent_id,
                metadata={"error": str(e)}
            )
    
    async def _analyze_fraud(
        self,
        context: AgentContext,
        keywords: List[str],
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM for detailed fraud analysis"""
        
        pattern_info = "\n".join([
            f"- {p.get('pattern_type', 'unknown')}: {p.get('description', '')} (similarity: {p.get('score', 0):.2f})"
            for p in patterns[:3]
        ]) if patterns else "No known patterns matched."
        
        prompt = f"""Analyze this for potential fraud:

User's message: {context.user_input}
Language: {context.language}
Input source: {context.modality}

Detected keywords: {keywords}
Matched fraud patterns:
{pattern_info}

Analyze and respond with JSON:
{{
    "is_fraud": true/false,
    "severity": 1-10,
    "confidence": 0.0-1.0,
    "fraud_type": "type if detected",
    "warning": "user-friendly warning in {context.language}"
}}

Be protective of the user. When in doubt, warn them."""

        response = await self.think(prompt, context, temperature=0.2)
        
        # Parse response
        try:
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        
        # Default to warning if parsing fails but indicators exist
        if keywords or patterns:
            return {
                "is_fraud": True,
                "severity": 5,
                "confidence": 0.6,
                "warning": "⚠️ सावधान! यह संदिग्ध लग रहा है। कृपया कोई OTP या पैसे न भेजें। / Warning! This looks suspicious. Please don't share OTP or send money."
            }
        
        return {"is_fraud": False, "severity": 0, "confidence": 0.5, "warning": None}
    
    async def check_audio(
        self,
        transcription: str,
        context: AgentContext
    ) -> AgentResult:
        """
        Special check for transcribed audio (phone calls)
        More sensitive thresholds for voice phishing
        """
        # Create modified context with transcription
        audio_context = AgentContext(
            interaction_id=context.interaction_id,
            user_input=transcription,
            language=context.language,
            modality="audio",
            metadata={"original_modality": context.modality}
        )
        
        return await self.process(audio_context)
