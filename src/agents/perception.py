"""
Perception Agent - Multimodal input processing (Audio, Image)

The "Eyes and Ears" of Sahayak as per research paper:
- Processes non-text data into vectors
- Converts audio to text + acoustic vectors
- OCRs images for text extraction
- Writes vectorized representations to episodic_memory
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path

from loguru import logger

from .base import BaseAgent, AgentContext, AgentResult, AgentState
from ..adapters import LLMAdapter, AudioProcessor, SpeechAdapter, VisionAdapter, TranscriptionResult
from ..memory import MemoryManager, MemoryType


PERCEPTION_PROMPT = """You are the Perception Agent for Sahayak.

Your job is to process and understand multimodal inputs:
- Audio: Transcribe and analyze voice input, detect tone and urgency
- Images: Extract text via OCR, identify document types (khata, ID, receipt)

When processing:
1. Extract key information from the input
2. Identify the language (Hindi, English, mixed)
3. Detect any urgency or stress indicators in voice
4. Summarize the content for other agents

Output structured information that can be used by Fraud and Retrieval agents."""


class PerceptionAgent(BaseAgent):
    """
    Multimodal processing agent
    
    Handles:
    - Audio → Text transcription + acoustic analysis
    - Image → OCR text extraction
    - Language detection
    - Urgency/stress detection
    """
    
    def __init__(
        self,
        llm: LLMAdapter,
        memory: MemoryManager,
        speech_adapter: Optional[SpeechAdapter] = None,
        vision_adapter: Optional[VisionAdapter] = None
    ):
        super().__init__(
            agent_id="perception",
            role="Perception Agent",
            llm=llm,
            memory=memory,
            system_prompt=PERCEPTION_PROMPT
        )
        self.audio_processor = AudioProcessor() # Initialize AudioProcessor
        
        # Lazy initialization of adapters
        self._speech = speech_adapter
        self._vision = vision_adapter
    
    @property
    def speech(self) -> SpeechAdapter:
        """Lazy load speech adapter"""
        if self._speech is None:
            self._speech = SpeechAdapter()
        return self._speech
    
    @property
    def vision(self) -> VisionAdapter:
        """Lazy load vision adapter"""
        if self._vision is None:
            self._vision = VisionAdapter()
        return self._vision
    
    async def process(self, context: AgentContext) -> AgentResult:
        """
        Process multimodal input based on modality type
        """
        self.set_state(AgentState.PROCESSING)
        
        try:
            if context.modality == "audio":
                result = await self._process_audio(context)
            elif context.modality == "image":
                result = await self._process_image(context)
            else:
                # Text input - just pass through with analysis
                result = await self._analyze_text(context)
            
            self.set_state(AgentState.COMPLETED)
            return result
            
        except Exception as e:
            logger.error(f"Perception error: {e}")
            self.set_state(AgentState.ERROR)
            
            return AgentResult(
                success=False,
                content=f"Could not process {context.modality} input.",
                agent_id=self.agent_id,
                metadata={"error": str(e)}
            )
    
    async def _process_audio(self, context: AgentContext) -> AgentResult:
        """Process audio input using AudioProcessor"""
        
        # In a real scenario, context.user_input would be the audio bytes/file
        # or a path to the audio file.
        
        # We assume context.user_input contains the path or bytes
        audio_data = context.user_input
        
        # Process audio
        try:
            result = await self.audio_processor.process(audio_data)
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return AgentResult(
                success=False,
                content=f"Audio processing failed: {str(e)}",
                agent_id=self.agent_id
            )
        
        # Log to memory
        await self.log_to_memory(
            content=f"Audio transcription: {result.text[:200]}...",
            context=context,
            memory_type=MemoryType.USER_INPUT,
            metadata={
                "transcription_full": result.text,
                "detected_language": result.language,
                "confidence": result.transcription_confidence,
                "emotion": result.emotion,
                "emotion_confidence": result.emotion_confidence,
                "source_modality": "audio"
            }
        )
        
        return AgentResult(
            success=True,
            content=result.text,
            agent_id=self.agent_id,
            confidence=result.transcription_confidence or 0.8,
            metadata={
                "transcription": result.text,
                "language": result.language,
                "emotion": result.emotion,
                "emotion_confidence": result.emotion_confidence,
                "type": "audio_transcription"
            }
        )
    
    async def _analyze_voice_content(
        self,
        transcription: TranscriptionResult,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Analyze voice content for urgency, stress, and intent"""
        
        prompt = f"""Analyze this transcribed voice input:

Text: {transcription.text}
Detected Language: {transcription.language}

Provide JSON analysis:
{{
    "urgency": "low/medium/high",
    "sentiment": "positive/neutral/negative/fearful",
    "intent": "brief description of what user wants",
    "key_entities": ["list", "of", "important", "terms"],
    "potential_fraud_indicators": true/false
}}

Consider: Is the user stressed? Are they describing a scam call? Do they mention OTP, money, urgent action?"""

        response = await self.think(prompt, context, temperature=0.2)
        
        try:
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        
        return {"urgency": "medium", "sentiment": "neutral", "intent": "unknown"}
    
    async def _process_image(self, context: AgentContext) -> AgentResult:
        """Process image input - OCR + document analysis"""
        
        image_path = context.metadata.get("image_path")
        image_bytes = context.metadata.get("image_bytes")
        
        if not image_path and not image_bytes:
            return AgentResult(
                success=False,
                content="No image data provided",
                agent_id=self.agent_id
            )
        
        # Extract text via OCR
        ocr_result = await self.vision.extract_text(image_path or image_bytes)
        
        # Analyze document type
        doc_analysis = await self._analyze_document(ocr_result.text, context)
        
        # Log to memory
        await self.log_to_memory(
            content=f"Image OCR: {ocr_result.text[:200]}...",
            context=context,
            memory_type=MemoryType.USER_INPUT,
            metadata={
                "ocr_text": ocr_result.text,
                "ocr_confidence": ocr_result.confidence,
                "document_type": doc_analysis.get("document_type"),
                "source_modality": "image"
            }
        )
        
        return AgentResult(
            success=True,
            content=ocr_result.text,
            agent_id=self.agent_id,
            confidence=ocr_result.confidence,
            metadata={
                "ocr_text": ocr_result.text,
                "document_analysis": doc_analysis,
                "languages": ocr_result.languages
            }
        )
    
    async def _analyze_document(
        self,
        ocr_text: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Analyze document type and extract key information"""
        
        prompt = f"""Analyze this OCR-extracted text from a document:

Text: {ocr_text[:1000]}

Identify:
{{
    "document_type": "khata/ledger/receipt/id_card/bank_statement/unknown",
    "language": "hi/en/mixed",
    "key_fields": {{"field_name": "value"}},
    "financial_indicators": ["any amounts, dates, transactions"]
}}

This is for alternative credit scoring - look for transaction patterns."""

        response = await self.think(prompt, context, temperature=0.2)
        
        try:
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        
        return {"document_type": "unknown", "language": "unknown"}
    
    async def _analyze_text(self, context: AgentContext) -> AgentResult:
        """Analyze text input for language and intent"""
        
        # Detect language and intent from text
        prompt = f"""Analyze this text input:

Text: {context.user_input}

Provide:
{{
    "language": "hi/en/mixed",
    "intent": "brief description",
    "urgency": "low/medium/high"
}}"""

        response = await self.think(prompt, context, temperature=0.2)
        
        analysis = {}
        try:
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                analysis = json.loads(response[start:end])
        except:
            analysis = {"language": "en", "intent": "general query", "urgency": "low"}
        
        return AgentResult(
            success=True,
            content=context.user_input,
            agent_id=self.agent_id,
            confidence=0.9,
            metadata={"analysis": analysis}
        )
