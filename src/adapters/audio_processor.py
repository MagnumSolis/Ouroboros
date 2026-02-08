"""
Audio Processor - Unified processing for speech transcription and emotion detection
"""

import io
from typing import Union
from dataclasses import dataclass
from loguru import logger

from .speech import SpeechAdapter, TranscriptionResult
from .emotion import EmotionAdapter


@dataclass
class AudioResult:
    """Result from audio processing"""
    text: str
    language: str
    emotion: str
    emotion_confidence: float
    transcription_confidence: float


class AudioProcessor:
    """
    Unified Audio Processor
    Orchestrates Speech-to-Text and Emotion Recognition
    """
    
    def __init__(self, speech_adapter: SpeechAdapter = None, emotion_adapter: EmotionAdapter = None):
        self.speech = speech_adapter or SpeechAdapter()
        self.emotion = emotion_adapter or EmotionAdapter()
        logger.info("✅ Audio Processor initialized")

    async def process(self, audio: Union[bytes, io.BytesIO], filename: str = "audio.wav") -> AudioResult:
        """
        Process audio buffer to get text and emotion
        
        Args:
            audio: Raw audio bytes or BytesIO buffer
            filename: Optional filename hint for format detection
            
        Returns:
            AudioResult with transcription and emotion analysis
        """
        try:
            # Normalize input to bytes
            if isinstance(audio, io.BytesIO):
                audio.seek(0)
                audio_bytes = audio.read()
            elif isinstance(audio, bytes):
                audio_bytes = audio
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")
            
            if len(audio_bytes) == 0:
                raise ValueError("Empty audio data received")
            
            logger.debug(f"Processing audio: {len(audio_bytes)} bytes")
            
            # 1. Transcribe using SpeechAdapter (async-safe)
            transcription: TranscriptionResult = await self.speech.transcribe(
                audio_bytes,
                language=None  # Auto-detect
            )
            
            logger.debug(f"Transcription: {transcription.text[:100]}...")
            
            # 2. Emotion Recognition (sync, but we'll run it directly)
            # Avoid asyncio.to_thread() which causes event loop issues in Streamlit
            try:
                audio_buffer = io.BytesIO(audio_bytes)
                emotion_result = self.emotion.predict_from_buffer(audio_buffer)
            except Exception as e:
                logger.warning(f"Emotion detection failed: {e}, using neutral")
                emotion_result = {"emotion": "neutral", "confidence": 0.0}
            
            return AudioResult(
                text=transcription.text,
                language=transcription.language or "unknown",
                emotion=emotion_result.get("emotion", "neutral"),
                emotion_confidence=emotion_result.get("confidence", 0.0),
                transcription_confidence=transcription.confidence or 0.8
            )

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise
