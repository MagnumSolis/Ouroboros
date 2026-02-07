
import asyncio
import io
import numpy as np
import librosa
from typing import Dict, Any, Optional, Union
from loguru import logger

from .speech import SpeechAdapter, TranscriptionResult
from .emotion import EmotionAdapter

class AudioResult:
    def __init__(
        self,
        text: str,
        language: str,
        emotion: str,
        emotion_confidence: float,
        transcription_confidence: float
    ):
        self.text = text
        self.language = language
        self.emotion = emotion
        self.emotion_confidence = emotion_confidence
        self.transcription_confidence = transcription_confidence

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
        """
        try:
            # 1. Transcribe (Async)
            # Create a bytes buffer if it's raw bytes
            if isinstance(audio, bytes):
                audio_buffer = audio
            else:
                audio_buffer = audio.read()
            
            # Use SpeechAdapter for transcription
            transcription: TranscriptionResult = await self.speech.transcribe(
                audio_buffer,
                language=None # Auto-detect
            )
            
            # 2. Emotion Recognition (Sync - CPU/GPU intensive)
            # We need to load audio with librosa for feature extraction
            # Librosa expects a path or file-like object.
            # Convert bytes to file-like object for librosa
            if isinstance(audio, bytes):
                f = io.BytesIO(audio)
            else:
                f = audio
                f.seek(0)
            
            # Extract features and predict using EmotionAdapter
            # Note: Librosa loading can be slow, might want to run in thread if blocking
            emotion_result = await asyncio.to_thread(
                self.emotion.predict_from_buffer, f
            )
            
            return AudioResult(
                text=transcription.text,
                language=transcription.language,
                emotion=emotion_result["emotion"],
                emotion_confidence=emotion_result["confidence"],
                transcription_confidence=transcription.confidence or 0.0
            )

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise e
