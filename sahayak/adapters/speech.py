"""
Speech Adapter - Unified interface for speech-to-text
Supports: Deepgram API and local OpenAI Whisper
"""

from typing import Optional, Union, BinaryIO
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
import asyncio

from loguru import logger

from ..config import settings


class SpeechProvider(str, Enum):
    """Available speech-to-text providers"""
    DEEPGRAM = "deepgram"
    WHISPER = "whisper"  # Local OpenAI Whisper


@dataclass
class TranscriptionResult:
    """Standard transcription result"""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    provider: Optional[SpeechProvider] = None
    words: Optional[list] = None  # Word-level timestamps if available


class BaseSpeechProvider(ABC):
    """Base class for speech-to-text providers"""
    
    @abstractmethod
    async def transcribe(
        self,
        audio: Union[str, Path, bytes, BinaryIO],
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe audio to text"""
        pass


class DeepgramProvider(BaseSpeechProvider):
    """Deepgram API provider - Real-time, accurate STT"""
    
    def __init__(self, api_key: str):
        from deepgram import DeepgramClient
        self.client = DeepgramClient(api_key)
    
    async def transcribe(
        self,
        audio: Union[str, Path, bytes, BinaryIO],
        language: Optional[str] = None
    ) -> TranscriptionResult:
        from deepgram import PrerecordedOptions, FileSource
        
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language=language or "hi",  # Default Hindi
            detect_language=language is None,
            punctuate=True,
        )
        
        # Handle different audio input types
        if isinstance(audio, (str, Path)):
            with open(audio, "rb") as f:
                audio_data = f.read()
        elif isinstance(audio, bytes):
            audio_data = audio
        else:
            audio_data = audio.read()
        
        payload: FileSource = {"buffer": audio_data}
        
        response = await asyncio.to_thread(
            self.client.listen.prerecorded.v("1").transcribe_file,
            payload,
            options
        )
        
        result = response.results.channels[0].alternatives[0]
        
        return TranscriptionResult(
            text=result.transcript,
            language=response.results.channels[0].detected_language,
            confidence=result.confidence,
            duration=response.metadata.duration,
            provider=SpeechProvider.DEEPGRAM,
            words=[
                {"word": w.word, "start": w.start, "end": w.end}
                for w in result.words
            ] if hasattr(result, 'words') else None
        )


class WhisperProvider(BaseSpeechProvider):
    """Local OpenAI Whisper provider - Offline, 100+ languages"""
    
    def __init__(self, model_size: str = "base"):
        import whisper
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        self.model_size = model_size
    
    async def transcribe(
        self,
        audio: Union[str, Path, bytes, BinaryIO],
        language: Optional[str] = None
    ) -> TranscriptionResult:
        import tempfile
        import os
        
        # Whisper needs a file path
        temp_path = None
        
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                temp_path = f.name
                audio_path = temp_path
        elif hasattr(audio, 'read'):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio.read())
                temp_path = f.name
                audio_path = temp_path
        else:
            audio_path = str(audio)
        
        try:
            # Run transcription in thread pool (Whisper is CPU-intensive)
            result = await asyncio.to_thread(
                self.model.transcribe,
                audio_path,
                language=language,
                verbose=False
            )
            
            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language"),
                provider=SpeechProvider.WHISPER,
                words=[
                    {"word": seg["text"], "start": seg["start"], "end": seg["end"]}
                    for seg in result.get("segments", [])
                ]
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


class SpeechAdapter:
    """
    Unified Speech-to-Text Adapter with fallback
    
    Priority:
    1. Deepgram (if configured) - Fast, accurate, real-time capable
    2. Local Whisper - Always available, works offline
    """
    
    def __init__(
        self,
        preferred_provider: Optional[SpeechProvider] = None,
        whisper_model: str = "base"
    ):
        self.provider: BaseSpeechProvider = None
        self.provider_type: SpeechProvider = None
        self._initialize_provider(preferred_provider, whisper_model)
    
    def _initialize_provider(
        self,
        preferred: Optional[SpeechProvider],
        whisper_model: str
    ):
        """Initialize the best available provider"""
        
        # Try Deepgram first if configured
        if settings.has_deepgram and preferred != SpeechProvider.WHISPER:
            try:
                self.provider = DeepgramProvider(api_key=settings.deepgram_api_key)
                self.provider_type = SpeechProvider.DEEPGRAM
                logger.info("Using Deepgram for speech-to-text")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Deepgram: {e}")
        
        # Fall back to Whisper
        try:
            self.provider = WhisperProvider(model_size=whisper_model)
            self.provider_type = SpeechProvider.WHISPER
            logger.info(f"Using local Whisper ({whisper_model}) for speech-to-text")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            raise RuntimeError("No speech-to-text provider available")
    
    @property
    def is_local(self) -> bool:
        """Check if using local Whisper"""
        return self.provider_type == SpeechProvider.WHISPER
    
    async def transcribe(
        self,
        audio: Union[str, Path, bytes, BinaryIO],
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio file path, bytes, or file-like object
            language: Optional language code (e.g., 'hi' for Hindi, 'en' for English)
                     If None, auto-detection is attempted
                     
        Returns:
            TranscriptionResult with text and metadata
        """
        return await self.provider.transcribe(audio, language)
    
    async def transcribe_to_text(
        self,
        audio: Union[str, Path, bytes, BinaryIO],
        language: Optional[str] = None
    ) -> str:
        """Simple transcription returning just the text"""
        result = await self.transcribe(audio, language)
        return result.text
