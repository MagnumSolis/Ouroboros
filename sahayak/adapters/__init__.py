"""Adapters module - External service integrations"""

from .llm import LLMAdapter, ChatMessage, LLMResponse, LLMProvider
from .embeddings import EmbeddingAdapter, EmbeddingProvider
from .speech import SpeechAdapter, SpeechProvider, TranscriptionResult
from .vision import VisionAdapter, OCRResult

__all__ = [
    "LLMAdapter", "ChatMessage", "LLMResponse", "LLMProvider",
    "EmbeddingAdapter", "EmbeddingProvider",
    "SpeechAdapter", "SpeechProvider", "TranscriptionResult",
    "VisionAdapter", "OCRResult"
]
