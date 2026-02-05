"""
Sahayak Configuration Settings
Uses Pydantic Settings for environment variable management
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # =========================================================================
    # LLM Providers
    # =========================================================================
    
    # Groq - Primary LLM for agent reasoning
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    
    # Google Gemini - Multimodal backup
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    
    # Perplexity - Primary for Demo (Online)
    perplexity_api_key: Optional[str] = None
    perplexity_model: str = "sonar"
    
    # OpenRouter - Free models (Llama 3.3, Mistral, etc.)
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    
    # =========================================================================
    # Embeddings
    # =========================================================================
    
    # Cohere - Production embeddings (optional, falls back to local)
    cohere_api_key: Optional[str] = None
    cohere_embed_model: str = "embed-multilingual-v3.0"
    
    # Local embedding model (Sentence Transformers)
    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # =========================================================================
    # Speech-to-Text
    # =========================================================================
    
    # Deepgram - Real-time STT (optional, falls back to Whisper)
    deepgram_api_key: Optional[str] = None
    
    # Whisper - Local STT
    whisper_model: str = "base"  # tiny, base, small, medium, large
    
    # =========================================================================
    # Vector Database
    # =========================================================================
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    
    log_level: str = "INFO"
    default_language: str = "hi"  # Hindi as default vernacular
    
    # =========================================================================
    # Helper Properties
    # =========================================================================
    
    @property
    def has_groq(self) -> bool:
        """Check if Groq API is configured"""
        return bool(self.groq_api_key)
    
    @property
    def has_gemini(self) -> bool:
        """Check if Gemini API is configured"""
        return bool(self.gemini_api_key)

    @property
    def has_perplexity(self) -> bool:
        """Check if Perplexity API is configured"""
        return bool(self.perplexity_api_key)
    
    @property
    def has_cohere(self) -> bool:
        """Check if Cohere API is configured"""
        return bool(self.cohere_api_key)
    
    @property
    def has_deepgram(self) -> bool:
        """Check if Deepgram API is configured"""
        return bool(self.deepgram_api_key)
    
    @property
    def has_openrouter(self) -> bool:
        """Check if OpenRouter API is configured"""
        return bool(self.openrouter_api_key)
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


# Singleton instance
settings = Settings()
