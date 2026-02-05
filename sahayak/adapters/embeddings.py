"""
Embedding Adapter - Unified interface for text embeddings
Supports: Cohere API and local Sentence Transformers
"""

from typing import List, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from loguru import logger

from ..config import settings


class EmbeddingProvider(str, Enum):
    """Available embedding providers"""
    COHERE = "cohere"
    LOCAL = "local"  # Sentence Transformers


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers"""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension"""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.embed([text])
        return embeddings[0]


class CohereProvider(BaseEmbeddingProvider):
    """Cohere API provider for embeddings"""
    
    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0"):
        import cohere
        self.client = cohere.AsyncClient(api_key=api_key)
        self.model = model
        self._dimension = 1024  # Cohere v3 models use 1024
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"
        )
        return [list(emb) for emb in response.embeddings]
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a query (different input_type for retrieval)"""
        response = await self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query"
        )
        return list(response.embeddings[0])


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local Sentence Transformers provider"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded local embedding model: {model_name} (dim={self._dimension})")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        # SentenceTransformers is synchronous, but we wrap for consistency
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    async def embed_query(self, text: str) -> List[float]:
        """For local models, query embedding is same as document"""
        return await self.embed_single(text)


class EmbeddingAdapter:
    """
    Unified Embedding Adapter with automatic fallback
    
    Priority:
    1. Cohere (if configured) - Production-quality, multilingual
    2. Local Sentence Transformers - Always available, no API needed
    """
    
    def __init__(self, preferred_provider: Optional[EmbeddingProvider] = None):
        self.provider: BaseEmbeddingProvider = None
        self.provider_type: EmbeddingProvider = None
        self._initialize_provider(preferred_provider)
    
    def _initialize_provider(self, preferred: Optional[EmbeddingProvider] = None):
        """Initialize the best available provider"""
        
        # Try Cohere first if configured and preferred
        if settings.has_cohere and preferred != EmbeddingProvider.LOCAL:
            try:
                self.provider = CohereProvider(
                    api_key=settings.cohere_api_key,
                    model=settings.cohere_embed_model
                )
                self.provider_type = EmbeddingProvider.COHERE
                logger.info(f"Using Cohere embeddings (dim={self.provider.dimension})")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere: {e}")
        
        # Fall back to local
        try:
            self.provider = LocalEmbeddingProvider(
                model_name=settings.local_embed_model
            )
            self.provider_type = EmbeddingProvider.LOCAL
            logger.info(f"Using local embeddings (dim={self.provider.dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize local embeddings: {e}")
            raise RuntimeError("No embedding provider available")
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension"""
        return self.provider.dimension
    
    @property
    def is_local(self) -> bool:
        """Check if using local provider"""
        return self.provider_type == EmbeddingProvider.LOCAL
    
    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
        return await self.provider.embed(texts)
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return await self.provider.embed_single(text)
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query (optimized for retrieval)
        Uses different input_type for Cohere
        """
        if hasattr(self.provider, 'embed_query'):
            return await self.provider.embed_query(text)
        return await self.embed_single(text)
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
