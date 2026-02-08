
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
from loguru import logger
from .manager import MemoryManager

class SemanticCache:
    """
    Semantic Cache for LLM responses using Qdrant.
    Reduces latency and cost for repeated similar queries.
    """
    
    def __init__(self, memory_manager: MemoryManager, collection_name: str = "semantic_cache"):
        self.memory = memory_manager
        self.collection = collection_name
        self.similarity_threshold = 0.95 # High threshold for exact/near-exact matches
        
    async def get(self, query: str) -> Optional[str]:
        """
        Retrieve cached response if a similar query exists.
        """
        try:
            results = await self.memory.search(
                collection=self.collection,
                query=query,
                limit=1,
                score_threshold=self.similarity_threshold
            )
            
            if results:
                hit = results[0]
                payload = hit
                
                # Check TTL
                expires_at = payload.get("expires_at")
                if expires_at:
                    if datetime.fromisoformat(expires_at) < datetime.now():
                        logger.debug(f"Cache miss (expired): {query[:50]}...")
                        # Async delete expired entry? For now just ignore
                        return None
                
                logger.info(f"✨ Semantic Cache HIT (score: {hit['score']:.4f})")
                return payload.get("response")
                
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            
        return None

    async def set(self, query: str, response: str, ttl_hours: int = 24):
        """
        Cache a response.
        """
        try:
            expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
            
            # Create a deterministic ID based on query hash to avoid duplicates
            # dict_id = hashlib.md5(query.encode()).hexdigest()
            # Actually Qdrant handles vectors, so we can just let it generate UUID or use hash
            
            await self.memory.store(
                collection=self.collection,
                content=query, # We embed the query
                payload={
                    "response": response,
                    "original_query": query,
                    "created_at": datetime.now().isoformat(),
                    "expires_at": expires_at
                }
            )
            logger.debug(f"Cached response for: {query[:50]}...")
            
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
