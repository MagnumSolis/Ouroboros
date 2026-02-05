"""
Memory Manager - Qdrant Blackboard for agent communication
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    UpdateStatus,
)
from loguru import logger

from ..config import settings
from ..adapters.embeddings import EmbeddingAdapter
from .schemas import MemoryEntry, MemoryType, FraudPattern, KnowledgeDocument, WorkingMemoryEntry
from .collections import COLLECTIONS, get_vector_params, CollectionConfig


class MemoryManager:
    """
    Qdrant-based Blackboard for multi-agent communication
    
    All agents read/write through this manager, enabling:
    - Asynchronous agent communication
    - Persistent memory across sessions
    - Semantic search over all entries
    """
    
    def __init__(
        self,
        embedding_adapter: Optional[EmbeddingAdapter] = None,
        auto_create_collections: bool = True
    ):
        """
        Initialize Memory Manager
        
        Args:
            embedding_adapter: Adapter for generating embeddings
            auto_create_collections: Create collections if they don't exist
        """
        # Initialize Qdrant Client
        if settings.qdrant_url and "cloud.qdrant.io" in settings.qdrant_url:
            logger.info(f"Connecting to Qdrant Cloud (port 443): {settings.qdrant_url}")
            self.client = QdrantClient(
                url=settings.qdrant_url,
                port=None,  # Force default port (443) based on scheme
                api_key=settings.qdrant_api_key,
            )
        elif settings.qdrant_url:
             logger.info(f"Connecting to Qdrant URL: {settings.qdrant_url}")
             self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            logger.info(f"Connecting to Qdrant Local: {settings.qdrant_host}:{settings.qdrant_port}")
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
            )
            
        # Verify connection immediately (Fail Fast)
        try:
            self.client.get_collections()
            logger.info("✅ Connected to Qdrant successfully")
        except Exception as e:
            logger.critical(f"❌ Failed to connect to Qdrant: {e}")
            raise RuntimeError(f"Could not connect to Qdrant Vector DB. Check credentials. Error: {e}")
        
        self.embedder = embedding_adapter or EmbeddingAdapter()
        
        # Update collection configs with actual embedding dimension
        self._update_vector_sizes()
        
        if auto_create_collections:
            self._ensure_collections()
        
        logger.info(f"Memory Manager initialized (embedding dim: {self.embedder.dimension})")
    
    def _update_vector_sizes(self):
        """Update collection vector sizes based on embedding model"""
        for config in COLLECTIONS.values():
            config.vector_size = self.embedder.dimension
    
    def _ensure_collections(self):
        """Create collections if they don't exist"""
        existing = {c.name for c in self.client.get_collections().collections}
        
        for name, config in COLLECTIONS.items():
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=get_vector_params(config),
                )
                logger.info(f"Created collection: {name}")
            else:
                logger.debug(f"Collection exists: {name}")
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    async def store(
        self,
        collection: str,
        content: str,
        payload: Dict[str, Any],
        id: Optional[str] = None
    ) -> str:
        """
        Store a document with its embedding
        
        Args:
            collection: Target collection name
            content: Text content to embed
            payload: Additional metadata
            id: Optional ID (generated if not provided)
            
        Returns:
            ID of stored document
        """
        doc_id = id or str(uuid.uuid4())
        
        # Generate embedding
        vector = await self.embedder.embed_single(content)
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload={**payload, "content": content}
                )
            ]
        )
        
        return doc_id
    
    async def search(
        self,
        collection: str,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Semantic search in a collection
        
        Args:
            collection: Collection to search
            query: Search query text
            limit: Max results to return
            filters: Optional field filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching documents with scores
        """
        # Embed query
        query_vector = await self.embedder.embed_query(query)
        
        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                **hit.payload
            }
            for hit in results
        ]
    
    async def get_by_id(
        self,
        collection: str,
        doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        results = self.client.retrieve(
            collection_name=collection,
            ids=[doc_id],
            with_payload=True,
        )
        
        if results:
            return {"id": results[0].id, **results[0].payload}
        return None
    
    async def delete(self, collection: str, doc_id: str) -> bool:
        """Delete a document by ID"""
        result = self.client.delete(
            collection_name=collection,
            points_selector=[doc_id],
        )
        return result.status == UpdateStatus.COMPLETED
    
    # =========================================================================
    # Specialized Operations
    # =========================================================================
    
    async def store_memory(self, entry: MemoryEntry) -> str:
        """Store a memory entry to episodic_memory"""
        return await self.store(
            collection="episodic_memory",
            content=entry.content,
            payload=entry.model_dump(),
            id=entry.id
        )
    
    async def store_fraud_pattern(self, pattern: FraudPattern) -> str:
        """Store a fraud pattern"""
        content = f"{pattern.description} {' '.join(pattern.indicators)}"
        return await self.store(
            collection="fraud_patterns",
            content=content,
            payload=pattern.model_dump(),
            id=pattern.id
        )
    
    async def store_knowledge(self, doc: KnowledgeDocument) -> str:
        """Store a knowledge base document"""
        return await self.store(
            collection="knowledge_base",
            content=doc.content,
            payload=doc.model_dump(),
            id=doc.id
        )
    
    async def store_task(self, task: WorkingMemoryEntry) -> str:
        """Store a working memory task"""
        content = f"{task.task_type}: {task.input_data}"
        return await self.store(
            collection="working_memory",
            content=content,
            payload=task.model_dump(),
            id=task.id
        )
    
    async def check_fraud_similarity(
        self,
        text: str,
        threshold: float = 0.75,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Check text against known fraud patterns
        
        Args:
            text: Text to check (transcribed call, message, etc.)
            threshold: Minimum similarity for alarm
            limit: Max patterns to return
            
        Returns:
            List of matching fraud patterns with scores
        """
        return await self.search(
            collection="fraud_patterns",
            query=text,
            limit=limit,
            score_threshold=threshold
        )
    
    async def retrieve_knowledge(
        self,
        query: str,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge documents
        
        Args:
            query: Search query
            doc_type: Optional filter by document type
            language: Optional filter by language
            limit: Max results
        """
        filters = {}
        if doc_type:
            filters["document_type"] = doc_type
        if language:
            filters["language"] = language
        
        return await self.search(
            collection="knowledge_base",
            query=query,
            limit=limit,
            filters=filters if filters else None
        )
    
    async def get_pending_tasks(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending tasks from working memory"""
        filters = {"status": "pending"}
        if agent_id:
            filters["assigned_agent"] = agent_id
        
        # For pending tasks, we search with a generic query
        # In practice, agents poll for their assigned tasks
        return await self.search(
            collection="working_memory",
            query="pending task",
            limit=10,
            filters=filters
        )
    
    async def get_interaction_history(
        self,
        interaction_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all entries for an interaction"""
        return await self.search(
            collection="episodic_memory",
            query="",  # We're filtering by ID, not searching
            limit=limit,
            filters={"interaction_id": interaction_id}
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections"""
        stats = {}
        for name in COLLECTIONS:
            info = self.client.get_collection(name)
            stats[name] = {
                "points_count": info.points_count,
                # "vectors_count": info.vectors_count, # Deprecated in newer clients
                "status": info.status,
            }
        return stats
    
    def clear_collection(self, collection: str) -> bool:
        """Clear all documents from a collection (use with caution!)"""
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection}")
        
        self.client.delete_collection(collection)
        config = COLLECTIONS[collection]
        self.client.create_collection(
            collection_name=collection,
            vectors_config=get_vector_params(config),
        )
        logger.warning(f"Cleared collection: {collection}")
        return True
