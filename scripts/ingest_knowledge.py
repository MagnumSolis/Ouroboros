#!/usr/bin/env python3
"""
Ingest Knowledge Base Files into Qdrant

Reads text files from data/knowledge_base and stores them in the knowledge_base collection.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.adapters import EmbeddingAdapter
from src.memory import MemoryManager
from loguru import logger


async def ingest_knowledge():
    """Ingest knowledge base files into Qdrant"""
    print("\n" + "=" * 50)
    print("📚 Ingesting Knowledge Base")
    print("=" * 50)
    
    # Initialize
    print("\n1. Initializing adapters...")
    embeddings = EmbeddingAdapter()
    memory = MemoryManager(embedding_adapter=embeddings)
    print(f"✅ Connected to Qdrant (embedding dim: {embeddings.dimension})")
    
    # Find knowledge files
    knowledge_dir = Path("data/knowledge_base")
    if not knowledge_dir.exists():
        print(f"❌ Knowledge directory not found: {knowledge_dir}")
        return False
    
    files = list(knowledge_dir.glob("*.txt"))
    if not files:
        print(f"⚠️  No .txt files found in {knowledge_dir}")
        return False
    
    print(f"\n2. Found {len(files)} knowledge files:")
    for f in files:
        print(f"   - {f.name}")
    
    # Ingest each file
    print("\n3. Ingesting into Qdrant...")
    success_count = 0
    
    for filepath in files:
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"⚠️  Skipping empty file: {filepath.name}")
                continue
            
            # Store in knowledge_base collection
            doc_id = await memory.store(
                collection="knowledge_base",
                content=content,
                payload={
                    "source": filepath.name,
                    "type": "scheme_info",
                    "language": "mixed"  # Hindi + English
                }
            )
            
            print(f"   ✅ Ingested: {filepath.name} (ID: {doc_id[:8]}...)")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to ingest {filepath.name}: {e}")
    
    print(f"\n4. Ingestion complete: {success_count}/{len(files)} files")
    
    # Verify
    print("\n5. Verifying knowledge_base collection...")
    try:
        stats = memory.get_collection_stats()
        kb_count = stats.get("knowledge_base", {}).get("points_count", 0)
        print(f"   ✅ knowledge_base collection has {kb_count} documents")
    except Exception as e:
        print(f"   ⚠️  Could not verify: {e}")
    
    return success_count > 0


if __name__ == "__main__":
    success = asyncio.run(ingest_knowledge())
    sys.exit(0 if success else 1)
