#!/usr/bin/env python3
"""
Setup Qdrant collections for Sahayak
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from src.config import settings
from src.memory.collections import COLLECTIONS, get_vector_params


def setup_collections():
    """Create all required Qdrant collections"""
    
    print("\n🛡️ SAHAYAK - Qdrant Setup")
    print("=" * 50)
    print(f"Connecting to: {settings.qdrant_url}")
    
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        
        # Check connection
        collections = client.get_collections()
        existing = {c.name for c in collections.collections}
        print(f"✅ Connected! Existing collections: {existing or 'none'}")
        
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("\n💡 Make sure Qdrant is running:")
        print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return False
    
    print("\n📦 Creating collections...")
    
    for name, config in COLLECTIONS.items():
        if name in existing:
            print(f"  ⏭️  {name} already exists")
        else:
            try:
                client.create_collection(
                    collection_name=name,
                    vectors_config=get_vector_params(config),
                )
                print(f"  ✅ Created: {name} (dim={config.vector_size})")
            except Exception as e:
                print(f"  ❌ Failed to create {name}: {e}")
                return False
    
    print("\n✅ All collections ready!")
    
    # Show collection info
    print("\n📊 Collection Info:")
    for name in COLLECTIONS:
        info = client.get_collection(name)
        print(f"  {name}: {info.points_count} points, status={info.status}")
    
    return True


if __name__ == "__main__":
    success = setup_collections()
    exit(0 if success else 1)
