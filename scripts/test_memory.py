
import asyncio
import sys
import os
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.memory import MemoryManager
from src.config import settings

async def test_memory_manager():
    print("\n" + "=" * 50)
    print("🧠 Testing Memory Manager (Qdrant)")
    print("=" * 50)
    
    try:
        # Initialize
        print(f"Connecting to Qdrant at {settings.qdrant_url or 'local'}...")
        memory = MemoryManager()
        print("✅ MemoryManager initialized")
        
        # Test 1: Store
        print("\nTesting Store...")
        collection = "episodic_memory"
        test_content = f"Test memory entry {datetime.now().isoformat()}"
        doc_id = await memory.store(
            collection=collection,
            content=test_content,
            payload={"type": "test", "timestamp": datetime.now().isoformat()}
        )
        print(f"✅ Stored document with ID: {doc_id}")
        
        # Test 2: Search (The critical part that was failing)
        print("\nTesting Search (query_points)...")
        # Allow some time for indexing if needed, though usually instant for single point
        await asyncio.sleep(1)
        
        results = await memory.search(
            collection=collection,
            query="Test memory",
            limit=5
        )
        
        print(f"✅ Search returned {len(results)} results")
        if len(results) > 0:
            print(f"   Top result: {results[0].get('content')} (score: {results[0].get('score'):.4f})")
            
            # Verify it's our doc
            found_id = results[0].get('id')
            if found_id == doc_id:
                print("   ✅ Validated retrieved ID matches stored ID")
            else:
                print(f"   ⚠️ Retrieved ID {found_id} != Stored ID {doc_id}")
        else:
            print("   ❌ No results found!")
            return False

        # Test 3: Delete (cleanup)
        print("\nTesting Delete...")
        success = await memory.delete(collection, doc_id)
        if success:
            print(f"✅ Deleted document {doc_id}")
        else:
            print(f"❌ Failed to delete document {doc_id}")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_memory_manager())
    sys.exit(0 if success else 1)
