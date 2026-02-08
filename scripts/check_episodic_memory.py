"""
Script to directly query and inspect Episodic Memory in Qdrant
Run from the sahayak directory: python scripts/check_episodic_memory.py
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import EmbeddingAdapter
from src.memory import MemoryManager


async def main():
    print("=" * 60)
    print("🧠 EPISODIC MEMORY INSPECTOR")
    print("=" * 60)
    
    # Initialize
    embeddings = EmbeddingAdapter()
    memory = MemoryManager(embedding_adapter=embeddings)
    
    # 1. Get collection stats
    stats = memory.get_collection_stats()
    episodic_stats = stats.get("episodic_memory", {})
    
    print(f"\n📊 Collection Stats:")
    print(f"   Total Entries: {episodic_stats.get('points_count', 0)}")
    print(f"   Status: {episodic_stats.get('status', 'Unknown')}")
    
    # 2. Search for recent entries (use empty query to get all)
    print("\n🔍 Recent Episodic Memories (last 10):")
    print("-" * 60)
    
    # Use a generic search to get recent entries
    results = await memory.search(
        collection="episodic_memory",
        query="user interaction",  # Generic query to match most entries
        limit=10,
        score_threshold=0.0  # Get everything
    )
    
    if not results:
        print("   No episodic memories found yet.")
        print("   💡 Try chatting with the dashboard first!")
    else:
        for i, entry in enumerate(results, 1):
            print(f"\n   [{i}] ID: {entry.get('id', 'N/A')[:8]}...")
            print(f"       Agent: {entry.get('agent_id', 'N/A')}")
            print(f"       Type: {entry.get('memory_type', 'N/A')}")
            print(f"       Content: {entry.get('content', '')[:100]}...")
            print(f"       Score: {entry.get('score', 0):.3f}")
    
    # 3. Search for specific content (example)
    print("\n" + "=" * 60)
    search_term = input("🔎 Enter a term to search in episodic memory (or press Enter to skip): ").strip()
    
    if search_term:
        print(f"\n   Searching for: '{search_term}'")
        search_results = await memory.search(
            collection="episodic_memory",
            query=search_term,
            limit=5,
            score_threshold=0.3
        )
        
        if search_results:
            print(f"\n   Found {len(search_results)} relevant memories:")
            for r in search_results:
                print(f"\n   • [{r.get('score', 0):.2f}] {r.get('content', '')[:200]}...")
        else:
            print("   No matching memories found.")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
