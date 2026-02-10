
"""
Reset script for demo preparation
Clears uploaded files AND their vectors from Qdrant
Run: python scripts/reset_for_demo.py [--force]
"""
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import EmbeddingAdapter
from src.memory import MemoryManager

def main():
    parser = argparse.ArgumentParser(description='Reset Sahayak for Demo')
    parser.add_argument('--force', action='store_true', help='Force reset without confirmation prompts')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧹 SAHAYAK DEMO RESET TOOL")
    print("=" * 60)
    
    # Initialize Memory Manager
    try:
        embeddings = EmbeddingAdapter()
        memory = MemoryManager(embedding_adapter=embeddings)
    except Exception as e:
        print(f"❌ Error initializing memory manager: {e}")
        return

    # Collections to clear
    collections_to_clear = [
        "knowledge_base", 
        "episodic_memory", 
        "working_memory", 
        "semantic_cache"
    ]

    # File directories to clear
    uploads_dir = Path(__file__).parent.parent / "data" / "uploads"
    
    # Check what needs to be done
    files_exist = uploads_dir.exists() and any(uploads_dir.iterdir())
    
    stats = memory.get_collection_stats()
    entries_exist = any(stats.get(c, {}).get("points_count", 0) > 0 for c in collections_to_clear)

    if not args.force:
        print(f"\n⚠️  WARNING: This will delete:")
        print(f"   - All files in {uploads_dir}")
        print(f"   - All entries in Qdrant collections: {', '.join(collections_to_clear)}")
        
        confirm = input("\nType 'RESET' to confirm full wipe: ").strip()
        if confirm != "RESET":
            print("❌ Reset cancelled.")
            return

    print("\n🚀 Starting Reset...")

    # 1. Clear Files
    if uploads_dir.exists():
        for f in uploads_dir.glob("*"):
            if f.is_file() and f.name != ".gitkeep":
                try:
                    f.unlink()
                    print(f"   🗑️  Deleted file: {f.name}")
                except Exception as e:
                    print(f"   ⚠️  Failed to delete {f.name}: {e}")
    print("   ✅ Local files cleared.")

    # 2. Clear Collections
    for col in collections_to_clear:
        count = stats.get(col, {}).get("points_count", 0)
        if count > 0 or args.force:
            try:
                memory.client.delete(collection_name=col, points_selector=models.Filter())
                # Alternatively preserve index but clear points:
                # memory.clear_collection(col) 
                # (But clear_collection implementation might vary, let's stick to manager logic if available)
                # The manager has a clear_collection method, let's use it properly
                memory.clear_collection(col)
                print(f"   ✨ Cleared {col} ({count} entries)")
            except Exception as e:
                print(f"   ⚠️  Error clearing {col}: {e}")
        else:
            print(f"   Example: {col} is already empty.")

    print("\n" + "=" * 60)
    print("✅  SYSTEM READY FOR DEMO")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    from qdrant_client.http import models # Import needed for filter
    main()
