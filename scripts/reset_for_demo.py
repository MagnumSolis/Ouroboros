"""
Reset script for demo preparation
Clears uploaded files AND their vectors from Qdrant
Run: python scripts/reset_for_demo.py
"""
import asyncio
import sys
import os
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import EmbeddingAdapter
from src.memory import MemoryManager


def main():
    print("=" * 60)
    print("🧹 DEMO RESET SCRIPT")
    print("=" * 60)
    
    # 1. Clear local upload files
    uploads_dir = Path(__file__).parent.parent / "data" / "uploads"
    
    print(f"\n📁 Upload Directory: {uploads_dir}")
    
    if uploads_dir.exists():
        files = list(uploads_dir.glob("*"))
        print(f"   Found {len(files)} file(s):")
        for f in files:
            print(f"   - {f.name}")
        
        confirm = input("\n⚠️  Delete these files? (y/n): ").strip().lower()
        if confirm == "y":
            for f in files:
                f.unlink()
            print("   ✅ Files deleted!")
        else:
            print("   ⏭️  Skipped file deletion.")
    else:
        print("   📂 Directory is empty or doesn't exist.")
    
    # 2. Clear Qdrant knowledge_base collection
    print("\n" + "-" * 60)
    print("🗄️  Qdrant Knowledge Base:")
    
    embeddings = EmbeddingAdapter()
    memory = MemoryManager(embedding_adapter=embeddings)
    
    stats = memory.get_collection_stats()
    kb_count = stats.get("knowledge_base", {}).get("points_count", 0)
    print(f"   Current entries: {kb_count}")
    
    if kb_count > 0:
        confirm = input(f"\n⚠️  Clear ALL {kb_count} entries from knowledge_base? (y/n): ").strip().lower()
        if confirm == "y":
            memory.clear_collection("knowledge_base")
            print("   ✅ Knowledge base cleared!")
        else:
            print("   ⏭️  Skipped Qdrant reset.")
    else:
        print("   📭 Collection is already empty.")
    
    # 3. Optionally clear episodic memory
    print("\n" + "-" * 60)
    em_count = stats.get("episodic_memory", {}).get("points_count", 0)
    print(f"🧠 Episodic Memory: {em_count} entries")
    
    if em_count > 0:
        confirm = input(f"⚠️  Also clear episodic memory? (y/n): ").strip().lower()
        if confirm == "y":
            memory.clear_collection("episodic_memory")
            print("   ✅ Episodic memory cleared!")
        else:
            print("   ⏭️  Kept episodic memory.")
    
    print("\n" + "=" * 60)
    print("✅ Reset complete! Ready for demo.")
    print("=" * 60)


if __name__ == "__main__":
    main()
