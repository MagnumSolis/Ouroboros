
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory import MemoryManager
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Use a direct client for raw access if needed, or better, use the manager
# We'll use the proper MemoryManager to respect the architecture

async def main():
    print("\n" + "=" * 60)
    print("🧠  QDRANT EPISODIC MEMORY TIMELINE")
    print("=" * 60 + "\n")
    
    try:
        # Initialize
        memory = MemoryManager()
        
        # We want to scroll back in time. Qdrant point scroll is best for this.
        # But MemoryManager abstract this. Let's use the raw client for this specific visualization
        # to guarantee we get the latest points in reverse order.
        client = memory.client
        
        # Scroll points from 'episodic_memory'
        # We'll just fetch the last 10 points. 
        # Since UUIDs aren't sequential, we rely on 'timestamp' in payload if available,
        # or we just search for "*" and sort.
        
        # Search for everything (limit 20)
        results = client.scroll(
            collection_name="episodic_memory",
            limit=10,
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Sort by timestamp (assuming standard payload structure)
        # Sahayak stores 'timestamp' in ISO format or float
        sorted_results = sorted(
            results, 
            key=lambda x: x.payload.get('timestamp', 0) if x.payload else 0,
            reverse=True # Newest first
        )
        
        if not sorted_results:
            print("   (No memories found. Start a conversation in the dashboard!)")
            return

        print(f"   Found {len(sorted_results)} recent interactions:\n")
        
        for point in sorted_results:
            payload = point.payload
            agent = payload.get("agent_id", "unknown").upper()
            content = payload.get("content", "")
            timestamp = payload.get("timestamp", "")
            
            # Format timestamp nicely if possible
            time_str = ""
            try:
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                elif isinstance(timestamp, str):
                    # Try parsing ISO
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = str(timestamp)[:8]

            # Color coding (ANSI)
            if "USER" in agent:
                icon = "👤"
                color = "\033[94m" # Blue
            elif "ASSISTANT" in agent or "ORCHESTRATOR" in agent:
                icon = "🤖"
                color = "\033[92m" # Green
            else:
                icon = "⚙️"
                color = "\033[90m" # Grey
            
            reset = "\033[0m"
            
            # Clean content (truncate if too long for demo)
            clean_content = content.replace("\n", " ")
            if len(clean_content) > 80:
                clean_content = clean_content[:77] + "..."
                
            print(f"   {color}[{time_str}] {icon} {agent:<12} : {clean_content}{reset}")

        print("\n" + "=" * 60)
        print("✅  Context retained. Ready for next turn.")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"Error inspecting memory: {e}")

if __name__ == "__main__":
    asyncio.run(main())
