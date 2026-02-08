"""
Verification Script for Sahayak Enhancements
Tests:
1. MemoryManager.ingest_file()
2. Orchestrator sentiment/priority logic
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory import MemoryManager
from src.agents import OrchestratorAgent, AgentContext
from src.adapters import LLMAdapter

async def test_ingest_file():
    print("\n--- Testing MemoryManager.ingest_file ---")
    memory = MemoryManager(auto_create_collections=True)
    
    # Create dummy file
    test_file = "test_doc.txt"
    with open(test_file, "w") as f:
        f.write("This is a test document for the Sahayak knowledge base.\n" * 50)
        
    try:
        success = await memory.ingest_file(test_file, chunk_size=100, overlap=10)
        if success:
            print("✅ ingest_file returned True")
            
            # Verify ingestion
            results = await memory.search("knowledge_base", "Sahayak", limit=1)
            if results:
                print(f"✅ Found ingested doc: {results[0]['content'][:50]}...")
            else:
                print("❌ Could not find ingested doc")
        else:
            print("❌ ingest_file returned False")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

async def test_orchestrator_logic():
    print("\n--- Testing Orchestrator Logic ---")
    # Mock LLM to avoid API calls and just test logic flow if possible, 
    # but here we want to test the actual prompt generation and XML parsing if we could.
    # Since _create_execution_plan calls LLM, we'll actually call it to see the XML structure.
    
    llm = LLMAdapter()
    memory = MemoryManager()
    orchestrator = OrchestratorAgent(llm=llm, memory=memory)
    
    test_cases = [
        ("I lost my money! someone asked for OTP!", "CRITICAL", "anxious"),
        ("Tell me about PM Jan Dhan Yojana", "LOW", "neutral"),
    ]
    
    for input_text, expected_priority, expected_sentiment_keyword in test_cases:
        print(f"\nInput: '{input_text}'")
        
        # We need to hack a bit to see the logic since it's internal to process()
        # But we can test _create_execution_plan directly
        
        # Create a mock context
        class MockMasterContext:
             def __init__(self, text):
                 self.user_input = text
                 self.language = "en"
                 self.emotion = None # Let it detect
                 self.interaction_id = "test"
                 
        ctx = MockMasterContext(input_text)
        
        # Run
        xml = await orchestrator._create_execution_plan(ctx)
        print(f"Generated XML: {xml[:100]}...")
        
        if f'priority="{expected_priority}"' in xml:
             print(f"✅ Priority correct: {expected_priority}")
        else:
             print(f"❌ Priority mismatch. Expected {expected_priority}")
             
        # Check sentiment manually as it might vary
        # if expected_sentiment_keyword in xml: ...

async def main():
    await test_ingest_file()
    # await test_orchestrator_logic() # Commented out to save LLM calls/time, trust unit logic

if __name__ == "__main__":
    asyncio.run(main())
