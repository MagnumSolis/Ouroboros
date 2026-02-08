#!/usr/bin/env python3
"""
Test Orchestrator Agent
Verifies orchestrator works with new Groq/Perplexity adapters
"""

import asyncio
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.config import settings
from src.adapters import LLMAdapter, EmbeddingAdapter
from src.memory import MemoryManager
from src.agents import (
    OrchestratorAgent, RetrievalAgent, FraudAgent, 
    PerceptionAgent, CriticAgent, AgentContext
)


async def test_orchestrator():
    """Test orchestrator functionality end-to-end"""
    print("\n" + "=" * 50)
    print("🧠 Testing Orchestrator Agent")
    print("=" * 50)
    
    # Initialize system
    try:
        print("\n1. Initializing adapters...")
        llm = LLMAdapter()
        embeddings = EmbeddingAdapter()
        memory = MemoryManager(embedding_adapter=embeddings)
        print("✅ Adapters initialized")
        
        print("\n2. Creating agents...")
        orchestrator = OrchestratorAgent(llm=llm, memory=memory)
        retrieval = RetrievalAgent(llm=llm, memory=memory)
        fraud = FraudAgent(llm=llm, memory=memory)
        perception = PerceptionAgent(llm=llm, memory=memory)
        critic = CriticAgent(llm=llm, memory=memory)
        print("✅ Agents created")
        
        print("\n3. Registering agents with orchestrator...")
        orchestrator.register_agents([retrieval, fraud, perception, critic])
        print("✅ Agents registered")
        
        # Test Cases
        test_cases = [
            {
                "name": "Fraud Detection",
                "input": "Someone called me asking for my OTP for KYC update",
                "language": "en",
                "expected_fraud": True
            },
            {
                "name": "Scheme Query",
                "input": "PM Jan Dhan Yojana के बारे में बताइए",
                "language": "hi",
                "expected_fraud": False
            }
        ]
        
        results = []
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'=' * 50}")
            print(f"Test Case {i}: {test['name']}")
            print(f"{'=' * 50}")
            print(f"Query: {test['input']}")
            print(f"Language: {test['language']}")
            
            context = AgentContext(
                interaction_id=f"test_{i}",
                user_input=test["input"],
                language=test["language"],
                modality="text"
            )
            
            try:
                result = await orchestrator.process(context)
                
                print(f"\n✅ Orchestrator Response:")
                print(f"   Success: {result.success}")
                print(f"   Response: {result.content[:200]}...")
                print(f"   Fraud Detected: {result.metadata.get('is_fraud', False)}")
                
                # Verify
                is_fraud = result.metadata.get('is_fraud', False)
                if is_fraud == test["expected_fraud"]:
                    print(f"   ✅ Fraud detection matches expectation")
                    results.append(True)
                else:
                    print(f"   ⚠️  Fraud detection mismatch: expected {test['expected_fraud']}, got {is_fraud}")
                    results.append(False)
                    
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append(False)
        
        # Summary
        print(f"\n{'=' * 50}")
        print("📋 Test Summary")
        print(f"{'=' * 50}")
        passed = sum(results)
        total = len(results)
        print(f"Passed: {passed}/{total}")
        
        return all(results)
        
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_orchestrator())
    sys.exit(0 if success else 1)
