#!/usr/bin/env python3
"""
Test script for API adapters
Verifies connectivity and basic functionality
"""

import asyncio
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sahayak.config import settings
from sahayak.adapters import LLMAdapter, EmbeddingAdapter, ChatMessage


async def test_llm():
    """Test LLM adapter"""
    print("\n" + "=" * 50)
    print("🧠 Testing LLM Adapter")
    print("=" * 50)
    
    adapter = LLMAdapter()
    
    print(f"Available providers: {adapter.available_providers}")
    
    if not adapter.available_providers:
        print("❌ No LLM providers configured!")
        print("   Set GROQ_API_KEY or GEMINI_API_KEY in .env")
        return False
    
    try:
        response = await adapter.simple_chat(
            prompt="Say 'Hello from Sahayak!' in exactly 5 words.",
            system_prompt="You are a helpful assistant. Be concise."
        )
        print(f"✅ Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_embeddings():
    """Test embedding adapter"""
    print("\n" + "=" * 50)
    print("📊 Testing Embedding Adapter")
    print("=" * 50)
    
    adapter = EmbeddingAdapter()
    
    print(f"Provider: {adapter.provider_type}")
    print(f"Dimension: {adapter.dimension}")
    print(f"Is local: {adapter.is_local}")
    
    try:
        # Test single embedding
        text = "Financial fraud detection system"
        embedding = await adapter.embed_single(text)
        print(f"✅ Embedded text: '{text[:30]}...'")
        print(f"   Vector shape: {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Test batch embedding
        texts = ["loan application", "credit score", "fraud alert"]
        embeddings = await adapter.embed(texts)
        print(f"✅ Batch embedded {len(embeddings)} texts")
        
        # Test similarity
        sim = adapter.cosine_similarity(embeddings[0], embeddings[1])
        print(f"   Similarity between '{texts[0]}' and '{texts[1]}': {sim:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_config():
    """Test configuration loading"""
    print("\n" + "=" * 50)
    print("⚙️ Testing Configuration")
    print("=" * 50)
    
    print(f"Groq configured: {settings.has_groq}")
    print(f"Gemini configured: {settings.has_gemini}")
    print(f"Cohere configured: {settings.has_cohere}")
    print(f"Deepgram configured: {settings.has_deepgram}")
    print(f"Qdrant URL: {settings.qdrant_url}")
    print(f"Log level: {settings.log_level}")
    
    return True


async def main():
    """Run all tests"""
    print("\n🛡️ SAHAYAK - Adapter Tests")
    print("=" * 50)
    
    results = []
    
    results.append(("Config", await test_config()))
    results.append(("Embeddings", await test_embeddings()))
    results.append(("LLM", await test_llm()))
    
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
