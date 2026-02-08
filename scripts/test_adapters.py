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

from src.config import settings
from src.adapters import LLMAdapter, EmbeddingAdapter, ChatMessage, LLMProvider


async def test_llm():
    """Test LLM adapter"""
    print("\n" + "=" * 50)
    print("🧠 Testing LLM Adapter")
    print("=" * 50)
    
    adapter = LLMAdapter()
    
    # Check configured providers
    print(f"Perplexity configured: {settings.has_perplexity}")
    print(f"Groq configured: {settings.has_groq}")
    print(f"OpenRouter configured: {settings.has_openrouter}")
    
    if not (settings.has_groq or settings.has_perplexity):
        print("❌ No primary LLM providers configured!")
        print("   Set GROQ_API_KEY or PERPLEXITY_API_KEY in .env")
        return False
    
    try:
        # Test Simple Chat (uses default priority)
        print("\nSending query...")
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
    
    # Infer provider type from private attribute or implementation
    if hasattr(adapter, 'provider') and hasattr(adapter.provider, 'model'):
        print(f"Provider Model: {adapter.provider.model}")
    print(f"Dimension: {adapter.dimension}")
    
    if not settings.has_cohere:
        print("⚠️  WARNING: COHERE_API_KEY not found. Using Local fallback.")
        print("   To test Cohere embeddings, add COHERE_API_KEY to .env")
    else:
        print(f"✅ Using Cohere Embeddings")
    
    try:
        # Test single embedding
        text = "Financial fraud detection system"
        embedding = await adapter.embed_query(text)
        print(f"✅ Embedded text: '{text[:30]}...'")
        print(f"   Vector shape: {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Test batch embedding
        texts = ["loan application", "credit score", "fraud alert"]
        embeddings = await adapter.embed(texts)
        print(f"✅ Batch embedded {len(embeddings)} texts")
        
        # Test similarity (Manual Calculation)
        import numpy as np
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
        sim = cosine_similarity(embeddings[0], embeddings[1])
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
    print(f"Perplexity configured: {settings.has_perplexity}")
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
