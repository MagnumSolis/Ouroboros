#!/usr/bin/env python3
"""
Quick test for audio processing pipeline
"""
import asyncio
import sys
sys.path.insert(0, "/mnt/Magnum_Data/Code/ouroboros/sahayak")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.adapters.speech import SpeechAdapter
from src.adapters.audio_processor import AudioProcessor, AudioResult


async def test_audio_pipeline():
    """Test the audio processing pipeline"""
    
    print("=" * 60)
    print("Testing Audio Processing Pipeline")
    print("=" * 60)
    
    # Check for sample audio file
    sample_files = [
        Path("/mnt/Magnum_Data/Code/ouroboros/sahayak/data/fraud_calls"),
        Path("/mnt/Magnum_Data/Code/ouroboros/sahayak/data")
    ]
    
    audio_file = None
    for folder in sample_files:
        if folder.exists():
            for f in folder.glob("*.wav"):
                audio_file = f
                break
            if audio_file:
                break
    
    if not audio_file:
        print("No sample audio file found. Testing with SpeechAdapter only.")
        speech = SpeechAdapter()
        print(f"✅ SpeechAdapter initialized with: {speech.provider_type}")
        return
    
    print(f"Using audio file: {audio_file}")
    
    # Initialize
    print("\n1. Initializing AudioProcessor...")
    processor = AudioProcessor()
    print("   ✅ AudioProcessor initialized")
    
    # Read audio
    print("\n2. Reading audio file...")
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    print(f"   ✅ Read {len(audio_bytes)} bytes")
    
    # Process
    print("\n3. Processing audio...")
    try:
        result: AudioResult = await processor.process(audio_bytes)
        print(f"   ✅ Transcription: {result.text[:100]}...")
        print(f"   ✅ Language: {result.language}")
        print(f"   ✅ Emotion: {result.emotion} (confidence: {result.emotion_confidence:.2f})")
        print(f"   ✅ Transcription confidence: {result.transcription_confidence:.2f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_audio_pipeline())
