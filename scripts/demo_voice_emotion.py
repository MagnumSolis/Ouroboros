#!/usr/bin/env python3
"""
Demo Script: Voice Emotion Detection
=====================================

Tests the EmotionAdapter with a sample audio file or records live audio.
Demonstrates voice-based sentiment analysis using Wav2Vec2.

Run: python scripts/demo_voice_emotion.py [audio_file.wav]
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def test_with_file(filepath: str):
    """Test emotion detection with an audio file"""
    from src.adapters.emotion import EmotionAdapter
    
    print_header("🎤 Voice Emotion Detection Demo")
    print(f"\nLoading audio: {filepath}")
    
    # Initialize adapter
    print("\n⏳ Loading Wav2Vec2 model (first run downloads ~360MB)...")
    adapter = EmotionAdapter()
    
    # Predict emotion
    print("\n🔍 Analyzing audio...")
    result = adapter.predict(filepath)
    
    # Display results
    print_header("📊 Results")
    
    emotion_icons = {
        'angry': '😠', 'calm': '😌', 'disgust': '🤢', 'fear': '😨',
        'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
    }
    
    icon = emotion_icons.get(result['emotion'], '🎭')
    print(f"\n{icon} Detected Emotion: {result['emotion'].upper()}")
    print(f"📈 Confidence: {result['confidence']*100:.1f}%")
    print(f"🤖 Model: {result.get('model', 'unknown')}")
    
    print("\n📊 All Emotion Scores:")
    sorted_scores = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)
    for emotion, score in sorted_scores:
        bar = "█" * int(score * 20)
        icon = emotion_icons.get(emotion, '🎭')
        print(f"   {icon} {emotion:10s}: {bar:20s} {score*100:.1f}%")
    
    if result.get('error'):
        print(f"\n⚠️ Note: {result['error']}")
    
    return result


def test_with_sample():
    """Create a simple test with sample audio (if available)"""
    # Check for any existing audio files
    sample_dirs = [
        Path("data/audio"),
        Path("data/test_audio"),
        Path("data"),
    ]
    
    for dir_path in sample_dirs:
        if dir_path.exists():
            audio_files = list(dir_path.glob("*.wav")) + list(dir_path.glob("*.mp3"))
            if audio_files:
                return test_with_file(str(audio_files[0]))
    
    print("\n⚠️ No audio file provided or found.")
    print("\nUsage: python scripts/demo_voice_emotion.py <audio_file.wav>")
    print("\nTip: Record a short audio clip (2-10 seconds) expressing an emotion!")
    return None


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            test_with_file(filepath)
        else:
            print(f"❌ File not found: {filepath}")
            sys.exit(1)
    else:
        test_with_sample()


if __name__ == "__main__":
    main()
