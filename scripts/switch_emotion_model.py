#!/usr/bin/env python3
"""
Switch Emotion Model Script
============================

Easily switch between different voice emotion detection models.
Run: python scripts/switch_emotion_model.py

Available models:
1. superb/wav2vec2-base-superb-er (4 emotions, 378MB, FAST)
2. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (7 emotions, 1.27GB)
3. r-f/wav2vec-english-speech-emotion-recognition (8 emotions, 1.27GB - may have compatibility issues)
"""

import os
import sys
import re

EMOTION_PY = os.path.join(os.path.dirname(__file__), "..", "src", "adapters", "emotion.py")

MODELS = {
    "1": {
        "name": "superb/wav2vec2-base-superb-er",
        "desc": "4 emotions (neutral, happy, angry, sad) - 378MB - RECOMMENDED",
        "emotions": ["neutral", "happy", "angry", "sad"]
    },
    "2": {
        "name": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "desc": "7 emotions - 1.27GB - May need download",
        "emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    },
    "3": {
        "name": "r-f/wav2vec-english-speech-emotion-recognition",
        "desc": "8 emotions - 1.27GB - Compatibility issues",
        "emotions": ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    }
}


def get_current_model():
    """Read current model from emotion.py"""
    with open(EMOTION_PY, 'r') as f:
        content = f.read()
    
    match = re.search(r'def __init__\(self, model_name: str = "([^"]+)"\)', content)
    if match:
        return match.group(1)
    return "unknown"


def set_model(model_name: str):
    """Update emotion.py with new model"""
    with open(EMOTION_PY, 'r') as f:
        content = f.read()
    
    # Replace the default model
    new_content = re.sub(
        r'def __init__\(self, model_name: str = "[^"]+"\)',
        f'def __init__(self, model_name: str = "{model_name}")',
        content
    )
    
    with open(EMOTION_PY, 'w') as f:
        f.write(new_content)


def main():
    print("\n" + "="*60)
    print(" 🎤 Voice Emotion Model Switcher")
    print("="*60)
    
    current = get_current_model()
    print(f"\n📍 Current model: {current}\n")
    
    print("Available models:")
    print("-" * 50)
    for key, model in MODELS.items():
        marker = "✅ " if model['name'] == current else "   "
        print(f"{marker}[{key}] {model['desc']}")
        print(f"       Model: {model['name']}")
        print(f"       Emotions: {', '.join(model['emotions'])}")
        print()
    
    choice = input("Enter choice (1-3) or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        print("Cancelled.")
        return
    
    if choice not in MODELS:
        print(f"❌ Invalid choice: {choice}")
        return
    
    selected = MODELS[choice]
    
    if selected['name'] == current:
        print(f"ℹ️ Already using this model!")
        return
    
    print(f"\n🔄 Switching to: {selected['name']}")
    set_model(selected['name'])
    print("✅ Model switched!")
    print("\n⚠️  Restart the dashboard to apply changes:")
    print("   streamlit run dashboard.py")


if __name__ == "__main__":
    main()
