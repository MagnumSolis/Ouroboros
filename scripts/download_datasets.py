
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import librosa
from loguru import logger

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = Path(__file__).parent.parent / "data"

def download_robocall_dataset():
    """Download Robocall Audio Dataset from GitHub"""
    target_dir = DATA_DIR / "fraud_calls"
    if target_dir.exists():
        logger.info(f"✅ Robocall dataset already exists at {target_dir}")
        return

    logger.info("⏳ Downloading Robocall Audio Dataset...")
    repo_url = "https://github.com/wspr-ncsu/robocall-audio-dataset.git"
    
    try:
        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
        logger.info("✅ Successfully downloaded Robocall dataset")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to download dataset: {e}")

def prepare_bhashini_instructions():
    """Instructions for Bhashini dataset"""
    print("\n" + "=" * 50)
    print("🇮🇳 Bhashini Dataset Instructions")
    print("=" * 50)
    print("1. Go to https://vatika.bhashini.co.in/#/dataVatika")
    print("2. Login/Register and download 'Audio' datasets for Hindi")
    print(f"3. Extract them to {DATA_DIR}/hindi_speech")
    print("=" * 50 + "\n")

def load_audio_dataset(directory: Path, label: str):
    """
    Load audio files from a directory and assign a label
    Returns: (features, labels)
    """
    features = []
    labels = []
    
    # We need EmotionAdapter to use its feature extractor
    from sahayak.adapters.emotion import EmotionAdapter
    adapter = EmotionAdapter()
    
    logger.info(f"Loading files from {directory} with label '{label}'...")
    
    files = list(directory.glob("*.wav")) + list(directory.glob("*.mp3"))
    
    for f in files[:50]: # Limit for demo
        try:
            # Extract features (1, 128, 130, 1)
            feat = adapter._extract_features(str(f))
            features.append(feat[0]) # Remove batch dim
            labels.append(label)
        except Exception as e:
            logger.warning(f"Error loading {f}: {e}")
            
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    download_robocall_dataset()
    prepare_bhashini_instructions()
    
    # Example usage for fine-tuning
    if (DATA_DIR / "fraud_calls").exists():
        logger.info("Ready for fine-tuning! Use the data in sahaya/data/")
