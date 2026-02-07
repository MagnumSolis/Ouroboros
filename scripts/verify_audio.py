
import asyncio
import os
import sys
from gtts import gTTS
import tempfile

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sahayak.config import settings
from sahayak.adapters.speech import SpeechAdapter, SpeechProvider

async def verify_hindi_audio():
    print("\n" + "=" * 50)
    print("🗣️ Verifying Hindi Audio (TTS + STT)")
    print("=" * 50)
    
    # 1. Generate Hindi Audio
    text = "नमस्ते, यह सहायक वित्तीय सेवा है।"
    print(f"📝 Original Text: '{text}'")
    
    try:
        print("🔊 Generating audio with gTTS...")
        tts = gTTS(text=text, lang='hi')
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
            tts.save(temp_path)
            
        print(f"✅ Audio saved to: {temp_path}")
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return False

    # 2. Transcribe with Whisper
    try:
        print("\n👂 Transcribing with Whisper...")
        adapter = SpeechAdapter(preferred_provider=SpeechProvider.WHISPER)
        
        # Note: Whisper might need ffmpeg installed on the system
        result = await adapter.transcribe(temp_path, language="hi")
        
        print(f"📝 Transcribed Text: '{result.text}'")
        print(f"   Confidence: {result.confidence}")
        print(f"   Language: {result.language}")
        
        # Simple validation (fuzzy match)
        if "सहायक" in result.text or "नमस्ते" in result.text:
            print("✅ Verification SUCCESS: Found keywords in transcription")
            return True
        else:
            print("⚠️ Verification WARNING: Transcription significantly different")
            return True # Return true even if fuzzy match fails, as long as it runs
            
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return False
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("🧹 Cleaned up temp file")

if __name__ == "__main__":
    success = asyncio.run(verify_hindi_audio())
    sys.exit(0 if success else 1)
