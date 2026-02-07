
import os
import asyncio
from gtts import gTTS
from io import BytesIO
from loguru import logger
from typing import Optional

class TTSAdapter:
    """
    Text-to-Speech Adapter
    Uses gTTS (Google Text-to-Speech) as default provider
    """
    
    def __init__(self):
        logger.info("✅ TTS Adapter initialized (Provider: gTTS)")

    async def synthesize(self, text: str, language: str = "hi") -> Optional[BytesIO]:
        """
        Convert text to speech
        Returns: BytesIO object containing MP3 audio
        """
        try:
            # Map language codes if necessary
            # gTTS uses standard iso codes (hi, en) which matches our system
            lang = language.lower()
            if lang not in ['hi', 'en']:
                lang = 'en' # Fallback
            
            # gTTS is synchronous and network blocking, run in thread
            audio_data = await asyncio.to_thread(self._generate_audio, text, lang)
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    def _generate_audio(self, text: str, lang: str) -> BytesIO:
        """Blocking gTTS call"""
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp

    async def save_to_file(self, text: str, filepath: str, language: str = "hi") -> bool:
        """Synthesize and save directly to file"""
        try:
            audio = await self.synthesize(text, language)
            if audio:
                with open(filepath, "wb") as f:
                    f.write(audio.read())
                return True
            return False
        except Exception as e:
            logger.error(f"TTS save failed: {e}")
            return False
