
import os
import io
import numpy as np
from typing import Dict, Any, Union, Optional
from loguru import logger

# Check for transformers availability
try:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, pipeline
    import librosa
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers/torch not installed. Voice emotion will use fallback mode.")


class EmotionAdapter:
    """
    Speech Emotion Recognition (SER) Adapter
    
    Uses pre-trained model from Hugging Face for emotion detection.
    Default: superb/wav2vec2-base-superb-er (~360MB, fast download)
    
    Emotions: neu (neutral), hap (happy), ang (angry), sad
    """
    
    # Emotion labels - will be updated from model config
    EMOTIONS = ['neutral', 'happy', 'angry', 'sad']
    
    # Map model labels to readable names
    LABEL_MAP = {
        'neu': 'neutral',
        'hap': 'happy', 
        'ang': 'angry',
        'sad': 'sad',
        'neutral': 'neutral',
        'happy': 'happy',
        'angry': 'angry',
        'fear': 'fear',
        'disgust': 'disgust',
        'surprise': 'surprise',
        'calm': 'calm'
    }
    
    def __init__(self, model_name: str = "superb/wav2vec2-base-superb-er"):
        """
        Initialize the emotion adapter with a pre-trained HuggingFace model.
        
        Args:
            model_name: HuggingFace model ID or local path
        """
        self.model = None
        self.feature_extractor = None
        self.model_name = model_name
        self.device = "cpu"
        
        if HF_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Running in fallback mode - install transformers and torch for voice emotion")
    
    def _load_model(self):
        """Load the pre-trained Wav2Vec2 model for emotion recognition"""
        try:
            logger.info(f"Loading emotion model: {self.model_name}")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
            
            logger.info(f"Using device: {self.device}")
            
            # Load feature extractor and model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Update emotion labels from model config if available
            if hasattr(self.model.config, 'id2label'):
                self.EMOTIONS = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            
            logger.info(f"✅ Emotion model loaded successfully! Emotions: {self.EMOTIONS}")
            
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            self.model = None
    
    def _load_audio(self, audio_data: Union[str, bytes, io.BytesIO], target_sr: int = 16000) -> np.ndarray:
        """
        Load audio from file path or buffer and resample to target sample rate.
        
        Args:
            audio_data: File path, bytes, or BytesIO buffer
            target_sr: Target sample rate (Wav2Vec2 expects 16kHz)
            
        Returns:
            Audio waveform as numpy array
        """
        if isinstance(audio_data, str):
            # File path
            y, sr = librosa.load(audio_data, sr=target_sr)
        elif isinstance(audio_data, bytes):
            # Raw bytes
            y, sr = librosa.load(io.BytesIO(audio_data), sr=target_sr)
        elif isinstance(audio_data, io.BytesIO):
            # BytesIO buffer
            audio_data.seek(0)
            y, sr = librosa.load(audio_data, sr=target_sr)
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
        
        return y
    
    def predict(self, audio_data: Union[str, bytes, io.BytesIO]) -> Dict[str, Any]:
        """
        Predict emotion from audio data.
        
        Args:
            audio_data: Audio file path, bytes, or BytesIO buffer
            
        Returns:
            Dict with detected emotion, confidence, and all scores
        """
        # Fallback if model not available
        if not HF_AVAILABLE or self.model is None:
            return self._fallback_prediction()
        
        try:
            # Load and process audio
            waveform = self._load_audio(audio_data)
            
            # Feature extraction
            inputs = self.feature_extractor(
                waveform, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get prediction
            probs_np = probs.cpu().numpy()[0]
            predicted_idx = np.argmax(probs_np)
            emotion = self.EMOTIONS[predicted_idx]
            confidence = float(probs_np[predicted_idx])
            
            return {
                "emotion": self.LABEL_MAP.get(emotion, emotion),
                "confidence": confidence,
                "all_scores": {self.LABEL_MAP.get(e, e): float(s) for e, s in zip(self.EMOTIONS, probs_np)},
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            return self._fallback_prediction(error=str(e))
    
    def predict_from_buffer(self, audio_buffer: io.BytesIO) -> Dict[str, Any]:
        """Predict emotion from audio buffer (backward compatibility)"""
        return self.predict(audio_buffer)
    
    def _fallback_prediction(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return neutral prediction when model is not available"""
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "all_scores": {e: 0.0 for e in self.EMOTIONS},
            "model": "fallback",
            "error": error or "Model not loaded"
        }
    
    # Keep backward compatibility methods
    def load_weights(self, path: str):
        """Backward compatibility - not needed for HF models"""
        logger.info("load_weights() is deprecated for HuggingFace models")
    
    def save_weights(self, path: str):
        """Backward compatibility - not needed for HF models"""
        logger.info("save_weights() is deprecated for HuggingFace models")
    
    def fine_tune(self, train_data, train_labels, epochs=5, batch_size=32):
        """Backward compatibility - fine-tuning would require additional setup"""
        logger.warning("fine_tune() requires manual setup for HuggingFace models")
        return {}
