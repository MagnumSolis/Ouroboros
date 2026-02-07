
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Input
from typing import Dict, Any, List, Union
from loguru import logger
import io

class EmotionAdapter:
    """
    Speech Emotion Recognition (SER) Adapter
    Uses CNN based model trained on mel-spectrograms
    """
    
    # RAVDESS Emotions (standard mapping)
    EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self._build_model()
        
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
            logger.info(f"✅ Loaded Emotion Model from {model_path}")
        else:
            logger.warning("⚠️ No pre-trained emotion model found. Using initialized weights (random).")

    def _build_model(self):
        """
        Build the CNN architecture suitable for SER
        Input: (128, 130, 1) Mel-Spectrogram
        """
        inputs = Input(shape=(128, 130, 1))
        
        # Block 1
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Block 2
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Block 3
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Block 4
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Classification
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(len(self.EMOTIONS), activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=x)
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def load_weights(self, path: str):
        try:
            self.model.load_weights(path)
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")

    def save_weights(self, path: str):
        self.model.save_weights(path)
        logger.info(f"Saved model weights to {path}")

    def _extract_features(self, audio_data: Union[str, io.BytesIO], duration: float = 2.5):
        """
        Extract Mel-Spectrogram features
        Fixed duration of 2.5s to match input shape (128, 130, 1)
        """
        # Load audio
        # If string, it's a path. If BytesIO, load directly.
        y, sr = librosa.load(audio_data, duration=duration, offset=0.5)
        
        # Pad or truncate to desired length
        # 2.5s * 22050Hz = ~55125 samples
        target_len = int(duration * sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
            
        # Extract Mel Spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Ensure exact shape (128, 130)
        # librosa output shape depends on hop_length (default 512)
        # 55125 / 512 ~= 108 frames. We need 130.
        # Let's resize or pad.
        
        if mel_spect.shape[1] < 130:
            mel_spect = np.pad(mel_spect, ((0, 0), (0, 130 - mel_spect.shape[1])))
        else:
            mel_spect = mel_spect[:, :130]
            
        # Add channel dimension
        mel_spect = mel_spect[..., np.newaxis]
        
        return np.expand_dims(mel_spect, axis=0) # Batch dimension

    def predict_from_buffer(self, audio_buffer: io.BytesIO) -> Dict[str, Any]:
        """Predict emotion from audio buffer"""
        try:
            features = self._extract_features(audio_buffer)
            predictions = self.model.predict(features, verbose=0)
            
            idx = np.argmax(predictions)
            emotion = self.EMOTIONS[idx]
            confidence = float(predictions[0][idx])
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "all_scores": {e: float(s) for e, s in zip(self.EMOTIONS, predictions[0])}
            }
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            return {"emotion": "neutral", "confidence": 0.0}

    def fine_tune(self, train_data, train_labels, epochs=5, batch_size=32):
        """
        Fine-tune the model on new data (e.g., Hindi dataset)
        """
        logger.info(f"Starting fine-tuning for {epochs} epochs...")
        
        # Freeze early layers? Or train all?
        # For similar domain (audio), training all with low LR is usually fine.
        
        history = self.model.fit(
            train_data, 
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
        return history.history
