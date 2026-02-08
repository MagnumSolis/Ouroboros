"""
Vision Adapter - Image processing and OCR
Supports: EasyOCR (local, Hindi/English)
"""

from typing import List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import asyncio

from loguru import logger


@dataclass 
class OCRResult:
    """OCR extraction result"""
    text: str
    boxes: List[dict]  # Bounding boxes with text
    languages: List[str]
    confidence: float


class VisionAdapter:
    """
    Vision processing adapter for OCR
    Uses EasyOCR for Hindi/English text extraction (e.g., khata images)
    """
    
    def __init__(self, languages: List[str] = None):
        """
        Initialize OCR reader
        
        Args:
            languages: List of language codes. Default: ['hi', 'en'] for Hindi+English
        """
        import easyocr
        
        self.languages = languages or ['hi', 'en']
        logger.info(f"Loading EasyOCR with languages: {self.languages}")
        
        # GPU if available, else CPU
        self.reader = easyocr.Reader(self.languages, gpu=True)
        logger.info("EasyOCR initialized")
    
    async def extract_text(
        self,
        image: Union[str, Path, bytes],
        detail: bool = True
    ) -> OCRResult:
        """
        Extract text from image
        
        Args:
            image: Image path or bytes
            detail: If True, include bounding boxes
            
        Returns:
            OCRResult with extracted text and metadata
        """
        # EasyOCR is synchronous, run in thread pool
        if isinstance(image, bytes):
            import numpy as np
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image))
            img_array = np.array(img)
            results = await asyncio.to_thread(
                self.reader.readtext, img_array
            )
        else:
            results = await asyncio.to_thread(
                self.reader.readtext, str(image)
            )
        
        # Process results
        boxes = []
        texts = []
        total_confidence = 0.0
        
        for (bbox, text, confidence) in results:
            boxes.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            })
            texts.append(text)
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        return OCRResult(
            text=" ".join(texts),
            boxes=boxes,
            languages=self.languages,
            confidence=avg_confidence
        )
    
    async def extract_text_simple(
        self,
        image: Union[str, Path, bytes]
    ) -> str:
        """Simple extraction returning just the text"""
        result = await self.extract_text(image, detail=False)
        return result.text
