"""
Document Processing Tool - PDF, text, and structured data processing

For ingesting and processing financial documents:
- RBI circulars and guidelines
- Government scheme PDFs
- User-uploaded bank statements
- Khata/ledger images
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ProcessedDocument:
    """Processed document with extracted information"""
    id: str
    title: str
    content: str
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    document_type: str = "unknown"


class DocumentProcessor:
    """
    Document processing for various file types
    
    Supports:
    - PDF documents
    - Text files
    - HTML pages
    - Structured data (JSON, CSV)
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Try to import optional dependencies
        self._pdf_available = False
        try:
            import pypdf
            self._pdf_available = True
        except ImportError:
            logger.warning("pypdf not available, PDF processing disabled")
    
    async def process_file(
        self,
        file_path: Union[str, Path],
        document_type: Optional[str] = None
    ) -> ProcessedDocument:
        """
        Process a file and extract text content
        
        Args:
            file_path: Path to the file
            document_type: Optional type hint (policy, scheme, etc.)
            
        Returns:
            ProcessedDocument with extracted content
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            content = await self._process_pdf(path)
        elif suffix in [".txt", ".md"]:
            content = await self._process_text(path)
        elif suffix == ".json":
            content = await self._process_json(path)
        else:
            content = await self._process_text(path)
        
        # Chunk the content
        chunks = self._chunk_text(content)
        
        return ProcessedDocument(
            id=str(path.stem),
            title=path.stem,
            content=content,
            chunks=chunks,
            metadata={
                "source_path": str(path),
                "file_type": suffix,
                "size_bytes": path.stat().st_size
            },
            document_type=document_type or self._detect_type(content)
        )
    
    async def _process_pdf(self, path: Path) -> str:
        """Extract text from PDF"""
        if not self._pdf_available:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")
        
        import pypdf
        
        text_parts = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        
        return "\n\n".join(text_parts)
    
    async def _process_text(self, path: Path) -> str:
        """Read text file"""
        return path.read_text(encoding="utf-8")
    
    async def _process_json(self, path: Path) -> str:
        """Convert JSON to readable text"""
        import json
        
        data = json.loads(path.read_text(encoding="utf-8"))
        
        if isinstance(data, dict):
            return self._dict_to_text(data)
        elif isinstance(data, list):
            return "\n\n".join(self._dict_to_text(item) for item in data if isinstance(item, dict))
        else:
            return str(data)
    
    def _dict_to_text(self, d: Dict[str, Any], prefix: str = "") -> str:
        """Convert dictionary to readable text"""
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._dict_to_text(value, prefix + "  "))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind("।")  # Hindi sentence end
                if last_period == -1:
                    last_period = chunk.rfind(".")
                if last_period > self.chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return chunks
    
    def _detect_type(self, content: str) -> str:
        """Detect document type from content"""
        content_lower = content.lower()
        
        if "rbi" in content_lower and "circular" in content_lower:
            return "policy"
        elif any(scheme in content_lower for scheme in ["pm-kisan", "mudra", "jan dhan", "pmjdy"]):
            return "scheme"
        elif "faq" in content_lower or "frequently asked" in content_lower:
            return "faq"
        elif "eligibility" in content_lower and "benefits" in content_lower:
            return "scheme"
        else:
            return "general"
    
    async def process_url(self, url: str) -> ProcessedDocument:
        """
        Fetch and process a URL
        
        Args:
            url: Web URL to process
            
        Returns:
            ProcessedDocument with extracted content
        """
        try:
            import httpx
            from bs4 import BeautifulSoup
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30)
                response.raise_for_status()
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string if soup.title else url
            
            chunks = self._chunk_text(text)
            
            return ProcessedDocument(
                id=url.replace("/", "_").replace(":", "_")[:50],
                title=title,
                content=text,
                chunks=chunks,
                metadata={"source_url": url},
                document_type=self._detect_type(text)
            )
            
        except ImportError:
            raise RuntimeError("httpx and beautifulsoup4 required. Run: pip install httpx beautifulsoup4")
        except Exception as e:
            logger.error(f"URL processing error: {e}")
            raise
