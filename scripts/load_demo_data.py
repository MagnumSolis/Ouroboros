
import sys
import os
import asyncio
from pathlib import Path
from loguru import logger
import uuid
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sahayak.memory.manager import MemoryManager
from sahayak.memory.schemas import KnowledgeDocument
from sahayak.tools.document import DocumentProcessor

async def load_data():
    """Load demo data into Qdrant"""
    
    # Initialize Memory Manager
    memory = MemoryManager(auto_create_collections=True)
    processor = DocumentProcessor()
    logger.info("✅ Memory Manager & Processor Initialized")
    
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return

    files = list(data_dir.glob("*.txt"))
    if not files:
        logger.warning("⚠️ No text files found in data directory")
        return
        
    logger.info(f"📂 Found {len(files)} documents to ingest...")
    
    count = 0
    for file_path in files:
        try:
            logger.info(f"Processing {file_path.name}...")
            
            # Process document
            processed_doc = await processor.process_file(file_path)
            
            # Create KnowledgeDocument (Schema expected by MemoryManager)
            doc = KnowledgeDocument(
                id=str(uuid.uuid4()),
                title=processed_doc.title,
                content=processed_doc.content,
                document_type=processed_doc.document_type,
                tags=["demo", "financial_literacy"],
                source_url=f"file://{file_path.name}",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                language="en"
            )
            
            # Ingest into 'knowledge_base' collection
            doc_id = await memory.store_knowledge(doc)
            
            if doc_id:
                logger.info(f"✅ Ingested: {file_path.name} (ID: {doc_id})")
                count += 1
            else:
                logger.error(f"❌ Failed to ingest: {file_path.name}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            
    logger.info(f"🎉 Demo Data Load Complete! Processed {count}/{len(files)} files.")

if __name__ == "__main__":
    asyncio.run(load_data())
