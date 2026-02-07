"""
Collection Definitions for Qdrant Blackboard

Based on research paper specifications:
- episodic_memory: RAM-based HNSW for fast access
- knowledge_base: Disk-based with quantization (4x memory reduction)
- fraud_patterns: RAM-based HNSW for real-time detection
- working_memory: RAM-based for agent state
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationConfig,
)


@dataclass
class CollectionConfig:
    """Configuration for a Qdrant collection"""
    name: str
    description: str
    vector_size: int
    distance: Distance = Distance.COSINE
    on_disk: bool = False  # RAM by default for speed
    # HNSW parameters for ANN search
    hnsw_m: int = 16  # Number of edges per node
    hnsw_ef_construct: int = 100  # Construction time/accuracy trade-off
    # Quantization for memory efficiency (4x reduction per Bosch case study)
    use_quantization: bool = False
    quantization_type: str = "scalar"  # scalar or binary


# Collection configurations aligned with research paper Table 2
# Using 384 dimensions for all-MiniLM-L6-v2 (local default)
# If using Cohere, dimension is 1024 (updated dynamically)

COLLECTIONS: Dict[str, CollectionConfig] = {
    "episodic_memory": CollectionConfig(
        name="episodic_memory",
        description="User interactions and agent activity logs",
        vector_size=384,
        on_disk=False,     # RAM for sub-millisecond access
        hnsw_m=16,
        hnsw_ef_construct=100,
        use_quantization=False,  # Keep full precision for recent memories
    ),
    
    "knowledge_base": CollectionConfig(
        name="knowledge_base",
        description="RBI policies, government schemes, FAQs",
        vector_size=384,
        on_disk=True,      # Disk-based for larger storage
        hnsw_m=32,         # More edges for better recall
        hnsw_ef_construct=200,
        use_quantization=True,  # 4x memory reduction per Bosch
    ),
    
    "fraud_patterns": CollectionConfig(
        name="fraud_patterns",
        description="Known fraud signatures, phishing scripts, threat vectors",
        vector_size=384,
        on_disk=False,     # RAM for real-time detection (per Flipkart)
        hnsw_m=16,
        hnsw_ef_construct=100,
        use_quantization=False,  # Full precision for security
    ),
    
    "working_memory": CollectionConfig(
        name="working_memory",
        description="Current task state and intermediate agent results",
        vector_size=384,
        on_disk=False,     # Must be fast for agent coordination
        hnsw_m=8,          # Smaller index, fewer entries
        hnsw_ef_construct=50,
        use_quantization=False,
    ),
    
    "semantic_cache": CollectionConfig(
        name="semantic_cache",
        description="Cached LLM responses for similar queries",
        vector_size=384,
        on_disk=False,     # RAM for fast cache lookup
        hnsw_m=16,
        hnsw_ef_construct=100,
        use_quantization=False,
    ),
}


def get_vector_params(config: CollectionConfig) -> VectorParams:
    """Get Qdrant VectorParams from collection config"""
    return VectorParams(
        size=config.vector_size,
        distance=config.distance,
        on_disk=config.on_disk,
        hnsw_config=HnswConfigDiff(
            m=config.hnsw_m,
            ef_construct=config.hnsw_ef_construct,
        ),
    )


def get_quantization_config(config: CollectionConfig) -> Optional[QuantizationConfig]:
    """Get quantization config for storage optimization"""
    if not config.use_quantization:
        return None
    
    # Scalar quantization: Float32 -> Int8 (4x reduction)
    return ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,  # Clip outliers
            always_ram=True,  # Keep quantized vectors in RAM
        )
    )

