"""
a particle embedding utility module
"""

import random

# Handle ChromaDB dependency gracefully
try:
    from chromadb.utils.embedding_functions import EmbeddingFunction
    CHROMADB_AVAILABLE = True
except ImportError:
    # Fallback base class when ChromaDB not available
    class EmbeddingFunction:
        def __call__(self, documents):
            raise NotImplementedError("Subclasses must implement __call__")
    CHROMADB_AVAILABLE = False

class ParticleLikeEmbedding(EmbeddingFunction):
    def __init__(self):
        self.chromadb_available = CHROMADB_AVAILABLE
        
    def __call__(self, documents: list[str]) -> list[list[float]]:
        embeddings = []
        for doc in documents:
            seed = sum(ord(c) for c in doc)
            random.seed(seed)
            vec = [random.uniform(-1, 1) for _ in range(12)]
            embeddings.append(vec)
        return embeddings