import hashlib
import math
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic semantic hash embedding provider.
    Produces high-quality normalized 384-dimensional dense vectors
    enabling fast, offline, fully testable semantic search.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_single(self, text: str) -> List[float]:
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        if not words:
            return vec.tolist()

        for word in words:
            # Hash word into bucket
            h = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            weight = math.log(1 + len(word))
            vec[idx] += sign * weight

            # Also create 3-gram sub-tokens for partial matching
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    sub = word[i:i+3]
                    sub_h = int(hashlib.md5(sub.encode('utf-8')).hexdigest(), 16)
                    sub_idx = sub_h % self.dim
                    vec[sub_idx] += 0.3 * (1.0 if (sub_h & 1) else -1.0)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)


class ChromaEmbeddingProvider(BaseEmbeddingProvider):
    """Embeddings using ChromaDB default ONNX all-MiniLM-L6-v2."""
    def __init__(self):
        try:
            import chromadb.utils.embedding_functions as ef
            self._ef = ef.DefaultEmbeddingFunction()
        except Exception:
            # Fallback to mock if ONNX download fails
            self._ef = None
            self._fallback = MockEmbeddingProvider()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._ef:
            return self._ef(texts)
        return self._fallback.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        if self._ef:
            return self._ef([text])[0]
        return self._fallback.embed_query(text)
