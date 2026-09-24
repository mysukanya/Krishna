import os
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from app.config import settings
from app.memory.embeddings import BaseEmbeddingProvider, ChromaEmbeddingProvider, MockEmbeddingProvider


class VectorSearchResult:
    def __init__(self, id: str, text: str, metadata: Dict[str, Any], distance: float, score: float):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.distance = distance
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "distance": self.distance,
            "score": self.score
        }


class BaseVectorStore(ABC):
    @abstractmethod
    def store(self, id: str, text: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None) -> None:
        pass

    @abstractmethod
    def search(self, query_text: str, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass


class InMemoryVectorStore(BaseVectorStore):
    """
    Lightweight zero-dependency in-memory/cosine similarity vector store.
    Ideal for testing, development, and standalone execution.
    """
    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.documents: Dict[str, Dict[str, Any]] = {}

    def store(self, id: str, text: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None) -> None:
        if embedding is None:
            embedding = self.embedding_provider.embed_query(text)
        self.documents[id] = {
            "id": id,
            "text": text,
            "metadata": metadata,
            "embedding": np.array(embedding, dtype=np.float32)
        }

    def search(self, query_text: str, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        if not self.documents:
            return []

        q_vec = np.array(self.embedding_provider.embed_query(query_text), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-6:
            q_vec = q_vec / q_norm

        results = []
        for doc_id, doc in self.documents.items():
            # Check metadata filter
            if filter_meta:
                match = all(doc["metadata"].get(k) == v for k, v in filter_meta.items())
                if not match:
                    continue

            d_vec = doc["embedding"]
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 1e-6:
                d_vec = d_vec / d_norm

            sim = float(np.dot(q_vec, d_vec))
            # Distance: 1 - cosine_similarity
            distance = max(0.0, 1.0 - sim)
            results.append(VectorSearchResult(
                id=doc_id,
                text=doc["text"],
                metadata=doc["metadata"],
                distance=distance,
                score=sim
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(self, id: str) -> bool:
        if id in self.documents:
            del self.documents[id]
            return True
        return False

    def clear(self) -> None:
        self.documents.clear()

    def count(self) -> int:
        return len(self.documents)


class ChromaVectorStore(BaseVectorStore):
    """Production vector store implementation backed by ChromaDB."""
    def __init__(self, persist_directory: str, collection_name: str):
        import chromadb
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store(self, id: str, text: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None) -> None:
        # ChromaDB requires all metadata values to be primitives (str, int, float, bool)
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = json.dumps(v)

        kwargs = {
            "ids": [id],
            "documents": [text],
            "metadatas": [clean_meta]
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding]

        self.collection.upsert(**kwargs)

    def search(self, query_text: str, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        if self.count() == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(top_k, self.count())
        }
        if filter_meta:
            kwargs["where"] = filter_meta

        res = self.collection.query(**kwargs)
        results = []
        if res and res.get("ids") and len(res["ids"]) > 0:
            ids = res["ids"][0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [[]])[0] if res.get("distances") else [0.0] * len(ids)

            for i in range(len(ids)):
                dist = distances[i] if i < len(distances) else 0.0
                # Chroma uses squared L2 metric by default for normalized vectors (0 to 2)
                score = max(0.0, 1.0 - (dist / 2.0))
                results.append(VectorSearchResult(
                    id=ids[i],
                    text=docs[i] if i < len(docs) else "",
                    metadata=metas[i] if i < len(metas) else {},
                    distance=dist,
                    score=score
                ))
        return results

    def delete(self, id: str) -> bool:
        try:
            self.collection.delete(ids=[id])
            return True
        except Exception:
            return False

    def clear(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def count(self) -> int:
        return self.collection.count()
