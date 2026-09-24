import time
import uuid
from typing import Any, Dict, List, Optional

from app.config import settings
from app.memory.embeddings import (
    BaseEmbeddingProvider,
    ChromaEmbeddingProvider,
    MockEmbeddingProvider,
)
from app.memory.vector_store import (
    BaseVectorStore,
    ChromaVectorStore,
    InMemoryVectorStore,
    VectorSearchResult,
)


class MemoryManager:
    def __init__(self, vector_store: Optional[BaseVectorStore] = None):
        if vector_store:
            self.store_backend = vector_store
        else:
            # Initialize configured store
            if settings.VECTOR_STORE_TYPE.lower() == "chroma":
                try:
                    self.store_backend = ChromaVectorStore(
                        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
                        collection_name=settings.CHROMA_COLLECTION_NAME
                    )
                except Exception:
                    # Fallback to in-memory store if Chroma initialization has disk or dependency issues
                    embedder = MockEmbeddingProvider()
                    self.store_backend = InMemoryVectorStore(embedding_provider=embedder)
            else:
                embedder = MockEmbeddingProvider()
                self.store_backend = InMemoryVectorStore(embedding_provider=embedder)

    def store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        source: str = "execution_result",
        doc_type: str = "task_knowledge",
        doc_id: Optional[str] = None
    ) -> str:
        """Stores a high-value memory snippet with strict metadata."""
        if not text or not text.strip():
            raise ValueError("Cannot store empty memory text")

        record_id = doc_id or f"mem_{uuid.uuid4().hex[:12]}"
        meta = {
            "task_id": task_id or "system",
            "timestamp": time.time(),
            "source": source,
            "type": doc_type,
            **(metadata or {})
        }
        self.store_backend.store(id=record_id, text=text.strip(), metadata=meta)
        return record_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_meta: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search returning formatted memory records."""
        if not query or not query.strip():
            return []
        results = self.store_backend.search(query_text=query.strip(), top_k=top_k, filter_meta=filter_meta)
        return [r.to_dict() for r in results]

    def retrieve_context(self, query: str, top_k: int = 3, score_threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Retrieves relevant contextual snippets for planner before task execution.
        Filters out low-relevance noise.
        """
        raw_results = self.search(query=query, top_k=top_k)
        relevant = [r for r in raw_results if r.get("score", 0.0) >= score_threshold]
        return relevant

    def delete(self, doc_id: str) -> bool:
        """Deletes a memory entry by ID."""
        return self.store_backend.delete(doc_id)

    def clear(self) -> None:
        """Clears all stored memories."""
        self.store_backend.clear()

    def count(self) -> int:
        """Returns total memory document count."""
        return self.store_backend.count()


# Global memory manager instance
memory_manager = MemoryManager()
