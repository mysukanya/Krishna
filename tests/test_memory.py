import pytest
from app.memory.embeddings import MockEmbeddingProvider
from app.memory.vector_store import InMemoryVectorStore
from app.memory.memory_manager import MemoryManager


@pytest.fixture
def clean_memory():
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore(embedding_provider=embedder)
    return MemoryManager(vector_store=store)


def test_memory_store_and_count(clean_memory):
    doc_id = clean_memory.store(
        text="Quarterly net profit was $45,000 in Q3.",
        metadata={"quarter": "Q3"},
        source="unit_test"
    )
    assert doc_id.startswith("mem_")
    assert clean_memory.count() == 1


def test_memory_semantic_search(clean_memory):
    clean_memory.store("Kubernetes deployment guide and cluster scaling.", {"topic": "devops"})
    clean_memory.store("Annual revenue report showing 20% margin.", {"topic": "finance"})

    results = clean_memory.search(query="cluster deployment and scaling", top_k=1)
    assert len(results) == 1
    assert "Kubernetes" in results[0]["text"]
    assert results[0]["score"] > 0.0


def test_memory_retrieve_context(clean_memory):
    clean_memory.store("System architecture uses FastAPI and ChromaDB.", {"type": "architecture"})
    context = clean_memory.retrieve_context(query="FastAPI architecture", top_k=2)
    assert len(context) >= 1
    assert "FastAPI" in context[0]["text"]


def test_memory_deletion(clean_memory):
    doc_id = clean_memory.store("Temporary note to delete.", {})
    assert clean_memory.count() == 1
    assert clean_memory.delete(doc_id) is True
    assert clean_memory.count() == 0


def test_memory_clear(clean_memory):
    clean_memory.store("Note 1", {})
    clean_memory.store("Note 2", {})
    assert clean_memory.count() == 2
    clean_memory.clear()
    assert clean_memory.count() == 0
