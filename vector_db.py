from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ── CONFIGURATION ────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
EMBED_MODEL       = "nomic-embed-text"
LLM_MODEL         = "llama3.2"
EMBED_DIMENSION   = 768
COLLECTION_NAME   = "rag_documents"
QDRANT_PATH       = "./qdrant_storage"
# ─────────────────────────────────────────────────────


def _configure_settings():
    """Tell LlamaIndex to use Ollama instead of OpenAI."""
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_qdrant_client() -> QdrantClient:
    """Create a local Qdrant client (stores data on disk — no Docker!)."""
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient):
    """Create the collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBED_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        print(f"[VectorDB] Created collection: '{COLLECTION_NAME}'")
    else:
        print(f"[VectorDB] Collection '{COLLECTION_NAME}' already exists.")


def ingest_documents(documents):
    """Embed and store documents in Qdrant."""
    _configure_settings()
    client = get_qdrant_client()
    ensure_collection(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[VectorDB] Ingesting {len(documents)} document(s)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("[VectorDB] Done!")
    return index


def load_index() -> VectorStoreIndex:
    """Load a previously ingested index from Qdrant."""
    _configure_settings()
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )


def get_query_engine(index: VectorStoreIndex, similarity_top_k: int = 3):
    """Build a query engine that retrieves top-k similar chunks."""
    return index.as_query_engine(similarity_top_k=similarity_top_k)