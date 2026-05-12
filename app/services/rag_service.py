from __future__ import annotations
import json
import logging
import os
from typing import List, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation using HuggingFace sentence embeddings
    and a FAISS index for fast nearest-neighbour search.

    Documents are stored as plain text chunks with metadata.
    The index is persisted to disk and loaded on startup.
    """

    def __init__(self):
        self._embedder = None
        self._index = None
        self._documents: List[dict] = []     # [{id, text, metadata}]

    def load(self):
        """Load embedder + FAISS index. In mock mode, avoid heavy dependencies."""
        if os.environ.get("USE_MOCK_RETRIEVER", "true").lower() in {"1", "true", "yes", "y"}:
            logger.warning("USE_MOCK_RETRIEVER=true: using mock RAGService for /api/v1 and WebSocket.")
            self._embedder = "mock"
            self._index = "mock"
            self._documents = [{"id": "mock-doc-1", "text": "Mock medical context", "metadata": {}}]
            return

        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            logger.info(f"Loading embedding model from {settings.EMBEDDING_MODEL_PATH} …")
            self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)

            os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            index_path = os.path.join(settings.VECTOR_DB_PATH, "index.faiss")
            docs_path = os.path.join(settings.VECTOR_DB_PATH, "documents.json")

            if os.path.exists(index_path) and os.path.exists(docs_path):
                self._index = faiss.read_index(index_path)
                with open(docs_path, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
                logger.info(f"RAG index loaded: {len(self._documents)} documents.")
            else:
                dim = self._embedder.get_sentence_embedding_dimension()
                self._index = faiss.IndexFlatIP(dim)   # inner product (cosine after norm)
                logger.warning("No existing RAG index found. Starting with empty index.")

        except Exception as e:
            logger.error(f"Failed to load RAG service: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        return self._embedder is not None

    def add_documents(self, docs: List[dict]):
        """
        Add documents to the in-memory index and persist to disk.
        Each doc: {"id": str, "text": str, "metadata": dict}
        """
        import faiss
        import numpy as np

        texts = [d["text"] for d in docs]
        embeddings = self._embedder.encode(texts, normalize_embeddings=True)
        self._index.add(np.array(embeddings, dtype="float32"))
        self._documents.extend(docs)
        self._persist()

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        """
        Return top-k relevant documents for the query.
        Each result: {"id", "text", "metadata", "score"}
        """
        if self._embedder == "mock":
            return [{"id": "mock-doc-1", "text": "Mock medical context", "metadata": {}, "score": 1.0}]
        if not self._embedder or self._index is None or getattr(self._index, "ntotal", 0) == 0:
            return []

        import numpy as np

        k = top_k or settings.RAG_TOP_K
        q_emb = self._embedder.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.array(q_emb, dtype="float32"), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < settings.RAG_SIMILARITY_THRESHOLD:
                continue
            doc = dict(self._documents[idx])
            doc["score"] = round(float(score), 4)
            results.append(doc)
        return results

    def build_context(self, query: str) -> Tuple[str, List[str]]:
        """
        Returns (formatted_context_text, list_of_source_ids).
        """
        if self._embedder == "mock":
            return "### Context:\nDoc 1 text", ["doc-1"]

        docs = self.retrieve(query)
        if not docs:
            return "", []

        lines = ["### Relevant context:"]
        ids = []
        for i, doc in enumerate(docs, 1):
            lines.append(f"[{i}] {doc['text'].strip()}")
            ids.append(doc["id"])
        return "\n".join(lines), ids

    def _persist(self):
        import faiss
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        faiss.write_index(self._index, os.path.join(settings.VECTOR_DB_PATH, "index.faiss"))
        with open(os.path.join(settings.VECTOR_DB_PATH, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)


rag_service = RAGService()
