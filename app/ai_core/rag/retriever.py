from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from app.ai_core.core.config import Settings

logger = logging.getLogger(__name__)


def _stable_doc_payload(doc: Any) -> dict[str, Any]:
    metadata = getattr(doc, 'metadata', {}) or {}
    content = getattr(doc, 'page_content', str(doc))
    return {
        'page_content': content,
        'metadata': metadata,
    }


def _fingerprint_docs(docs: list[Any]) -> str:
    h = hashlib.sha256()
    for payload in sorted((_stable_doc_payload(d) for d in docs), key=lambda x: (x['metadata'].get('type', ''), x['metadata'].get('id', ''), x['page_content'])):
        h.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        logger.warning('Could not read FAISS corpus metadata %s: %s', path, exc)
    return None


def _save_json(path: Path, data: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    except Exception as exc:
        logger.warning('Could not write FAISS corpus metadata %s: %s', path, exc)


def _cuda_device_id(device: str | None) -> int:
    if not device:
        return 0
    parts = str(device).split(':', maxsplit=1)
    if len(parts) == 2 and parts[0].lower() == 'cuda':
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def _maybe_move_faiss_to_gpu(vectorstore: Any, settings: Settings) -> Any:
    if not settings.use_faiss_gpu:
        return vectorstore

    device = settings.embedding_device
    if not str(device).lower().startswith('cuda'):
        logger.warning('USE_FAISS_GPU=true but EMBEDDING_DEVICE=%s is not CUDA; keeping FAISS on CPU.', device)
        return vectorstore

    try:
        import faiss

        if not hasattr(faiss, 'StandardGpuResources'):
            logger.warning('USE_FAISS_GPU=true but installed FAISS has no GPU support; install faiss-gpu to enable it.')
            return vectorstore

        resources = faiss.StandardGpuResources()
        device_id = _cuda_device_id(device)
        vectorstore.index = faiss.index_cpu_to_gpu(resources, device_id, vectorstore.index)
        vectorstore._faiss_gpu_resources = resources
        logger.info('Moved FAISS index to GPU device %s.', device_id)
    except Exception as exc:
        logger.warning('Could not move FAISS index to GPU; keeping CPU FAISS: %s', exc)
    return vectorstore


class KeywordRetriever:
    """No-dependency fallback retriever used in smoke tests."""

    def __init__(self, docs: list[Any], k: int = 5):
        self.docs = docs
        self.k = k

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t.strip('.,;:!?()[]{}"\'').lower() for t in text.split() if len(t.strip()) >= 2}

    def invoke(self, query: str) -> list[Any]:
        q_tokens = self._tokens(query)
        scored: list[tuple[float, Any]] = []
        for doc in self.docs:
            text = getattr(doc, 'page_content', str(doc))
            d_tokens = self._tokens(text)
            overlap = len(q_tokens & d_tokens)
            conf = float(getattr(doc, 'metadata', {}).get('source_confidence', 0.75))
            score = overlap + math.log1p(len(text)) * 0.01 + conf * 0.1
            if overlap > 0:
                scored.append((score, doc))
        if not scored:
            return self.docs[: self.k]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.k]]

    def get_relevant_documents(self, query: str) -> list[Any]:
        return self.invoke(query)


class RAGRetriever:
    """Factory for Hybrid BM25 + FAISS + CrossEncoder retriever."""

    def __init__(self, retriever: Any, vectorstore: Any | None = None):
        self.retriever = retriever
        self.vectorstore = vectorstore

    def invoke(self, query: str) -> list[Any]:
        if hasattr(self.retriever, 'invoke'):
            return self.retriever.invoke(query)
        return self.retriever.get_relevant_documents(query)

    @classmethod
    def build(
        cls,
        docs: list[Any],
        settings: Settings,
        force_rebuild: bool = False,
        corpus_source: str = 'unknown',
    ) -> 'RAGRetriever':
        if settings.use_mock_retriever:
            logger.warning('USE_MOCK_RETRIEVER=true: using keyword retriever.')
            return cls(KeywordRetriever(docs, k=settings.rerank_top_n))

        from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': settings.embedding_device},
            encode_kwargs={'normalize_embeddings': True},
        )

        vectorstore_path = Path(settings.vectorstore_path)
        meta_path = vectorstore_path / 'corpus_meta.json'
        desired_meta = {
            'corpus_source': corpus_source,
            'doc_count': len(docs),
            'fingerprint': _fingerprint_docs(docs),
            'embedding_model': settings.embedding_model,
        }

        existing_meta = _load_json(meta_path)
        index_exists = (vectorstore_path / 'index.faiss').exists() and (vectorstore_path / 'index.pkl').exists()
        meta_matches = existing_meta == desired_meta

        if index_exists and meta_matches and not force_rebuild:
            logger.info('Loading FAISS index from %s (source=%s, docs=%d)', vectorstore_path, corpus_source, len(docs))
            vectorstore = FAISS.load_local(
                str(vectorstore_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            if force_rebuild:
                reason = 'forced by REBUILD_FAISS_ON_START=true'
            elif not index_exists:
                reason = 'index missing'
            elif not existing_meta:
                reason = 'metadata missing'
            else:
                reason = 'corpus fingerprint changed'
            logger.info('Building FAISS index with %d documents from %s (%s)', len(docs), corpus_source, reason)
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(str(vectorstore_path))
            _save_json(meta_path, desired_meta)

        vectorstore = _maybe_move_faiss_to_gpu(vectorstore, settings)

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = settings.bm25_k
        faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': settings.faiss_k})
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.40, 0.60],
        )

        if settings.use_cross_encoder:
            cross_encoder_device = settings.cross_encoder_device or settings.embedding_device
            cross_encoder = HuggingFaceCrossEncoder(
                model_name=settings.cross_encoder_model,
                model_kwargs={'device': cross_encoder_device},
            )
            reranker = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_n)
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=hybrid_retriever,
            )
        else:
            retriever = hybrid_retriever

        return cls(retriever=retriever, vectorstore=vectorstore)


def format_docs(docs: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    sources: list[dict[str, Any]] = []
    for doc in docs:
        metadata = getattr(doc, 'metadata', {}) or {}
        content = getattr(doc, 'page_content', str(doc))
        lines.append(
            f"[{metadata.get('type', '')} | conf={float(metadata.get('source_confidence', 0)):.2f}] {content}"
        )
        sources.append({
            'type': metadata.get('type'),
            'id': metadata.get('id'),
            'name': metadata.get('name'),
            'source_confidence': metadata.get('source_confidence'),
            'preview': content[:240],
        })
    return '\n\n'.join(lines), sources
