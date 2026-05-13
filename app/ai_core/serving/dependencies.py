"""
Dependency injection helpers for FastAPI.

Provides a singleton SafeVoiceMedicalChatbot instance that is built once
(using lru_cache) and injected into route handlers via FastAPI Depends().
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.ai_core.core.config import Settings, get_settings
from app.ai_core.models.llm_loader import load_llm
from app.ai_core.models.medical_chatbot import SafeVoiceMedicalChatbot
from app.ai_core.rag.kg import MedicalKG
from app.ai_core.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


def _maybe_raise_or_log(settings: Settings, message: str, exc: Exception | None = None) -> None:
    """Avoid silently falling back to local data when Neo4j was explicitly configured."""
    if settings.has_neo4j and not settings.allow_local_graph_fallback:
        if exc is not None:
            raise RuntimeError(message) from exc
        raise RuntimeError(message)
    if exc is not None:
        logger.warning("%s: %s", message, exc)
    else:
        logger.warning("%s", message)


def _open_kg(settings: Settings) -> MedicalKG | None:
    if not settings.has_neo4j:
        return None

    kg = MedicalKG.from_settings(settings)
    if kg is None or getattr(kg, "driver", None) is None:
        _maybe_raise_or_log(settings, "Neo4j settings exist but driver could not be created")
        return None

    try:
        kg.driver.verify_connectivity()
    except Exception as exc:  # pragma: no cover - needs real Neo4j
        _maybe_raise_or_log(settings, "Neo4j credentials exist but connectivity check failed", exc)
        return None

    return kg


def _build_retriever(settings: Settings) -> Any:
    """
    Build the RAG retriever.

    Production behavior:
    - If Neo4j credentials are configured and USE_NEO4J_CORPUS_FOR_FAISS=true,
      build FAISS/BM25 documents from the Neo4j graph.
    - FAISS auto-rebuilds when the corpus fingerprint changes.
    - If Neo4j is configured but unavailable, do not silently use local seed data
      unless ALLOW_LOCAL_GRAPH_FALLBACK=true.
    """
    from app.ai_core.data.feature_loader import build_corpus_from_neo4j, build_corpus_from_seed_data
    from app.ai_core.rag.retriever import KeywordRetriever

    docs: list[Any] = []
    corpus_source = "seed"

    if settings.has_neo4j and settings.use_neo4j_corpus_for_faiss:
        kg = _open_kg(settings)
        if kg is not None:
            try:
                docs = build_corpus_from_neo4j(kg.driver, database=kg.database)
                corpus_source = "neo4j"
                logger.info("Built RAG corpus from Neo4j: %d docs", len(docs))
            except Exception as exc:  # pragma: no cover - needs real Neo4j
                _maybe_raise_or_log(settings, "Could not build RAG corpus from Neo4j", exc)
                docs = []

            if not docs:
                _maybe_raise_or_log(settings, "Neo4j corpus is empty; refusing to rebuild FAISS from local seed data")

    if not docs:
        docs = build_corpus_from_seed_data()
        corpus_source = "seed"
        logger.info("Built RAG corpus from local seed data: %d docs", len(docs))

    if not docs:
        return RAGRetriever(KeywordRetriever([], k=settings.rerank_top_n))

    return RAGRetriever.build(
        docs,
        settings,
        force_rebuild=settings.rebuild_faiss_on_start,
        corpus_source=corpus_source,
    )


def _build_graph_engine(settings: Settings) -> Any:
    """
    Build KG engine.

    If Neo4j credentials are present, use Neo4j. Do not silently fall back to the
    embedded local graph unless ALLOW_LOCAL_GRAPH_FALLBACK=true.
    """
    from app.ai_core.rag.graph_query import LocalGraphQueryEngine, Neo4jGraphQueryEngine

    if not settings.has_neo4j:
        logger.info("Neo4j is not configured. Using LocalGraphQueryEngine.")
        return LocalGraphQueryEngine()

    try:
        kg = _open_kg(settings)
        if kg is None:
            return LocalGraphQueryEngine()
        return Neo4jGraphQueryEngine(kg.driver, database=kg.database)
    except Exception as exc:  # pragma: no cover - needs real Neo4j
        _maybe_raise_or_log(settings, "Could not build Neo4jGraphQueryEngine", exc)
        return LocalGraphQueryEngine()


def _build_stt(settings: Settings) -> Any:
    if not settings.enable_stt:
        from app.ai_core.voice.stt import NullSTT
        return NullSTT()

    import os
    backend = os.environ.get("STT_BACKEND", "whisper").lower()

    try:
        if backend == "faster_whisper":
            from app.ai_core.voice.stt import FasterWhisperSTT
            model_size = os.environ.get("WHISPER_MODEL_SIZE", "small")
            device = os.environ.get("WHISPER_DEVICE", "cpu")
            logger.info("Loading FasterWhisperSTT (model=%s, device=%s)", model_size, device)
            return FasterWhisperSTT(model_size=model_size, device=device, language="vi")

        elif backend == "zipformer":
            from app.ai_core.voice.stt import ZipformerSTT
            logger.info("Loading ZipformerSTT from %s", settings.stt_model_dir)
            return ZipformerSTT(model_dir=settings.stt_model_dir)

        else:  # default: whisper
            from app.ai_core.voice.stt import WhisperSTT
            model_size = os.environ.get("WHISPER_MODEL_SIZE", "small")
            device = os.environ.get("WHISPER_DEVICE", "cpu")
            logger.info("Loading WhisperSTT (model=%s, device=%s)", model_size, device)
            return WhisperSTT(model_size=model_size, device=device, language="vi")

    except Exception as exc:
        logger.warning("Failed to load STT backend '%s': %s. STT disabled.", backend, exc)
        from app.ai_core.voice.stt import NullSTT
        return NullSTT()


def _build_tts(settings: Settings) -> Any:
    if not settings.enable_tts:
        from app.ai_core.voice.tts import NullTTS
        return NullTTS()

    import os
    backend = os.environ.get("TTS_BACKEND", "edge_tts").lower()

    try:
        if backend == "viettts":
            from app.ai_core.voice.tts import VietTTSEngine
            logger.info("Loading VietTTSEngine")
            return VietTTSEngine()

        elif backend == "gtts":
            from app.ai_core.voice.tts import GTTSEngine
            logger.info("Loading GTTSEngine")
            return GTTSEngine()

        elif backend == "vieneu":
            from app.ai_core.voice.tts import VieNeuTTSEngine
            logger.info("Loading VieNeuTTSEngine (voice=%s)", settings.tts_voice_id)
            return VieNeuTTSEngine(voice_id=settings.tts_voice_id)

        else:  # default: edge_tts
            from app.ai_core.voice.tts import EdgeTTSEngine
            voice = settings.tts_voice_id or "HoaiMy"
            logger.info("Loading EdgeTTSEngine (voice=%s)", voice)
            return EdgeTTSEngine(voice_id=voice)

    except Exception as exc:
        logger.warning("Failed to load TTS backend '%s': %s. TTS disabled.", backend, exc)
        from app.ai_core.voice.tts import NullTTS
        return NullTTS()


@lru_cache(maxsize=1)
def get_chatbot() -> SafeVoiceMedicalChatbot:
    """
    Build and cache a SafeVoiceMedicalChatbot singleton.
    Called once at startup and reused for every request.
    """
    settings = get_settings()
    logger.info("Building SafeVoiceMedicalChatbot …")

    # Use the DB-backed FastAPI LLM service as the single shared LLM backend.
    # This prevents loading two HuggingFace models and makes /api/predict,
    # /api/rag/query and /api/v1/chat all follow the active model config in DB.
    try:
        from app.services.llm_service import llm_service

        if not llm_service.is_loaded:
            llm_service.load()
        llm = llm_service
        logger.info("SafeVoiceMedicalChatbot is using app.services.llm_service.llm_service")
    except Exception as exc:
        logger.warning("Falling back to ai_core load_llm because app llm_service is unavailable: %s", exc)
        llm = load_llm(settings)

    retriever = _build_retriever(settings)
    graph_engine = _build_graph_engine(settings)
    stt_engine = _build_stt(settings)
    tts_engine = _build_tts(settings)

    bot = SafeVoiceMedicalChatbot(
        settings=settings,
        llm=llm,
        retriever=retriever,
        graph_engine=graph_engine,
        stt_engine=stt_engine,
        tts_engine=tts_engine,
    )
    logger.info("SafeVoiceMedicalChatbot ready.")
    return bot


def reset_chatbot_cache() -> None:
    """Clear cached SafeVoiceMedicalChatbot.

    Usually not needed for normal DB model reload because the chatbot holds the
    shared llm_service object. It is still useful when retriever/graph/voice
    settings are changed and a full chatbot rebuild is desired.
    """
    get_chatbot.cache_clear()
