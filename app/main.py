import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.db.database import init_db
from app.api import router_chat, router_websocket, router_session, router_model_config, router_tokens, router_warnings
from app.services.llm_service import llm_service
from app.services.rag_service import rag_service
from app.ai_core.core.config import get_settings as get_ai_settings
from app.ai_core.core.config import Settings as AISettings
from app.ai_core.serving.dependencies import get_chatbot

from app.services.llm_service import llm_service
from app.services.summarizer import summarizer_service

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== AI Server starting up ===")

    init_db()
    logger.info("Database initialized and default model/warning configs seeded.")

    # Load app LLM/RAG services after DB is ready so LLM can read model params from DB.
    logger.info("Loading app LLM service from active DB model config...")
    llm_service.load()
    summarizer_service.set_llm(llm_service)
    logger.info(f"App LLM loaded: {llm_service.is_loaded}")

    logger.info("Loading app RAG service...")
    skip_legacy = str(getattr(settings, "SKIP_LEGACY_SERVICES", "false")).lower() == "true"

    if not skip_legacy:
        logger.info("Loading app RAG service...")
        rag_service.load()
        logger.info(f"App RAG loaded: {rag_service.is_loaded}")
    else:
        logger.info("SKIP_LEGACY_SERVICES=true: skip legacy RAG service.")
    logger.info(f"App RAG loaded: {rag_service.is_loaded}")

    # Init AI core used by /api/v1/chat, /api/stt, /api/tts.
    ai_settings = get_ai_settings()
    ai_settings.audio_output_dir.mkdir(parents=True, exist_ok=True)

    get_chatbot()
    logger.info("AI Module models loaded.")

    logger.info("=== AI Server ready ===")
    yield

    logger.info("=== AI Server shutting down ===")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_settings = get_ai_settings()
ai_settings.audio_output_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/artifacts/audio",
    StaticFiles(directory=str(ai_settings.audio_output_dir)),
    name="audio",
)

app.include_router(router_chat.router, prefix="/api/v1")
app.include_router(router_chat.voice_router)
app.include_router(router_websocket.router)
app.include_router(router_session.router, prefix="/api/v1")
app.include_router(router_model_config.router, prefix="/api/v1")
app.include_router(router_tokens.router, prefix="/api/v1")
app.include_router(router_warnings.router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
def health(ai_settings: AISettings = Depends(get_ai_settings)):
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "llm_loaded": llm_service.is_loaded,
        "rag_loaded": rag_service.is_loaded,
        "ai_core_status": {
            "app_name": ai_settings.app_name,
            "env": ai_settings.app_env,
            "model_name": ai_settings.model_name if not ai_settings.use_mock_llm else "mock",
            "retriever": "mock" if ai_settings.use_mock_retriever else "hybrid",
            "neo4j_configured": ai_settings.has_neo4j,
            "voice_enabled": {
                "stt": ai_settings.enable_stt,
                "tts": ai_settings.enable_tts,
            },
        },
    }