from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Ép load đúng file /workspace/ai_server/.env
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _resolve_project_path(value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    app_name: str = Field("Medical Voice RAG AI Module", alias="APP_NAME")
    app_version: str = Field("1.0.0", alias="APP_VERSION")
    app_env: str = Field("local", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8000, alias="PORT")

    # Dev/smoke mode
    use_mock_llm: bool = Field(False, alias="USE_MOCK_LLM")
    use_mock_retriever: bool = Field(False, alias="USE_MOCK_RETRIEVER")

    # Neo4j
    neo4j_uri: str | None = Field(None, alias="NEO4J_URI")
    neo4j_user: str | None = Field(None, alias="NEO4J_USER")
    neo4j_password: str | None = Field(None, alias="NEO4J_PASSWORD")
    neo4j_database: str | None = Field(None, alias="NEO4J_DATABASE")

    allow_local_graph_fallback: bool = Field(False, alias="ALLOW_LOCAL_GRAPH_FALLBACK")
    use_neo4j_corpus_for_faiss: bool = Field(True, alias="USE_NEO4J_CORPUS_FOR_FAISS")
    rebuild_faiss_on_start: bool = Field(False, alias="REBUILD_FAISS_ON_START")

    # LLM
    model_name: str = Field("Qwen/Qwen2.5-7B-Instruct", alias="MODEL_NAME")
    model_load_in_4bit: bool = Field(False, alias="MODEL_LOAD_IN_4BIT")
    model_torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        "float16",
        alias="MODEL_TORCH_DTYPE",
    )
    model_max_new_tokens: int = Field(512, alias="MODEL_MAX_NEW_TOKENS")
    model_temperature: float = Field(0.3, alias="MODEL_TEMPERATURE")
    model_repetition_penalty: float = Field(1.1, alias="MODEL_REPETITION_PENALTY")

    # Retrieval
    embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_device: str = Field("cpu", alias="EMBEDDING_DEVICE")
    vectorstore_path: Path = Field(
        PROJECT_ROOT / "artifacts/vectorstore/medical_faiss_v4",
        alias="VECTORSTORE_PATH",
    )
    bm25_k: int = Field(6, alias="BM25_K")
    faiss_k: int = Field(6, alias="FAISS_K")
    rerank_top_n: int = Field(5, alias="RERANK_TOP_N")
    use_hyde: bool = Field(True, alias="USE_HYDE")
    use_cross_encoder: bool = Field(True, alias="USE_CROSS_ENCODER")
    cross_encoder_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="CROSS_ENCODER_MODEL",
    )

    rag_grounding_mode: Literal["off", "warn", "strict"] = Field(
        "strict",
        alias="RAG_GROUNDING_MODE",
    )

    # Voice
    enable_stt: bool = Field(False, alias="ENABLE_STT")
    enable_tts: bool = Field(False, alias="ENABLE_TTS")
    stt_backend: str = Field("whisper", alias="STT_BACKEND")
    whisper_model_size: str = Field("small", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field("cuda", alias="WHISPER_DEVICE")
    stt_model_dir: Path = Field(
        PROJECT_ROOT / "artifacts/models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        alias="STT_MODEL_DIR",
    )
    tts_backend: str = Field("edge_tts", alias="TTS_BACKEND")
    tts_voice_id: str | None = Field("HoaiMy", alias="TTS_VOICE_ID")
    audio_output_dir: Path = Field(
        PROJECT_ROOT / "artifacts/audio",
        alias="AUDIO_OUTPUT_DIR",
    )

    # MLflow
    mlflow_tracking_uri: str = Field(str(PROJECT_ROOT / "mlruns"), alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("medical_voice_rag_v4", alias="MLFLOW_EXPERIMENT_NAME")

    @field_validator("neo4j_database", mode="before")
    @classmethod
    def normalize_neo4j_database(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or normalized.lower() in {"none", "null", "default", "auto"}:
                return None
            return normalized
        return value

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        self.vectorstore_path = _resolve_project_path(self.vectorstore_path)
        self.stt_model_dir = _resolve_project_path(self.stt_model_dir)
        self.audio_output_dir = _resolve_project_path(self.audio_output_dir)

        mlflow_uri = str(self.mlflow_tracking_uri)
        if "://" not in mlflow_uri:
            self.mlflow_tracking_uri = str(_resolve_project_path(mlflow_uri))

        return self

    @property
    def has_neo4j(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_user and self.neo4j_password)

    def ensure_dirs(self) -> None:
        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings