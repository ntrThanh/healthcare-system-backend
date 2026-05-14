from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# app/core/config.py -> app/core -> app -> ai_server
# parents[2] is the real project root. parents[3] points outside the project.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the project .env no matter where uvicorn/python is started from.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_sqlite_url(url: str) -> str:
    """Resolve sqlite:///./file.db relative to PROJECT_ROOT instead of CWD."""
    if not url.startswith("sqlite:///"):
        return url

    raw_path = url.replace("sqlite:///", "", 1)

    # Special sqlite in-memory DB.
    if raw_path == ":memory:":
        return url

    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return f"sqlite:///{path.as_posix()}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        populate_by_name=True,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "AI Server"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "local"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Security ─────────────────────────────────────────────────────────
    API_KEYS: str = "dev-key-change-me"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8080"
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "mysql+pymysql://healthcare_user:healthcare_password@localhost:3306/healthcare_db?charset=utf8mb4"

    # ── Runtime switches ─────────────────────────────────────────────────
    USE_MOCK_LLM: bool = False
    USE_MOCK_RETRIEVER: bool = False
    SKIP_LEGACY_SERVICES: bool = False

    # ── LLM ──────────────────────────────────────────────────────────────
    LLM_MODEL_PATH: str = "./models/llm"
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_DO_SAMPLE: bool = True
    LLM_DEVICE: str = "cpu"
    LLM_MAX_CONCURRENT_REQUESTS: str = "auto"
    LLM_CONCURRENCY_RESERVED_VRAM_GB: float = 4.0
    LLM_CONCURRENCY_ESTIMATED_REQUEST_VRAM_GB: float = 4.0
    LLM_CONCURRENCY_MAX_AUTO: int = 4

    # HuggingFace model params used to seed DB on first startup.
    MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_LOAD_IN_4BIT: bool = False
    MODEL_TORCH_DTYPE: str = "float16"
    MODEL_MAX_NEW_TOKENS: int = 512
    MODEL_TEMPERATURE: float = 0.3
    MODEL_REPETITION_PENALTY: float = 1.1

    # ── Summarizer LLM ───────────────────────────────────────────────────
    SUMMARIZER_MODEL_PATH: str = "./models/llm"
    SUMMARIZER_MAX_TOKENS: int = 256

    # ── RAG / legacy vector service ──────────────────────────────────────
    # Keep backward-compatible names while also supporting .env names used by ai_core.
    EMBEDDING_MODEL_PATH: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    VECTOR_DB_PATH: str = Field(
        "artifacts/vectorstore/medical_faiss_v4",
        alias="VECTORSTORE_PATH",
    )
    RAG_TOP_K: int = 3
    RAG_SIMILARITY_THRESHOLD: float = 0.5

    # ── Conversation ─────────────────────────────────────────────────────
    MAX_HISTORY_TURNS: int = 10
    SUMMARIZE_AFTER_TURNS: int = 6

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        self.DATABASE_URL = _resolve_sqlite_url(self.DATABASE_URL)

        # Resolve only real filesystem paths. HuggingFace repo IDs contain '/'
        # but should remain unchanged, e.g. Qwen/Qwen2.5-7B-Instruct.
        if self.LLM_MODEL_PATH.startswith("./") or self.LLM_MODEL_PATH.startswith("../"):
            self.LLM_MODEL_PATH = str(_resolve_project_path(self.LLM_MODEL_PATH))
        if self.SUMMARIZER_MODEL_PATH.startswith("./") or self.SUMMARIZER_MODEL_PATH.startswith("../"):
            self.SUMMARIZER_MODEL_PATH = str(_resolve_project_path(self.SUMMARIZER_MODEL_PATH))

        self.VECTOR_DB_PATH = str(_resolve_project_path(self.VECTOR_DB_PATH))
        return self

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.API_KEYS.split(",") if k.strip()}

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


settings = Settings()
