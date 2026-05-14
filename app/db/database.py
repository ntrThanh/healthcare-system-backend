from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session as DBSession
from app.core.config import settings
from app.db.models import Base


def _quote_mysql_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def _ensure_database_exists(database_url: str) -> None:
    url = make_url(database_url)
    if not url.drivername.startswith("mysql") or not url.database:
        return

    database_name = url.database
    server_url = url.set(database=None)
    server_engine = create_engine(
        server_url,
        connect_args={"charset": "utf8mb4"},
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    try:
        with server_engine.begin() as conn:
            conn.execute(text(
                "CREATE DATABASE IF NOT EXISTS "
                f"{_quote_mysql_identifier(database_name)} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            ))
    finally:
        server_engine.dispose()


def _engine_options(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        return {
            "connect_args": {"check_same_thread": False},
            "pool_pre_ping": True,
        }
    if database_url.startswith("mysql"):
        return {
            "connect_args": {"charset": "utf8mb4"},
            "pool_pre_ping": True,
            "pool_recycle": 3600,
        }
    return {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }


_ensure_database_exists(settings.DATABASE_URL)
engine = create_engine(settings.DATABASE_URL, **_engine_options(settings.DATABASE_URL))

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables on startup and seed default rows from .env."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_default_model_config(db)
        seed_default_warning_keywords(db)
    finally:
        db.close()


def seed_default_model_config(db: DBSession):
    """Seed the first HuggingFace model config from .env if DB is empty."""
    from app.db.models import AIModelConfig

    exists = db.query(AIModelConfig).first()
    if exists:
        return

    def get_setting(*names, default=None):
        for name in names:
            if hasattr(settings, name):
                value = getattr(settings, name)
                if value is not None:
                    return value
        return default

    model = AIModelConfig(
        name="default",
        provider="huggingface",
        model_name=str(get_setting("MODEL_NAME", "model_name", "LLM_MODEL_PATH", default="Qwen/Qwen2.5-7B-Instruct")),
        is_active=True,
        load_in_4bit=bool(get_setting("MODEL_LOAD_IN_4BIT", "model_load_in_4bit", default=False)),
        torch_dtype=str(get_setting("MODEL_TORCH_DTYPE", "model_torch_dtype", default="float16")),
        device=str(get_setting("LLM_DEVICE", "llm_device", default="auto")),
        max_new_tokens=int(get_setting("MODEL_MAX_NEW_TOKENS", "LLM_MAX_NEW_TOKENS", default=512)),
        temperature=float(get_setting("MODEL_TEMPERATURE", "LLM_TEMPERATURE", default=0.3)),
        repetition_penalty=float(get_setting("MODEL_REPETITION_PENALTY", default=1.1)),
        do_sample=float(get_setting("MODEL_TEMPERATURE", "LLM_TEMPERATURE", default=0.3)) > 0,
        trust_remote_code=True,
        description="Seeded automatically from .env.example/.env",
    )
    db.add(model)
    db.commit()


def seed_default_warning_keywords(db: DBSession):
    """Seed a basic dangerous-keyword list. Can be edited by API later."""
    from app.db.models import WarningKeyword

    if db.query(WarningKeyword).first():
        return

    warning = (
        "Cảnh báo: Nội dung bạn gửi có dấu hiệu nguy hiểm hoặc tự làm hại bản thân/người khác. "
        "Hãy liên hệ người thân, giáo viên hoặc cơ sở y tế gần nhất để được hỗ trợ ngay."
    )
    phrases = [
        "tôi muốn tự tử",
        "muốn tự tử",
        "tự tử",
        "tự sát",
        "tự hại",
        "tự làm đau mình",
        "không muốn sống",
        "kết thúc cuộc đời",
        "kill myself",
        "suicide",
        "self harm",
        "hurt myself",
        "end my life",
        "i want to die",
    ]
    for phrase in phrases:
        db.add(WarningKeyword(phrase=phrase, warning_message=warning, severity="danger"))
    db.commit()


def get_db():
    """FastAPI dependency — yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
