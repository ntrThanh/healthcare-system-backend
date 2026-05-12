from sqlalchemy import (
    Column, String, Text, Integer, DateTime, Boolean, ForeignKey, Float
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


def gen_uuid() -> str:
    return str(uuid.uuid4())


class APIClient(Base):
    """
    Registered API clients (web apps, mobile apps, etc.).
    Each client has one or more API keys stored in .env,
    but we track usage metadata here.
    """
    __tablename__ = "api_clients"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(128), nullable=False)      
    api_key_hint = Column(String(16), nullable=False)  
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

    sessions = relationship("Session", back_populates="client", cascade="all, delete")


class User(Base):
    """
    End-users who chat with the AI.
    Identified by an external user_id supplied by the caller.
    """
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    external_id = Column(String(256), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    last_seen_at = Column(DateTime, onupdate=func.now())

    sessions = relationship("Session", back_populates="user", cascade="all, delete")


class Session(Base):
    """
    A conversation session. One user can have multiple sessions.
    Each session maintains its own rolling summary.
    """
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    client_id = Column(String(36), ForeignKey("api_clients.id"), nullable=True)
    title = Column(String(256), nullable=True)          # auto-generated from first msg
    summary = Column(Text, nullable=True)               # rolling conversation summary
    summary_up_to_turn = Column(Integer, default=0)     # how many turns are summarized
    context = Column(Text, nullable=True)               # optional system context/persona
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="sessions")
    client = relationship("APIClient", back_populates="sessions")
    messages = relationship(
        "Message", back_populates="session",
        cascade="all, delete", order_by="Message.turn_index"
    )


class Message(Base):
    """
    Individual chat turns. role = 'user' | 'assistant'.
    """
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)           # "user" | "assistant"
    content = Column(Text, nullable=False)
    turn_index = Column(Integer, nullable=False)        # 0-based within session
    rag_sources = Column(Text, nullable=True)           # JSON list of retrieved doc ids
    latency_ms = Column(Float, nullable=True)           # generation latency
    created_at = Column(DateTime, server_default=func.now())

    session = relationship("Session", back_populates="messages")


class AIModelConfig(Base):
    """
    HuggingFace model configuration stored in DB.
    One active row is used by LLMService when loading/generating.
    """
    __tablename__ = "ai_model_configs"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(128), nullable=False, unique=True, index=True)
    provider = Column(String(64), default="huggingface")
    model_name = Column(String(512), nullable=False)
    is_active = Column(Boolean, default=False, index=True)
    is_loaded = Column(Boolean, default=False)

    load_in_4bit = Column(Boolean, default=False)
    torch_dtype = Column(String(32), default="float16")
    device = Column(String(32), default="auto")
    trust_remote_code = Column(Boolean, default=True)

    max_new_tokens = Column(Integer, default=512)
    temperature = Column(Float, default=0.3)
    repetition_penalty = Column(Float, default=1.1)
    do_sample = Column(Boolean, default=True)

    description = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class SessionTokenLedger(Base):
    """
    Token ledger per session.
    Positive delta = tokens added through API.
    Negative delta = estimated tokens consumed by chat.
    """
    __tablename__ = "session_token_ledger"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    delta_tokens = Column(Integer, nullable=False)
    reason = Column(String(256), nullable=False, default="manual")
    meta = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    session = relationship("Session")


class WarningKeyword(Base):
    """
    Dangerous keyword / phrase detector configuration.
    Used to attach warnings to chat responses.
    """
    __tablename__ = "warning_keywords"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    phrase = Column(String(256), nullable=False, unique=True, index=True)
    language = Column(String(32), default="mixed")
    severity = Column(String(32), default="danger")
    warning_message = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
