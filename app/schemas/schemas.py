from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# ── Request schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Caller's unique user identifier")
    session_id: Optional[str] = Field(None, description="Existing session ID; None = create new")
    message: str = Field(..., min_length=1, max_length=4096)
    context: Optional[str] = Field(None, description="System context / persona override")
    stream: bool = Field(False, description="Stream tokens via SSE (REST) or WS frames")


class SessionCreateRequest(BaseModel):
    user_id: str
    context: Optional[str] = None
    title: Optional[str] = None


class SessionResetRequest(BaseModel):
    keep_context: bool = Field(True, description="Retain the system context when resetting")


class ChatCancelRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = Field(None, description="Cancel one session. None = cancel all active replies of this user")


# ── Response schemas ──────────────────────────────────────────────────────────

class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    turn_index: int
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    rag_sources: Optional[List[Any]] = None
    latency_ms: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_remaining: int = 0
    warnings: Optional[List[str]] = None
    intent: Optional[str] = None
    risk_level: Optional[str] = None
    confidence: Optional[str] = None
    blocked: bool = False
    audio_url: Optional[str] = None


class SessionOut(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    summary: Optional[str]
    context: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    sessions: List[SessionOut]
    total: int


# ── WebSocket message envelope ────────────────────────────────────────────────

class WSIncoming(BaseModel):
    """Message sent by the client over WebSocket."""
    type: str = "chat"                   # "chat" | "cancel" | "ping"
    user_id: str
    session_id: Optional[str] = None
    message: str = ""
    context: Optional[str] = None


class WSOutgoing(BaseModel):
    """Message sent by the server over WebSocket."""
    type: str                            # "token" | "warning" | "done" | "cancelled" | "error" | "pong"
    session_id: Optional[str] = None
    content: Optional[str] = None       # token chunk or full answer
    rag_sources: Optional[List[str]] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    token_remaining: Optional[int] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    llm_loaded: bool
    rag_loaded: bool

# ── Model config API ─────────────────────────────────────────────────────────

class ModelConfigBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    provider: str = "huggingface"
    model_name: str = Field(..., min_length=1, max_length=512)
    is_active: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "float16"
    device: str = "auto"
    trust_remote_code: bool = True
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    repetition_penalty: float = Field(1.1, ge=0.1, le=5.0)
    do_sample: bool = True
    description: Optional[str] = None


class ModelConfigCreate(ModelConfigBase):
    pass


class ModelConfigUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    provider: Optional[str] = None
    model_name: Optional[str] = Field(None, min_length=1, max_length=512)
    is_active: Optional[bool] = None
    load_in_4bit: Optional[bool] = None
    torch_dtype: Optional[str] = None
    device: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    max_new_tokens: Optional[int] = Field(None, ge=1, le=8192)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.1, le=5.0)
    do_sample: Optional[bool] = None
    description: Optional[str] = None


class ModelConfigOut(ModelConfigBase):
    id: str
    is_loaded: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    models: List[ModelConfigOut]
    total: int


# ── Token API ────────────────────────────────────────────────────────────────

class TokenAddRequest(BaseModel):
    session_id: str
    tokens: int = Field(..., gt=0)
    reason: str = "manual_add"


class TokenUsageResponse(BaseModel):
    session_id: str
    added_tokens: int
    used_tokens: int
    remaining_tokens: int


# ── Warning keyword API ──────────────────────────────────────────────────────

class WarningKeywordBase(BaseModel):
    phrase: str = Field(..., min_length=1, max_length=256)
    language: str = "mixed"
    severity: str = "danger"
    warning_message: str
    is_active: bool = True


class WarningKeywordCreate(WarningKeywordBase):
    pass


class WarningKeywordUpdate(BaseModel):
    phrase: Optional[str] = Field(None, min_length=1, max_length=256)
    language: Optional[str] = None
    severity: Optional[str] = None
    warning_message: Optional[str] = None
    is_active: Optional[bool] = None


class WarningKeywordOut(WarningKeywordBase):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
