from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = 'default'
    enable_tts: bool = False


class ChatResponse(BaseModel):
    answer: str
    session_id: str = 'default'
    intent: str
    risk_level: str
    confidence: str
    blocked: bool = False
    warnings: list[str] = Field(default_factory=list)
    audio_url: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app_name: str
    env: str
    model_name: str
    neo4j_configured: bool
    voice_enabled: dict[str, bool]
