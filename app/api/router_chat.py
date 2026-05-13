from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession
from pathlib import Path
from uuid import uuid4

from app.api.deps import get_current_api_key, get_db_session
from app.schemas.schemas import ChatCancelRequest, ChatRequest, ChatResponse
from app.services.chat_service import chat_service
from app.services.cancellation import cancellation_manager

from app.ai_core.models.medical_chatbot import SafeVoiceMedicalChatbot
from app.ai_core.serving.dependencies import get_chatbot
from app.ai_core.core.config import get_settings, Settings as AISettings

router = APIRouter(prefix="/chat", tags=["Chat"])
voice_router = APIRouter(prefix="/api", tags=["Voice"])


def _audio_url(audio_path: str | None, settings: AISettings) -> str | None:
    if not audio_path:
        return None
    path = Path(audio_path)
    try:
        rel = path.relative_to(settings.audio_output_dir)
        return f"/artifacts/audio/{rel.as_posix()}"
    except ValueError:
        return None



@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    """
    Send a message and get a response.

    - If `stream=false` (default): returns a full JSON response.
    - If `stream=true`: returns an SSE stream of tokens.
    """
    if req.stream:
        async def token_generator():
            async for token in chat_service.chat_stream(
                req.user_id, req.session_id, req.message, req.context, db
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    result = await chat_service.chat(
        req.user_id, req.session_id, req.message, req.context, db
    )
    return ChatResponse(**result)


@router.post("/cancel")
async def cancel_chat_endpoint(
    req: ChatCancelRequest,
    api_key: str = Depends(get_current_api_key),
):
    """Cancel an active streaming answer."""
    cancelled = await cancellation_manager.cancel(req.user_id, req.session_id)
    return {
        "cancelled": cancelled > 0,
        "cancelled_count": cancelled,
        "message": "Đã gửi yêu cầu dừng." if cancelled else "Không có câu trả lời nào đang chạy.",
    }


@voice_router.post("/stt")
def stt(
    file: UploadFile = File(...),
    bot: SafeVoiceMedicalChatbot = Depends(get_chatbot),
):
    """Speech to Text API."""
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    tmp_dir = Path("/tmp/medical_voice_rag_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"stt_{uuid4().hex}{suffix}"

    with tmp_path.open("wb") as f:
        f.write(file.file.read())

    try:
        text = bot.voice_to_text(tmp_path)
        return {"text": text}
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@voice_router.post("/tts")
def tts(
    text: str,
    bot: SafeVoiceMedicalChatbot = Depends(get_chatbot),
    settings: AISettings = Depends(get_settings),
):
    """Text to Speech API."""
    audio_path = bot._speak(text)
    if not audio_path:
        raise HTTPException(
            status_code=500,
            detail="TTS generation failed or is disabled.",
        )

    return {"audio_url": _audio_url(audio_path, settings)}
