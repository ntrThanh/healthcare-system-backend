import json
import logging
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session as DBSession

from app.api.deps import get_db_session
from app.core.security import validate_ws_api_key
from app.schemas.schemas import WSIncoming, WSOutgoing
from app.services.chat_service import chat_service
from app.services.cancellation import cancellation_manager, is_cancel_message

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    db: DBSession = Depends(get_db_session),
):
    """
    WebSocket endpoint for streaming chat.

    Client can stop the active answer by sending either:
      {"type":"cancel","user_id":"...","session_id":"..."}
    or a chat message whose content is: "tắt", "dừng", "ngắt", "stop", "cancel".
    """
    try:
        await validate_ws_api_key(websocket)
    except Exception:
        return

    await websocket.accept()
    logger.info("WebSocket connection accepted")

    active_task: asyncio.Task | None = None
    active_user_id: str | None = None
    active_session_id: str | None = None

    async def stop_active(reason: str = "cancelled"):
        nonlocal active_task, active_user_id, active_session_id
        if active_user_id:
            await cancellation_manager.cancel(active_user_id, active_session_id)
        if active_task and not active_task.done():
            active_task.cancel()
        await _send(websocket, WSOutgoing(type="cancelled", content=reason, session_id=active_session_id))

    async def stream_answer(incoming: WSIncoming):
        full_tokens = []
        try:
            async for token in chat_service.chat_stream(
                incoming.user_id,
                incoming.session_id,
                incoming.message,
                incoming.context,
                db,
            ):
                if token == "[CANCELLED]":
                    await _send(websocket, WSOutgoing(type="cancelled", content="Đã dừng."))
                    return
                full_tokens.append(token)
                await _send(websocket, WSOutgoing(type="token", content=token, session_id=incoming.session_id))

            await _send(websocket, WSOutgoing(
                type="done",
                session_id=incoming.session_id,
                content="".join(full_tokens).strip(),
            ))
        except asyncio.CancelledError:
            logger.info("WS chat generation cancelled")
        except Exception as e:
            logger.error(f"WS chat error: {e}")
            await _send(websocket, WSOutgoing(type="error", error=str(e)))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                incoming = WSIncoming(**json.loads(raw))
            except Exception as e:
                await _send(websocket, WSOutgoing(type="error", error=f"Invalid message: {e}"))
                continue

            if incoming.type == "ping":
                await _send(websocket, WSOutgoing(type="pong"))
                continue

            if incoming.type == "cancel" or is_cancel_message(incoming.message):
                active_user_id = incoming.user_id or active_user_id
                active_session_id = incoming.session_id or active_session_id
                await stop_active("Đã dừng câu trả lời đang chạy.")
                continue

            if incoming.type != "chat" or not incoming.message:
                await _send(websocket, WSOutgoing(type="error", error="Expected type='chat' with a message"))
                continue

            if active_task and not active_task.done():
                await stop_active("Đã dừng câu trả lời cũ để nhận câu hỏi mới.")

            active_user_id = incoming.user_id
            active_session_id = incoming.session_id
            active_task = asyncio.create_task(stream_answer(incoming))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        if active_user_id:
            await cancellation_manager.cancel(active_user_id, active_session_id)
        if active_task and not active_task.done():
            active_task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


async def _send(ws: WebSocket, msg: WSOutgoing):
    await ws.send_text(msg.model_dump_json(exclude_none=True))
