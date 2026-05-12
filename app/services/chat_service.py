from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from sqlalchemy.orm import Session as DBSession

from app.ai_core.serving.dependencies import get_chatbot
from app.db.models import Message
from app.services.cancellation import cancellation_manager, is_cancel_message
from app.services.llm_service import llm_service
from app.services.session_manager import session_manager
from app.services.summarizer import summarizer_service
from app.services.token_service import token_service
from app.services.warning_service import warning_service

logger = logging.getLogger(__name__)


class ChatService:
    """Main chat API orchestrator.

    Rule: /api/v1/chat must pass through the same SafeVoiceMedicalChatbot
    pipeline as /api/predict:

        safety guard -> entity extraction -> KG context -> vector context
        -> context confidence/grounding -> SAFE_SYSTEM_PROMPT_V4 -> LLM
        -> response validator

    This service only adds server features around that medical pipeline:
    API user/session DB, warning keywords from DB, token ledger, persistence,
    cancellation endpoint, and summarization.
    """

    async def chat(
        self,
        user_id: str,
        session_id: str | None,
        message: str,
        context: str | None,
        db: DBSession,
    ) -> dict[str, Any]:
        user = session_manager.get_or_create_user(user_id, db)
        session = session_manager.get_or_create_session(user, session_id, context, db)

        input_warnings = warning_service.detect(db, message)

        if is_cancel_message(message):
            cancelled = await cancellation_manager.cancel(user_id, session.id)
            return self._empty_response(
                session_id=session.id,
                answer="Đã dừng câu trả lời đang chạy." if cancelled else "Không có câu trả lời nào đang chạy để dừng.",
                token_remaining=token_service.usage(db, session.id)["remaining_tokens"],
                warnings=input_warnings,
            )

        cancel_event = await cancellation_manager.start(user_id, session.id)
        try:
            t0 = time.perf_counter()
            history_text = self._build_db_history(session.id, db)
            extra_context = self._build_extra_context(session.context)

            result = await self._run_safevoice_pipeline(
                message=message,
                session_id=session.id,
                history_text=history_text,
                extra_context=extra_context,
                cancel_event=cancel_event,
            )

            if cancel_event.is_set():
                return self._empty_response(
                    session_id=session.id,
                    answer="Đã dừng câu trả lời đang chạy.",
                    token_remaining=token_service.usage(db, session.id)["remaining_tokens"],
                    warnings=input_warnings,
                    result=result,
                )

            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            answer = str(result.get("answer") or "").strip()
            sources = result.get("sources") or []
            source_ids = self._extract_source_ids(sources)

            output_warnings = warning_service.detect(db, answer)
            bot_warnings = [str(w) for w in result.get("warnings") or []]
            warnings = self._merge_warnings(input_warnings, bot_warnings, output_warnings)

            prompt_tokens = token_service.estimate_tokens(
                "\n\n".join([extra_context, history_text, message]),
                llm_service.tokenizer,
            )
            completion_tokens = token_service.estimate_tokens(answer, llm_service.tokenizer)

            _, assistant_msg = session_manager.save_turn(
                session=session,
                user_content=message,
                assistant_content=answer,
                rag_source_ids=source_ids,
                latency_ms=latency_ms,
                db=db,
            )
            token_service.charge_tokens(db, session.id, prompt_tokens, completion_tokens)
            usage = token_service.usage(db, session.id)

            await summarizer_service.maybe_summarize(session.id, db)

            return {
                "session_id": session.id,
                "message_id": assistant_msg.id,
                "answer": answer,
                "rag_sources": sources or None,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "token_remaining": usage["remaining_tokens"],
                "warnings": warnings or None,
                "intent": result.get("intent"),
                "risk_level": result.get("risk_level"),
                "confidence": result.get("confidence"),
                "blocked": bool(result.get("blocked", False)),
                "audio_url": result.get("audio_url") or result.get("audio_path"),
            }
        finally:
            await cancellation_manager.finish(user_id, session.id, cancel_event)

    async def chat_stream(
        self,
        user_id: str,
        session_id: str | None,
        message: str,
        context: str | None,
        db: DBSession,
    ):
        """SSE-compatible stream.

        SafeVoiceMedicalChatbot currently returns a complete response, so this
        method runs the same safe pipeline once and emits the final answer in
        small chunks. The important point is that stream and non-stream use the
        same safety/context/prompt path.
        """
        result = await self.chat(user_id, session_id, message, context, db)
        for warning in result.get("warnings") or []:
            yield f"[WARNING] {warning}\n"

        text = result.get("answer") or ""
        for i in range(0, len(text), 32):
            yield text[i:i + 32]
            await asyncio.sleep(0)

    async def _run_safevoice_pipeline(
        self,
        message: str,
        session_id: str,
        history_text: str,
        extra_context: str,
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        if cancel_event.is_set():
            return {"answer": "", "session_id": session_id, "blocked": False, "warnings": []}

        bot = get_chatbot()
        return await asyncio.to_thread(
            bot.chat,
            message,
            session_id=session_id,
            enable_tts=False,
            history_override=history_text,
            extra_context=extra_context,
        )

    def _build_db_history(self, session_id: str, db: DBSession) -> str:
        from app.core.config import settings
        from app.db.models import Session as ChatSession

        session = db.query(ChatSession).filter_by(id=session_id).first()
        if not session:
            return "Chưa có lịch sử."

        parts: list[str] = []
        if session.summary:
            parts.append(f"Tóm tắt trước đó: {session.summary.strip()}")

        recent_messages = (
            db.query(Message)
            .filter_by(session_id=session_id)
            .filter(Message.turn_index >= session.summary_up_to_turn)
            .order_by(Message.turn_index)
            .limit(settings.MAX_HISTORY_TURNS * 2)
            .all()
        )
        if recent_messages:
            lines = []
            for msg in recent_messages:
                label = "Người dùng" if msg.role == "user" else "Trợ lý"
                lines.append(f"{label}: {msg.content}")
            parts.append("Các lượt gần đây:\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "Chưa có lịch sử."

    def _build_extra_context(self, context: str | None) -> str:
        if not context or not context.strip():
            return "Không có."
        return context.strip()

    def _extract_source_ids(self, sources: list[Any]) -> list[str]:
        ids: list[str] = []
        for source in sources:
            sid = self._source_to_id(source)
            if sid and sid not in ids:
                ids.append(sid)
        return ids

    def _source_to_id(self, source: Any) -> str:
        if isinstance(source, str):
            return source
        if isinstance(source, dict):
            for key in ("id", "source", "title", "name"):
                value = source.get(key)
                if value:
                    return str(value)
            metadata = source.get("metadata")
            if isinstance(metadata, dict):
                for key in ("source", "id", "title", "name"):
                    value = metadata.get(key)
                    if value:
                        return str(value)
            return str(source)[:256]
        return str(source)[:256]

    def _merge_warnings(self, *groups: list[str]) -> list[str]:
        merged: list[str] = []
        for group in groups:
            for item in group or []:
                text = str(item).strip()
                if text and text not in merged:
                    merged.append(text)
        return merged

    def _empty_response(
        self,
        session_id: str,
        answer: str,
        token_remaining: int,
        warnings: list[str] | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = result or {}
        return {
            "session_id": session_id,
            "message_id": "",
            "answer": answer,
            "rag_sources": None,
            "latency_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "token_remaining": token_remaining,
            "warnings": warnings or None,
            "intent": result.get("intent"),
            "risk_level": result.get("risk_level"),
            "confidence": result.get("confidence"),
            "blocked": bool(result.get("blocked", False)),
            "audio_url": result.get("audio_url") or result.get("audio_path"),
        }


chat_service = ChatService()
