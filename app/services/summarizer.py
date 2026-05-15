from __future__ import annotations
import asyncio
import logging
from typing import List

from app.core.config import settings
from app.db.models import Session as ChatSession, Message

logger = logging.getLogger(__name__)


SUMMARIZE_PROMPT_TEMPLATE = """You are a conversation summarizer.
Given the conversation history below, write a concise summary (3-5 sentences) that captures:
- The main topics discussed
- Key information or decisions made
- Any unresolved questions

Previous summary (if any):
{prev_summary}

New messages to incorporate:
{messages}

Write only the summary, nothing else."""


class SummarizerService:
    """
    Runs asynchronously after each user turn.
    Reads unsummarized messages, calls the LLM, and updates session.summary.
    """

    def __init__(self):
        # Reuse the main LLM service for summarization by default.
        # If you have a lighter model, load it separately here.
        self._llm = None

    def set_llm(self, llm_service):
        self._llm = llm_service

    async def maybe_summarize(self, session_id: str):
        """
        Called after each assistant reply.
        Triggers summarization if enough new turns have accumulated.
        Runs in a background asyncio task to avoid blocking the response.
        """
        asyncio.create_task(self._summarize_task(session_id))

    async def _summarize_task(self, session_id: str):
        try:
            await asyncio.to_thread(self._do_summarize, session_id)
        except Exception as e:
            logger.error(f"Summarizer error for session {session_id}: {e}")

    def _do_summarize(self, session_id: str):
        """Blocking summarization — runs in thread pool."""
        from app.db.database import SessionLocal

        db = SessionLocal()
        try:
            session: ChatSession = db.query(ChatSession).filter_by(id=session_id).first()
            if not session:
                return

            all_messages: List[Message] = (
                db.query(Message)
                .filter_by(session_id=session_id)
                .order_by(Message.turn_index)
                .all()
            )

            unsummarized_count = len(all_messages) - session.summary_up_to_turn
            if unsummarized_count < settings.SUMMARIZE_AFTER_TURNS * 2:  # *2 because user+assistant
                return   # not enough new turns yet

            # Only summarize turns we haven't summarized yet
            new_msgs = all_messages[session.summary_up_to_turn:]
            if not new_msgs:
                return

            formatted = "\n".join(
                f"{m.role.upper()}: {m.content}" for m in new_msgs
            )
            prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
                prev_summary=session.summary or "(none)",
                messages=formatted,
            )

            if not self._llm or not self._llm.is_loaded:
                logger.warning("LLM not available for summarization; skipping.")
                return

            try:
                summary, _ = self._llm.generate(prompt)
                session.summary = summary.strip()
                session.summary_up_to_turn = len(all_messages)
                db.commit()
                logger.info(f"Session {session_id} summarized up to turn {len(all_messages)}.")
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to write summary for session {session_id}: {e}")
        finally:
            db.close()


summarizer_service = SummarizerService()
