from __future__ import annotations
import logging
from typing import Optional, List, Tuple

from sqlalchemy.orm import Session as DBSession
from app.db.models import User, Session as ChatSession, Message
from app.core.config import settings

import json

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions, user lookup/creation,
    and building the final prompt context for the LLM.
    """

    # ── User ─────────────────────────────────────────────────────────────────

    def get_or_create_user(self, external_id: str, db: DBSession) -> User:
        user = db.query(User).filter_by(external_id=external_id).first()
        if not user:
            user = User(external_id=external_id)
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new user: {user.id} ({external_id})")
        return user

    # ── Session ───────────────────────────────────────────────────────────────

    def get_or_create_session(
        self,
        user: User,
        session_id: Optional[str],
        context: Optional[str],
        db: DBSession,
    ) -> ChatSession:
        def serialize_context(ctx):
            if ctx is None:
                return None
            if isinstance(ctx, str):
                return ctx
            return json.dumps(ctx, ensure_ascii=False)

        context_text = serialize_context(context)

        if session_id:
            session = db.query(ChatSession).filter_by(id=session_id, user_id=user.id).first()

            if not session:
                session = ChatSession(
                    id=session_id,
                    user_id=user.id,
                    context=context_text,
                )
                db.add(session)
                db.commit()
                db.refresh(session)

            if context_text:
                session.context = context_text
                db.commit()

            return session

        session = ChatSession(
            user_id=user.id,
            context=context_text,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        logger.info(f"Created new session: {session.id}")
        return session

    def get_session(self, session_id: str, user_id: str, db: DBSession) -> Optional[ChatSession]:
        return db.query(ChatSession).filter_by(id=session_id, user_id=user_id).first()

    def list_sessions(self, user_id: str, db: DBSession) -> List[ChatSession]:
        return (
            db.query(ChatSession)
            .filter_by(user_id=user_id, is_active=True)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )

    def reset_session(self, session_id: str, keep_context: bool, db: DBSession):
        session = db.query(ChatSession).filter_by(id=session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")
        db.query(Message).filter_by(session_id=session_id).delete()
        session.summary = None
        session.summary_up_to_turn = 0
        if not keep_context:
            session.context = None
        db.commit()
        logger.info(f"Reset session {session_id}")

    # ── Prompt building ───────────────────────────────────────────────────────

    def build_prompt(
        self,
        session: ChatSession,
        user_message: str,
        rag_context: str,
        db: DBSession,
    ) -> str:
        """
        Build the full LLM prompt:
          [System context]
          [Rolling summary if exists]
          [RAG context if any]
          [Recent conversation turns]
          [Current user message]
        """
        parts: List[str] = []

        # 1. System / persona context
        if session.context:
            parts.append(f"### System:\n{session.context.strip()}")

        # 2. Rolling summary (replaces older turns)
        if session.summary:
            parts.append(f"### Conversation summary so far:\n{session.summary.strip()}")

        # 3. RAG retrieved context
        if rag_context:
            parts.append(rag_context)

        # 4. Recent message history (most recent N turns, after summarized turns)
        recent_msgs = (
            db.query(Message)
            .filter_by(session_id=session.id)
            .filter(Message.turn_index >= session.summary_up_to_turn)
            .order_by(Message.turn_index)
            .limit(settings.MAX_HISTORY_TURNS * 2)
            .all()
        )
        if recent_msgs:
            history_lines = []
            for m in recent_msgs:
                label = "User" if m.role == "user" else "Assistant"
                history_lines.append(f"{label}: {m.content}")
            parts.append("### Conversation:\n" + "\n".join(history_lines))

        # 5. Current user turn
        parts.append(f"User: {user_message}\nAssistant:")

        return "\n\n".join(parts)

    def save_turn(
        self,
        session: ChatSession,
        user_content: str,
        assistant_content: str,
        rag_source_ids: List[str],
        latency_ms: float,
        db: DBSession,
    ) -> Tuple[Message, Message]:
        """Persist both the user message and the assistant reply."""
        import json

        all_msgs = db.query(Message).filter_by(session_id=session.id).all()
        next_index = len(all_msgs)

        user_msg = Message(
            session_id=session.id,
            role="user",
            content=user_content,
            turn_index=next_index,
        )
        db.add(user_msg)

        asst_msg = Message(
            session_id=session.id,
            role="assistant",
            content=assistant_content,
            turn_index=next_index + 1,
            rag_sources=json.dumps(rag_source_ids) if rag_source_ids else None,
            latency_ms=latency_ms,
        )
        db.add(asst_msg)

        # Auto-title from first user message
        if not session.title and user_content:
            session.title = user_content[:80]

        db.commit()
        db.refresh(user_msg)
        db.refresh(asst_msg)
        return user_msg, asst_msg


session_manager = SessionManager()
