from __future__ import annotations

import json
from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from app.db.models import SessionTokenLedger


class TokenService:
    """Estimate, charge, and query token usage per chat session."""

    def estimate_tokens(self, text: str, tokenizer=None) -> int:
        if not text:
            return 0
        if tokenizer is not None:
            try:
                return int(len(tokenizer.encode(text)))
            except Exception:
                pass
        # Vietnamese/English rough fallback: 1 token ≈ 4 chars.
        return max(1, len(text) // 4)

    def add_tokens(self, db: DBSession, session_id: str, tokens: int, reason: str = "manual_add") -> SessionTokenLedger:
        entry = SessionTokenLedger(
            session_id=session_id,
            delta_tokens=abs(int(tokens)),
            reason=reason or "manual_add",
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry

    def charge_tokens(
        self,
        db: DBSession,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        reason: str = "chat_usage",
    ) -> SessionTokenLedger:
        total = int(prompt_tokens) + int(completion_tokens)
        entry = SessionTokenLedger(
            session_id=session_id,
            delta_tokens=-total,
            reason=reason,
            meta=json.dumps({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
            }, ensure_ascii=False),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry

    def usage(self, db: DBSession, session_id: str) -> dict:
        added = db.query(func.coalesce(func.sum(SessionTokenLedger.delta_tokens), 0)).filter(
            SessionTokenLedger.session_id == session_id,
            SessionTokenLedger.delta_tokens > 0,
        ).scalar() or 0
        used_negative = db.query(func.coalesce(func.sum(SessionTokenLedger.delta_tokens), 0)).filter(
            SessionTokenLedger.session_id == session_id,
            SessionTokenLedger.delta_tokens < 0,
        ).scalar() or 0
        used = abs(int(used_negative))
        return {
            "session_id": session_id,
            "added_tokens": int(added),
            "used_tokens": used,
            "remaining_tokens": int(added) - used,
        }


token_service = TokenService()
