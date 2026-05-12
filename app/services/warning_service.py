from __future__ import annotations

from sqlalchemy.orm import Session as DBSession

from app.db.models import WarningKeyword


class WarningService:
    """Simple DB-configurable keyword warning detector."""

    def detect(self, db: DBSession, text: str) -> list[str]:
        if not text:
            return []
        lowered = text.lower()
        rows = db.query(WarningKeyword).filter_by(is_active=True).all()
        warnings: list[str] = []
        for row in rows:
            phrase = (row.phrase or "").strip().lower()
            if phrase and phrase in lowered and row.warning_message not in warnings:
                warnings.append(row.warning_message)
        return warnings


warning_service = WarningService()
