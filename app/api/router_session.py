from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session as DBSession

from app.api.deps import get_current_api_key, get_db_session
from app.db.models import Session as ChatSession, Message
from app.schemas.schemas import (
    SessionCreateRequest, SessionOut, SessionListResponse, SessionResetRequest
)
from app.services.session_manager import session_manager

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("", response_model=SessionOut)
async def create_session(
    req: SessionCreateRequest,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    """Create a new session for a user."""
    user = session_manager.get_or_create_user(req.user_id, db)
    session = session_manager.get_or_create_session(user, None, req.context, db)
    if req.title:
        session.title = req.title
        db.commit()
    return _session_out(session, db)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    user_id: str = Query(...),
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    """List all active sessions for a user."""
    user = session_manager.get_or_create_user(user_id, db)
    sessions = session_manager.list_sessions(user.id, db)
    return SessionListResponse(
        sessions=[_session_out(s, db) for s in sessions],
        total=len(sessions),
    )


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    """Get a specific session with its current summary."""
    user = session_manager.get_or_create_user(user_id, db)
    session = session_manager.get_session(session_id, user.id, db)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_out(session, db)


@router.post("/{session_id}/reset", response_model=SessionOut)
async def reset_session(
    session_id: str,
    req: SessionResetRequest,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    """Reset a session (delete messages and summary)."""
    try:
        session_manager.reset_session(session_id, req.keep_context, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    session = db.query(ChatSession).filter_by(id=session_id).first()
    return _session_out(session, db)


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    """Soft-delete a session."""
    session = db.query(ChatSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.is_active = False
    db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_out(session, db) -> SessionOut:
    count = db.query(Message).filter_by(session_id=session.id).count()
    return SessionOut(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        summary=session.summary,
        context=session.context,
        is_active=session.is_active,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=count,
    )
