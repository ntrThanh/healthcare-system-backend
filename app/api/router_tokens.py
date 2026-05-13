from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from app.api.deps import get_current_api_key, get_db_session
from app.db.models import Session as ChatSession
from app.schemas.schemas import TokenAddRequest, TokenUsageResponse
from app.services.token_service import token_service

router = APIRouter(prefix="/tokens", tags=["Session Tokens"])


@router.get("/sessions/{session_id}", response_model=TokenUsageResponse)
def get_session_tokens(
    session_id: str,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    if not db.query(ChatSession).filter_by(id=session_id).first():
        raise HTTPException(status_code=404, detail="Session not found")
    return TokenUsageResponse(**token_service.usage(db, session_id))


@router.post("/add", response_model=TokenUsageResponse)
def add_session_tokens(
    req: TokenAddRequest,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    if not db.query(ChatSession).filter_by(id=req.session_id).first():
        raise HTTPException(status_code=404, detail="Session not found")
    token_service.add_tokens(db, req.session_id, req.tokens, req.reason)
    return TokenUsageResponse(**token_service.usage(db, req.session_id))
