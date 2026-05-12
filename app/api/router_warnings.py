from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from app.api.deps import get_current_api_key, get_db_session
from app.db.models import WarningKeyword
from app.schemas.schemas import WarningKeywordCreate, WarningKeywordUpdate, WarningKeywordOut

router = APIRouter(prefix="/warnings", tags=["Warnings"])


@router.get("", response_model=list[WarningKeywordOut])
def list_warning_keywords(
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    return db.query(WarningKeyword).order_by(WarningKeyword.created_at.desc()).all()


@router.post("", response_model=WarningKeywordOut)
def create_warning_keyword(
    req: WarningKeywordCreate,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    if db.query(WarningKeyword).filter_by(phrase=req.phrase).first():
        raise HTTPException(status_code=409, detail="Phrase already exists")
    row = WarningKeyword(**req.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.patch("/{keyword_id}", response_model=WarningKeywordOut)
def update_warning_keyword(
    keyword_id: str,
    req: WarningKeywordUpdate,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    row = db.query(WarningKeyword).filter_by(id=keyword_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Warning keyword not found")
    for key, value in req.model_dump(exclude_unset=True).items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return row


@router.delete("/{keyword_id}", status_code=204)
def delete_warning_keyword(
    keyword_id: str,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    row = db.query(WarningKeyword).filter_by(id=keyword_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Warning keyword not found")
    db.delete(row)
    db.commit()
