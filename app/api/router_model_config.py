from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from app.api.deps import get_current_api_key, get_db_session
from app.db.models import AIModelConfig
from app.schemas.schemas import ModelConfigCreate, ModelConfigUpdate, ModelConfigOut, ModelListResponse
from app.services.llm_service import llm_service

router = APIRouter(prefix="/models", tags=["Model Configs"])


def _reload_shared_model():
    llm_service.reload_from_db()
    # SafeVoiceMedicalChatbot holds the shared llm_service object, so cache reset
    # is not strictly required. Keep this defensive hook for full rebuilds.
    try:
        from app.ai_core.serving.dependencies import reset_chatbot_cache, get_chatbot

        reset_chatbot_cache()
        get_chatbot()
    except Exception:
        pass



def _set_only_active(db: DBSession, model: AIModelConfig):
    db.query(AIModelConfig).update({AIModelConfig.is_active: False})
    model.is_active = True
    db.commit()
    db.refresh(model)


@router.get("", response_model=ModelListResponse)
def list_models(
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    rows = db.query(AIModelConfig).order_by(AIModelConfig.created_at.desc()).all()
    return ModelListResponse(models=rows, total=len(rows))


@router.post("", response_model=ModelConfigOut)
def create_model(
    req: ModelConfigCreate,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    if db.query(AIModelConfig).filter_by(name=req.name).first():
        raise HTTPException(status_code=409, detail="Model name already exists")
    row = AIModelConfig(**req.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    if req.is_active:
        _set_only_active(db, row)
    return row


@router.get("/{model_id}", response_model=ModelConfigOut)
def get_model(
    model_id: str,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    row = db.query(AIModelConfig).filter_by(id=model_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Model config not found")
    return row


@router.patch("/{model_id}", response_model=ModelConfigOut)
def update_model(
    model_id: str,
    req: ModelConfigUpdate,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    row = db.query(AIModelConfig).filter_by(id=model_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Model config not found")

    data = req.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(row, key, value)
    row.is_loaded = False
    db.commit()
    db.refresh(row)

    if data.get("is_active") is True:
        _set_only_active(db, row)
    return row


@router.delete("/{model_id}", status_code=204)
def delete_model(
    model_id: str,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    row = db.query(AIModelConfig).filter_by(id=model_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Model config not found")
    if row.is_active:
        raise HTTPException(status_code=400, detail="Cannot delete active model. Activate another model first.")
    db.delete(row)
    db.commit()


@router.post("/{model_id}/activate", response_model=ModelConfigOut)
def activate_model(
    model_id: str,
    reload_now: bool = False,
    api_key: str = Depends(get_current_api_key),
    db: DBSession = Depends(get_db_session),
):
    row = db.query(AIModelConfig).filter_by(id=model_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Model config not found")
    _set_only_active(db, row)
    if reload_now:
        _reload_shared_model()
        db.refresh(row)
    return row


@router.post("/reload")
def reload_active_model(
    api_key: str = Depends(get_current_api_key),
):
    _reload_shared_model()
    return {"loaded": llm_service.is_loaded, "message": "Active model reloaded from database and shared chatbot refreshed"}
