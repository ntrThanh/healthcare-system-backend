from fastapi import Depends, Header
from sqlalchemy.orm import Session as DBSession
from typing import Optional

from app.db.database import get_db
from app.core.security import validate_api_key, API_KEY_HEADER
from fastapi.security import APIKeyHeader


async def get_current_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    return validate_api_key(x_api_key)


async def get_db_session(db: DBSession = Depends(get_db)) -> DBSession:
    return db
