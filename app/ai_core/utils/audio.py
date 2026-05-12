from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4


def save_upload_to_tmp(upload_file, suffix: str = '.wav') -> Path:
    tmp_dir = Path('/tmp/medical_voice_rag_uploads')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f'{uuid4().hex}{suffix}'
    with path.open('wb') as f:
        shutil.copyfileobj(upload_file, f)
    return path
