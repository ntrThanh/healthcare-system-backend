from __future__ import annotations

import logging
from typing import Any

from app.ai_core.models.llm_loader import get_response_text

logger = logging.getLogger(__name__)

HYDE_PROMPT = (
    'Bạn là bác sĩ chuyên khoa. Viết một đoạn văn ngắn (2-3 câu) '
    'mô tả thông tin y tế chuyên môn liên quan đến câu hỏi sau. '
    'Trả lời như trích đoạn từ sách giáo khoa y khoa, không giải thích thêm.\n\n'
    'Câu hỏi: {query}\nĐoạn văn y tế:'
)


def expand_query_hyde(query: str, llm: Any, use_hyde: bool = True) -> str:
    if not use_hyde:
        return query
    try:
        response = llm.invoke(HYDE_PROMPT.format(query=query))
        hypothetical = get_response_text(response).strip()
        if len(hypothetical) < 20:
            return query
        logger.debug('HyDE expanded query: %s', hypothetical[:160])
        return hypothetical
    except Exception as exc:
        logger.warning('HyDE failed; falling back to raw query: %s', exc)
        return query
