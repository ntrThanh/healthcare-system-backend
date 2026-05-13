from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal


GroundingMode = Literal['off', 'warn', 'strict']


_STOPWORDS_VI = {
    'la', 'là', 'gi', 'gì', 'co', 'có', 'cua', 'của', 've', 'về',
    'benh', 'bệnh', 'trieu', 'triệu', 'chung', 'chứng', 'dau', 'dấu',
    'hieu', 'hiệu', 'khi', 'nao', 'nào', 'nhu', 'như', 'the', 'thế',
    'toi', 'tôi', 'ban', 'bạn', 'can', 'cần', 'nen', 'nên',
    'khong', 'không', 'dung', 'dùng', 'thuoc', 'thuốc',
    'dieu', 'điều', 'tri', 'trị', 'thong', 'tin', 'tham', 'khao',
    'lao',  # avoid over-weighting very short/generic disease-word variants
}


@dataclass(frozen=True)
class GroundingDecision:
    mode: str
    grounded: bool
    source_relevant: bool
    should_fallback: bool
    confidence_override: str | None
    warnings: list[str]


def normalize_vi(text: str) -> str:
    text = unicodedata.normalize('NFD', text or '')
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'D').lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def important_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_vi(text).split()
        if len(token) >= 2 and token not in _STOPWORDS_VI
    }


def sources_are_relevant(query: str, sources: list[dict[str, Any]]) -> bool:
    """Return True when at least one retrieved source appears to match the query.

    This is intentionally conservative for medical production mode. If the graph/vectorstore
    does not contain a source whose entity name or specific content overlaps with the query,
    the system should not let the LLM produce an unsupported answer.
    """
    q_tokens = important_tokens(query)
    if not q_tokens:
        return False

    for source in sources or []:
        name = source.get('name') or ''
        preview = source.get('preview') or ''

        name_tokens = important_tokens(name)
        source_tokens = important_tokens(f'{name} {preview}')

        # Strong signal: query explicitly mentions a retrieved entity name.
        if q_tokens & name_tokens:
            return True

        # Secondary signal: query mentions a specific token present in the retrieved text.
        # This catches names like insulin/metformin/amlodipine/HIV when present in preview.
        rare_hits = {
            token
            for token in q_tokens & source_tokens
            if len(token) >= 4 and token not in {'benh', 'trieu', 'chung', 'dieu', 'tri'}
        }
        if rare_hits:
            return True

    return False


def has_any_entities(entities: dict[str, list[str]]) -> bool:
    return any(bool(values) for values in entities.values())


def kg_context_found(kg_ctx: str) -> bool:
    normalized = normalize_vi(kg_ctx)
    return 'khong tim thay' not in normalized and len(normalized) > 20


def evaluate_grounding(
    query: str,
    kg_ctx: str,
    sources: list[dict[str, Any]],
    entities: dict[str, list[str]],
    mode: GroundingMode = 'strict',
) -> GroundingDecision:
    """Decide whether a generated answer is allowed to rely on retrieved context.

    Modes:
        strict: return a fallback instead of invoking/generating with the LLM when no relevant
            source is found. Best for production medical answers.
        warn: allow LLM generation, but force very_low confidence and attach a warning.
        off: preserve the old behavior.
    """
    mode = mode or 'strict'
    if mode not in {'off', 'warn', 'strict'}:
        mode = 'strict'

    if mode == 'off':
        return GroundingDecision(
            mode=mode,
            grounded=True,
            source_relevant=True,
            should_fallback=False,
            confidence_override=None,
            warnings=[],
        )

    source_relevant = sources_are_relevant(query, sources)
    entity_found = has_any_entities(entities)
    kg_found = kg_context_found(kg_ctx)
    grounded = source_relevant or (entity_found and kg_found)

    if grounded:
        return GroundingDecision(
            mode=mode,
            grounded=True,
            source_relevant=source_relevant,
            should_fallback=False,
            confidence_override=None,
            warnings=[],
        )

    warning = (
        'Không tìm thấy thực thể hoặc tài liệu liên quan đủ mạnh trong '
        'Knowledge Graph/vectorstore'
    )
    return GroundingDecision(
        mode=mode,
        grounded=False,
        source_relevant=source_relevant,
        should_fallback=(mode == 'strict'),
        confidence_override='very_low',
        warnings=[warning],
    )
