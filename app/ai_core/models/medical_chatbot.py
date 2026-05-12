from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.ai_core.core.config import Settings
from app.ai_core.models.base_model import BaseMedicalModel
from app.ai_core.models.llm_loader import get_response_text
from app.ai_core.models.memory import ConversationSummaryBuffer
from app.ai_core.models.safety import ContextConfidenceChecker, MedicalSafetyGuard, ResponseValidator
from app.ai_core.rag.hyde import expand_query_hyde
from app.ai_core.rag.relevance import evaluate_grounding
from app.ai_core.rag.retriever import format_docs
from app.ai_core.voice.stt import NullSTT
from app.ai_core.voice.tts import NullTTS

logger = logging.getLogger(__name__)

SAFE_SYSTEM_PROMPT_V4 = """<|im_start|>system
Bạn là trợ lý y tế AI hỗ trợ thông tin sức khỏe bằng tiếng Việt.

VAI TRÒ: Cung cấp thông tin y tế tham khảo dựa trên dữ liệu KG và tài liệu.
KHÔNG PHẢI: Bác sĩ, không chẩn đoán, không kê đơn thuốc.

QUY TẮC BẮT BUỘC:
1. Chỉ cung cấp thông tin tham khảo từ KG và tài liệu được cung cấp.
2. KHÔNG đưa ra chẩn đoán bệnh cụ thể cho người dùng.
3. KHÔNG kê đơn hoặc tư vấn liều thuốc cụ thể.
4. LUÔN kết thúc bằng khuyến nghị gặp bác sĩ.
5. Nếu không có dữ liệu, nói rõ "Tôi không có thông tin về vấn đề này".
6. Trả lời ngắn gọn, rõ ràng, phù hợp đọc thành giọng nói.
7. Trả lời hoàn toàn bằng tiếng Việt cho tôi.

ĐỘ TIN CẬY NGỮ CẢNH: {confidence_level}
{confidence_warning}
<|im_end|>
<|im_start|>user
[NGỮ CẢNH BỔ SUNG TỪ API CHAT]
{extra_context}

[KNOWLEDGE GRAPH — MULTI-HOP]
{kg_ctx}

[TÀI LIỆU Y TẾ — HYBRID RERANKED]
{vec_ctx}

[LỊCH SỬ HỘI THOẠI — TÓM TẮT]
{history}

Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
"""


class SafeVoiceMedicalChatbot(BaseMedicalModel):
    """Productionized version of the notebook v4 pipeline."""

    def __init__(
        self,
        settings: Settings,
        llm: Any,
        retriever: Any,
        graph_engine: Any,
        stt_engine: Any | None = None,
        tts_engine: Any | None = None,
        safety_guard: MedicalSafetyGuard | None = None,
        response_validator: ResponseValidator | None = None,
        context_checker: ContextConfidenceChecker | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.gqe = graph_engine
        self.stt = stt_engine or NullSTT()
        self.tts = tts_engine or NullTTS()
        self.guard = safety_guard or MedicalSafetyGuard()
        self.validator = response_validator or ResponseValidator()
        self.ctx_checker = context_checker or ContextConfidenceChecker()
        self._memories: dict[str, ConversationSummaryBuffer] = {}
        self._audit_log: list[dict[str, Any]] = []

    def predict(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.chat(message, **kwargs)

    def _memory(self, session_id: str) -> ConversationSummaryBuffer:
        if session_id not in self._memories:
            self._memories[session_id] = ConversationSummaryBuffer()
        return self._memories[session_id]

    def reset(self, session_id: str = 'default') -> None:
        if session_id in self._memories:
            self._memories[session_id].clear()
        self._audit_log = [e for e in self._audit_log if e.get('session_id') != session_id]

    def _extract_entities(self, query: str) -> dict[str, list[str]]:
        ql = query.lower()
        disease_kw = self.gqe.list_names('Disease')
        drug_kw = self.gqe.list_names('Drug')
        symptom_kw = self.gqe.list_names('Symptom')
        return {
            'diseases': [n for n in disease_kw if any(w in ql for w in n.lower().split()[:2])],
            'drugs': [n for n in drug_kw if n.lower() in ql],
            'symptoms': [n for n in symptom_kw if any(w in ql for w in n.lower().split()[:2])],
        }

    def _get_kg_context(self, query: str, entities: dict[str, list[str]]) -> str:
        parts: list[str] = []
        for name in entities.get('diseases', [])[:2]:
            ctx = self.gqe.get_multihop_context(name)
            if ctx:
                parts.append(ctx)
        for name in entities.get('drugs', [])[:1]:
            info = self.gqe.get_drug_info(name)
            if info:
                parts.append(self.gqe.format(info))
        if entities.get('symptoms') and not entities.get('diseases'):
            for name in entities['symptoms'][:2]:
                diseases = self.gqe.get_symptom_diseases(name)
                if diseases:
                    parts.append(f"'{name}' liên quan đến: {', '.join(d['disease'] for d in diseases)}")
        return '\n\n'.join(parts) or 'Không tìm thấy trong Knowledge Graph.'

    def _get_vec_context(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        expanded_query = expand_query_hyde(query, self.llm, use_hyde=self.settings.use_hyde)
        docs = self.retriever.invoke(expanded_query)
        return format_docs(docs)

    def _speak(self, text: str, output_path: Path | None = None) -> str | None:
        if not self.settings.enable_tts:
            return None
        output_path = output_path or (self.settings.audio_output_dir / f'{uuid4().hex}.wav')
        try:
            return self.tts.synthesize(text[:600], output_path=output_path)
        except Exception as exc:
            logger.warning('TTS failed: %s', exc)
            return None

    def voice_to_text(self, audio_input: Any) -> str:
        import numpy as np

        if isinstance(audio_input, (str, Path)):
            return self.stt.transcribe_file(audio_input)
        if isinstance(audio_input, tuple):
            sample_rate, samples = audio_input
            if hasattr(samples, 'dtype') and samples.dtype != np.float32:
                samples = samples.astype(np.float32) / 32768.0
            return self.stt.transcribe_array(samples, int(sample_rate))
        raise ValueError(f'Unsupported audio input type: {type(audio_input)}')

    def voice_chat(self, audio_input: Any, session_id: str = 'default', enable_tts: bool = True) -> tuple[str, dict[str, Any]]:
        user_text = self.voice_to_text(audio_input)
        if not user_text.strip():
            return '', {
                'answer': 'Không nhận được giọng nói. Vui lòng thử lại.',
                'session_id': session_id,
                'intent': 'unknown',
                'risk_level': 'low',
                'confidence': 'very_low',
                'blocked': False,
                'warnings': ['empty_transcription'],
                'audio_path': None,
                'sources': [],
            }
        result = self.chat(user_text, session_id=session_id, enable_tts=enable_tts)
        return user_text, result

    def chat(
        self,
        message: str,
        session_id: str = 'default',
        enable_tts: bool = False,
        history_override: str | None = None,
        extra_context: str | None = None,
    ) -> dict[str, Any]:
        safety = self.guard.check(message)
        if not safety.allowed:
            audio_path = self._speak(safety.block_reason or '', None) if enable_tts else None
            result = {
                'answer': safety.block_reason or '',
                'session_id': session_id,
                'intent': safety.intent.value,
                'risk_level': safety.risk_level,
                'confidence': 'blocked',
                'blocked': True,
                'warnings': [],
                'audio_path': audio_path,
                'sources': [],
            }
            self._audit(message, safety, result)
            return result

        emergency_prefix = ''
        if safety.intent.value == 'emergency':
            emergency_prefix = f'🚨 {safety.disclaimer}\n\nĐang cung cấp thông tin tham khảo thêm:\n\n'

        entities = self._extract_entities(message)
        kg_ctx = self._get_kg_context(message, entities)
        vec_ctx, sources = self._get_vec_context(message)
        ctx_eval = self.ctx_checker.check(kg_ctx, vec_ctx, message)
        grounding = evaluate_grounding(
            query=message,
            kg_ctx=kg_ctx,
            sources=sources,
            entities=entities,
            mode=self.settings.rag_grounding_mode,
        )
        if grounding.warnings:
            ctx_eval['warnings'].extend(grounding.warnings)
        if grounding.confidence_override:
            ctx_eval['confidence'] = grounding.confidence_override
        if grounding.should_fallback:
            fallback_answer = (
                'Tôi chưa có dữ liệu tham chiếu đáng tin cậy về nội dung này trong '
                'Knowledge Graph/vectorstore hiện tại, nên không thể trả lời dựa trên nguồn của hệ thống. '
                'Bạn nên tham khảo bác sĩ hoặc nguồn y tế chính thống để được tư vấn phù hợp.'
            )
            disclaimer_block = f'\n\n---\n{safety.disclaimer}' if safety.disclaimer else ''
            final_answer = f'{fallback_answer}{disclaimer_block}\n*Độ tin cậy dữ liệu: very_low*'
            result = {
                'answer': final_answer,
                'session_id': session_id,
                'intent': safety.intent.value,
                'risk_level': safety.risk_level,
                'confidence': 'very_low',
                'blocked': False,
                'warnings': ctx_eval['warnings'],
                'audio_path': None,
                'sources': [],
            }
            self._audit(message, safety, result)
            return result

        confidence = ctx_eval['confidence']
        confidence_warning = (
            '⚠️ Dữ liệu tham chiếu hạn chế — hãy thận trọng và xác minh với chuyên gia y tế.'
            if confidence in ('medium', 'very_low') else ''
        )

        memory = self._memory(session_id)
        history_text = history_override if history_override is not None else memory.render()
        prompt = SAFE_SYSTEM_PROMPT_V4.format(
            extra_context=(extra_context or 'Không có.').strip() or 'Không có.',
            kg_ctx=kg_ctx,
            vec_ctx=vec_ctx,
            history=history_text or 'Chưa có lịch sử.',
            question=message,
            confidence_level=confidence.upper(),
            confidence_warning=confidence_warning,
        )
        raw = self.llm.invoke(prompt)
        response = get_response_text(raw).strip()
        if '<|im_end|>' in response:
            response = response.split('<|im_end|>')[0].strip()

        cleaned, validator_warnings = self.validator.validate(response, safety.intent)
        disclaimer_block = f'\n\n---\n{safety.disclaimer}' if safety.disclaimer else ''
        confidence_block = f'\n*Độ tin cậy dữ liệu: {confidence}*' if confidence != 'high' else ''
        final_answer = f'{emergency_prefix}{cleaned}{disclaimer_block}{confidence_block}'

        if history_override is None:
            memory.add(message, cleaned, llm=self.llm)
        audio_path = self._speak(cleaned, None) if enable_tts else None
        result = {
            'answer': final_answer,
            'session_id': session_id,
            'intent': safety.intent.value,
            'risk_level': safety.risk_level,
            'confidence': confidence,
            'blocked': False,
            'warnings': ctx_eval['warnings'] + validator_warnings,
            'audio_path': audio_path,
            'sources': sources,
        }
        self._audit(message, safety, result)
        return result

    def _audit(self, query: str, safety: Any, result: dict[str, Any]) -> None:
        entry = {
            'ts': dt.datetime.now(dt.UTC).isoformat(),
            'session_id': result.get('session_id'),
            'query': query[:300],
            'intent': safety.intent.value,
            'risk_level': safety.risk_level,
            'allowed': safety.allowed,
            'confidence': result.get('confidence'),
            'warnings': result.get('warnings', []),
            'blocked': result.get('blocked', False),
            'response_len': len(result.get('answer', '')),
        }
        self._audit_log.append(entry)
        if safety.risk_level in ('high', 'critical'):
            logger.warning('[AUDIT HIGH-RISK] %s', entry)

    def audit_summary(self) -> dict[str, Any]:
        from collections import Counter

        return {
            'total_requests': len(self._audit_log),
            'blocked': sum(1 for e in self._audit_log if e.get('blocked')),
            'intents': dict(Counter(e['intent'] for e in self._audit_log)),
            'risk_levels': dict(Counter(e['risk_level'] for e in self._audit_log)),
        }
