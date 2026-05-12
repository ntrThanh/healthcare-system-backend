from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    SAFE_INFO = 'safe_info'
    BORDERLINE = 'borderline'
    DIAGNOSIS_REQ = 'diagnosis_request'
    PRESCRIPTION_REQ = 'prescription_req'
    EMERGENCY = 'emergency'
    OFF_TOPIC = 'off_topic'


@dataclass
class SafetyResult:
    allowed: bool
    intent: QueryIntent
    risk_level: str
    block_reason: Optional[str] = None
    disclaimer: str = ''
    suggested_redirect: Optional[str] = None


class MedicalSafetyGuard:
    """Rule-based safety guard for the medical chatbot."""

    DIAGNOSIS_PATTERNS = [
        r'tôi (bị|mắc|có) (bệnh gì|bệnh\s)',
        r'(chẩn đoán|xác định) (bệnh|tôi|giúp tôi)',
        r'tôi có bị .+ không',
        r'tôi có mắc .+ không',
        r'bệnh của tôi là gì',
        r'(em|mình|con|cháu|ba|mẹ) (bị|mắc) (bệnh|gì)',
        r'(diagnose|diagnosis)',
        r'tôi (đang|có thể) bị',

        r'.+ là bệnh gì',
        r'.+ là bị gì',
        r'.+ bị bệnh gì',
        r'.+ có phải .+ không',
        r'.+ có đúng là .+ không',
        r'.+ có phải dấu hiệu của .+ không',
        r'(tôi|em|mình|con|cháu) .+ là bị gì',
        r'(tôi|em|mình|con|cháu) .+ có phải bị .+ không',
        r'(tôi|em|mình|con|cháu) .+ liệu có bị .+ không',
        r'(tôi|em|mình|con|cháu) .+ đang mắc bệnh gì',
        r'(triệu chứng này|biểu hiện này) là bệnh gì',
        r'(kết luận|xác nhận) tôi bị .+',
    ]

    PRESCRIPTION_PATTERNS = [
        r'(kê|cho tôi|cần) (đơn|thuốc|toa)',
        r'uống thuốc gì (để|cho)',
        r'liều lượng .+ cho tôi',
        r'tôi nên dùng thuốc gì',
        r'prescri(be|ption)',
        r'thuốc .+ liều bao nhiêu cho (tôi|bé|em)',
        r'có nên uống .+ không',
    ]

    EMERGENCY_PATTERNS = [
        r'(cấp cứu|911|115|khẩn cấp)',
        r'đau ngực (dữ dội|dữ|dội|mạnh|cấp|không chịu được)',
        r'(khó thở|thở gấp|thở không nổi|không thở được)',
        r'(mất ý thức|ngất|bất tỉnh|lơ mơ)',
        r'(co giật|động kinh)',
        r'(xuất huyết|chảy máu) (nhiều|không cầm|liên tục)',
        r'(tê liệt|yếu liệt|méo miệng) đột ngột',
        r'đột ngột (không nhìn|không nói|không đi|yếu một bên)',
        r'(đau đầu dữ dội|đau đầu đột ngột)',
        r'(tai nạn|té ngã|chấn thương) .*(nặng|đầu|ngực|bụng)',
        r'(sốc phản vệ|dị ứng nặng|phù mặt|phù môi)',
        r'(nuốt phải|uống nhầm) .*(thuốc|hóa chất|chất độc)',
        r'(bỏng nặng|bỏng rộng)',
        r'(sốt cao co giật|trẻ co giật)',
    ]

    BORDERLINE_PATTERNS = [
        r'(triệu chứng|dấu hiệu) (của tôi|mình|em)',
        r'tôi (đang|bị) (đau|khó thở|sốt|ho|chóng mặt)',
        r'(các|những) triệu chứng .+ có thể là',
        r'(các|những) bệnh .+ gây ra .+ triệu chứng',

        r'(triệu chứng|dấu hiệu) (của|bệnh) .+',
        r'.+ có (triệu chứng|dấu hiệu) gì',
        r'.+ gây ra (triệu chứng|dấu hiệu) gì',
        r'.+ có nguy hiểm không',
        r'.+ có nghiêm trọng không',
        r'.+ có cần đi khám không',
        r'.+ khi nào cần đi khám',
        r'.+ khi nào cần cấp cứu',
        r'(tôi|em|mình|con|cháu) bị .+ thì làm sao',
        r'(tôi|em|mình|con|cháu) bị .+ nên làm gì',
        r'(tôi|em|mình|con|cháu) bị .+ có sao không',
        r'(đau|sốt|ho|khó thở|chóng mặt|buồn nôn|đau bụng|đau đầu|mệt mỏi|tiêu chảy|nôn|phát ban)',
        r'(xét nghiệm|kết quả xét nghiệm|chỉ số) .+ có ý nghĩa gì',
        r'(huyết áp|đường huyết|cholesterol|men gan|creatinin|hba1c) .+ cao',
        r'(huyết áp|đường huyết|cholesterol|men gan|creatinin|hba1c) .+ thấp',
    ]

    OFF_TOPIC_SAFE = [
        r'(thời tiết|bóng đá|phim|âm nhạc|chính trị)',
        r'(lập trình|code|python|javascript|java|sql|html|css)',
        r'(nấu ăn|công thức|recipe)(?!.*sức khỏe)',
        r'(du lịch|vé máy bay|khách sạn)',
        r'(chứng khoán|bitcoin|crypto|đầu tư)',
        r'(game|liên minh|free fire|minecraft)',
        r'(viết bài văn|làm thơ|dịch tiếng anh)',
    ]

    DISCLAIMER_TEMPLATES = {
        QueryIntent.SAFE_INFO: (
            '⚕️ *Thông tin mang tính tham khảo giáo dục. '
            'Vui lòng tham khảo ý kiến bác sĩ trước khi áp dụng.*'
        ),
        QueryIntent.BORDERLINE: (
            '⚠️ *Lưu ý quan trọng: Các triệu chứng bạn mô tả cần được bác sĩ thăm khám '
            'trực tiếp để có đánh giá chính xác. Thông tin dưới đây chỉ mang tính tham khảo.*'
        ),
        QueryIntent.EMERGENCY: (
            '🚨 *KHẨN CẤP: Nếu bạn hoặc người thân đang trong tình trạng nguy hiểm, '
            'hãy gọi ngay 115 (Cấp cứu) hoặc đến phòng cấp cứu gần nhất!*'
        ),
    }

    BLOCK_RESPONSES = {
        QueryIntent.DIAGNOSIS_REQ: (
            'Tôi không thể đưa ra chẩn đoán bệnh. Đây là nhiệm vụ của bác sĩ được đào tạo '
            'chuyên môn với đầy đủ thông tin lâm sàng về bạn.\n\n'
            'Tôi có thể giúp bạn:\n'
            '• Cung cấp thông tin về các bệnh, triệu chứng\n'
            '• Giải thích kết quả xét nghiệm (tham khảo)\n'
            '• Hướng dẫn chuẩn bị gặp bác sĩ\n\n'
            'Bạn muốn tìm hiểu thông tin gì cụ thể?'
        ),
        QueryIntent.PRESCRIPTION_REQ: (
            'Tôi không thể kê đơn thuốc hoặc tư vấn liều dùng cụ thể cho cá nhân. '
            'Việc dùng thuốc sai liều hoặc không đúng chỉ định có thể gây nguy hiểm.\n\n'
            'Vui lòng:\n'
            '• Gặp bác sĩ để được kê đơn phù hợp\n'
            '• Tham khảo dược sĩ về thông tin thuốc\n\n'
            'Tôi có thể cung cấp thông tin chung về nhóm thuốc nếu bạn muốn.'
        ),
        QueryIntent.OFF_TOPIC: (
            'Tôi được thiết kế chuyên biệt cho thông tin y tế sức khỏe. '
            'Câu hỏi này nằm ngoài phạm vi hỗ trợ của tôi. '
            'Hãy hỏi về bệnh, triệu chứng, thuốc, hoặc xét nghiệm y tế nhé!'
        ),
    }

    def __init__(self) -> None:
        flags = re.IGNORECASE | re.UNICODE
        self._diag_re = [re.compile(p, flags) for p in self.DIAGNOSIS_PATTERNS]
        self._presc_re = [re.compile(p, flags) for p in self.PRESCRIPTION_PATTERNS]
        self._emerg_re = [re.compile(p, flags) for p in self.EMERGENCY_PATTERNS]
        self._border_re = [re.compile(p, flags) for p in self.BORDERLINE_PATTERNS]
        self._offtop_re = [re.compile(p, flags) for p in self.OFF_TOPIC_SAFE]

    @staticmethod
    def _match_any(text: str, patterns: list[re.Pattern]) -> bool:
        return any(p.search(text) for p in patterns)

    def check(self, query: str) -> SafetyResult:
        q = query.strip()

        if self._match_any(q, self._emerg_re):
            return SafetyResult(
                allowed=True,
                intent=QueryIntent.EMERGENCY,
                risk_level='critical',
                disclaimer=self.DISCLAIMER_TEMPLATES[QueryIntent.EMERGENCY],
                suggested_redirect='115 hoặc phòng cấp cứu gần nhất',
            )
        if self._match_any(q, self._diag_re):
            return SafetyResult(False, QueryIntent.DIAGNOSIS_REQ, 'high', self.BLOCK_RESPONSES[QueryIntent.DIAGNOSIS_REQ])
        if self._match_any(q, self._presc_re):
            return SafetyResult(False, QueryIntent.PRESCRIPTION_REQ, 'high', self.BLOCK_RESPONSES[QueryIntent.PRESCRIPTION_REQ])
        if self._match_any(q, self._offtop_re):
            return SafetyResult(False, QueryIntent.OFF_TOPIC, 'low', self.BLOCK_RESPONSES[QueryIntent.OFF_TOPIC])
        if self._match_any(q, self._border_re):
            return SafetyResult(True, QueryIntent.BORDERLINE, 'medium', disclaimer=self.DISCLAIMER_TEMPLATES[QueryIntent.BORDERLINE])
        return SafetyResult(True, QueryIntent.SAFE_INFO, 'low', disclaimer=self.DISCLAIMER_TEMPLATES[QueryIntent.SAFE_INFO])


class ResponseValidator:
    """Detect unsafe/hallucinated answer patterns before returning output."""

    HALLUCINATION_SIGNALS = [
        r'(chẩn đoán|kết luận) (bạn|anh|chị|em) (bị|mắc|có)',
        r'(bạn|anh|chị) (nên|cần) (uống|dùng) .{3,30} \d+\s*(mg|ml|viên)',
        r'100%|chắc chắn|khẳng định (bạn|anh|chị)',
        r'(bệnh|chứng) của bạn là',
        r'theo (kinh nghiệm|chẩn đoán) của tôi',
        r'tôi (kết luận|chẩn đoán|xác định)',
    ]
    REQUIRED_DISCLAIMER_SIGNAL = ['bác sĩ', 'chuyên gia', 'tham khảo', 'thăm khám', 'y tế']

    def __init__(self) -> None:
        flags = re.IGNORECASE | re.UNICODE
        self._hall_re = [re.compile(p, flags) for p in self.HALLUCINATION_SIGNALS]

    def validate(self, response: str, intent: QueryIntent) -> tuple[str, list[str]]:
        warnings: list[str] = []
        cleaned = response
        for pattern in self._hall_re:
            if pattern.search(cleaned):
                warnings.append(f'Unsafe pattern removed: {pattern.pattern[:60]}')
                cleaned = self._remove_dangerous_sentence(cleaned, pattern)

        has_disclaimer = any(kw in cleaned.lower() for kw in self.REQUIRED_DISCLAIMER_SIGNAL)
        if not has_disclaimer and intent != QueryIntent.SAFE_INFO:
            cleaned += '\n\n*Lưu ý: Thông tin trên chỉ mang tính tham khảo. Hãy tham khảo bác sĩ để được tư vấn phù hợp.*'
            warnings.append('Auto-added medical disclaimer')
        return cleaned, warnings

    @staticmethod
    def _remove_dangerous_sentence(text: str, pattern: re.Pattern) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        safe = [s for s in sentences if not pattern.search(s)]
        return ' '.join(safe) if safe else text


class ContextConfidenceChecker:
    MIN_KG_CONTEXT_LEN = 50
    MIN_VEC_CONTEXT_LEN = 100

    def check(self, kg_context: str, vec_context: str, query: str) -> dict:
        has_kg = len(kg_context.strip()) > self.MIN_KG_CONTEXT_LEN
        has_vec = len(vec_context.strip()) > self.MIN_VEC_CONTEXT_LEN
        no_kg_signal = 'không tìm thấy' in kg_context.lower()
        confidence = 'high'
        warnings: list[str] = []
        if not has_kg and not has_vec:
            confidence = 'very_low'
            warnings.append('Không có dữ liệu từ KG hoặc vector store')
        elif no_kg_signal or not has_kg:
            confidence = 'medium'
            warnings.append('Thông tin Knowledge Graph hạn chế')
        elif not has_vec:
            confidence = 'medium'
            warnings.append('Vector store không tìm thấy tài liệu liên quan')
        return {
            'confidence': confidence,
            'has_kg': has_kg,
            'has_vec': has_vec,
            'warnings': warnings,
        }
