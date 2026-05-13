from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.ai_core.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LLMTextResponse:
    content: str


class MockMedicalLLM:
    """Lightweight LLM substitute for CI/smoke tests."""

    def invoke(self, prompt: str) -> LLMTextResponse:
        question = prompt.split('Câu hỏi:')[-1] if 'Câu hỏi:' in prompt else prompt
        lower = question.lower()

        if 'metformin' in lower:
            content = (
                'Metformin là thuốc thuộc nhóm Biguanide, thường được dùng trong điều trị đái tháo đường type 2 theo chỉ định của bác sĩ. '
                'Thông tin này chỉ mang tính tham khảo; không tự ý dùng hoặc thay đổi liều thuốc nếu chưa được bác sĩ/dược sĩ tư vấn.'
            )
        elif 'đau ngực' in lower or 'không thở' in lower or 'cấp cứu' in lower:
            content = (
                'Đau ngực dữ dội hoặc khó thở là dấu hiệu có thể nguy hiểm. Hãy gọi 115 hoặc đến cơ sở cấp cứu gần nhất ngay. '
                'Tôi chỉ có thể cung cấp thông tin tham khảo sau khi tình trạng khẩn cấp đã được xử trí bởi nhân viên y tế.'
            )
        elif 'đái tháo đường' in lower:
            content = (
                'Đái tháo đường type 2 là rối loạn chuyển hóa đường mạn tính. '
                'Dữ liệu tham chiếu cho thấy các triệu chứng thường gặp gồm khát nước nhiều, tiểu nhiều và mệt mỏi. '
                'Điều trị thường bao gồm thay đổi lối sống và thuốc như Metformin hoặc Insulin theo chỉ định bác sĩ. '
                'Bạn nên gặp bác sĩ để được đánh giá và tư vấn phù hợp.'
            )
        elif 'tăng huyết áp' in lower:
            content = (
                'Tăng huyết áp là tình trạng áp lực máu trong động mạch tăng cao bất thường. '
                'Triệu chứng có thể gồm đau đầu, chóng mặt hoặc đau ngực, nhưng nhiều người không có triệu chứng rõ. '
                'Việc điều trị cần bác sĩ đánh giá nguy cơ tim mạch và theo dõi huyết áp.'
            )
        else:
            content = (
                'Dựa trên dữ liệu tham chiếu được cung cấp, tôi có thể giải thích thông tin y tế ở mức tổng quan. '
                'Tôi không thay thế bác sĩ và không thể chẩn đoán hoặc kê đơn. '
                'Bạn nên gặp bác sĩ hoặc cơ sở y tế để được tư vấn phù hợp.'
            )

        return LLMTextResponse(content=content)


class HuggingFaceMedicalLLM:
    def __init__(self, settings: Settings):
        import torch
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }

        torch_dtype = dtype_map.get(settings.model_torch_dtype, torch.bfloat16)

        quantization_config = None
        if settings.model_load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch_dtype,
            )

        logger.info(
            'Loading LLM: %s | 4bit=%s | dtype=%s',
            settings.model_name,
            settings.model_load_in_4bit,
            settings.model_torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if settings.model_load_in_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                quantization_config=quantization_config,
                device_map={"": 0},
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to("cuda")

        model.eval()

        text_generation = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.model_max_new_tokens,
            temperature=settings.model_temperature,
            do_sample=settings.model_temperature > 0,
            repetition_penalty=settings.model_repetition_penalty,
            return_full_text=False,
        )

        self._llm = HuggingFacePipeline(pipeline=text_generation)

    def invoke(self, prompt: str) -> Any:
        return self._llm.invoke(prompt)


def get_response_text(resp: Any) -> str:
    if hasattr(resp, 'content'):
        return str(resp.content)
    return str(resp)


def load_llm(settings: Settings):
    if settings.use_mock_llm:
        logger.warning('USE_MOCK_LLM=true: using mock model for local smoke test.')
        return MockMedicalLLM()

    return HuggingFaceMedicalLLM(settings)