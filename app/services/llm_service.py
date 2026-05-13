from __future__ import annotations

import logging
import time
from typing import Callable, Generator, Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def get_setting(*names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(settings, name):
            value = getattr(settings, name)
            if value is not None:
                return value
    return default


class LLMService:
    """HuggingFace causal LM service. Runtime parameters come from DB."""

    def __init__(self):
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._active_config_id: str | None = None
        self._active_model_name: str | None = None
        self._max_new_tokens = 512
        self._temperature = 0.3
        self._repetition_penalty = 1.1
        self._do_sample = True

    def _get_active_config(self):
        """Read active model config from DB. DB should already be initialized."""
        try:
            from app.db.database import SessionLocal
            from app.db.models import AIModelConfig

            db = SessionLocal()
            try:
                return db.query(AIModelConfig).filter_by(is_active=True).order_by(AIModelConfig.updated_at.desc()).first()
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Cannot read active model config from DB, fallback to .env: {e}")
            return None

    def load(self, force_reload: bool = False):
        try:
            use_mock_llm = get_setting("use_mock_llm", "USE_MOCK_LLM", default=False)
            active = self._get_active_config()

            if active:
                self._active_config_id = active.id
                model_name = active.model_name
                self._max_new_tokens = int(active.max_new_tokens or 512)
                self._temperature = float(active.temperature or 0.0)
                self._repetition_penalty = float(active.repetition_penalty or 1.0)
                self._do_sample = bool(active.do_sample)
                torch_dtype_name = active.torch_dtype or "float16"
                load_in_4bit = bool(active.load_in_4bit)
                trust_remote_code = bool(active.trust_remote_code)
            else:
                model_name = get_setting("model_name", "MODEL_NAME", "LLM_MODEL_PATH", default="Qwen/Qwen2.5-1.5B-Instruct")
                self._max_new_tokens = int(get_setting("model_max_new_tokens", "MODEL_MAX_NEW_TOKENS", "LLM_MAX_NEW_TOKENS", default=512))
                self._temperature = float(get_setting("model_temperature", "MODEL_TEMPERATURE", "LLM_TEMPERATURE", default=0.3))
                self._repetition_penalty = float(get_setting("model_repetition_penalty", "MODEL_REPETITION_PENALTY", default=1.1))
                self._do_sample = self._temperature > 0
                torch_dtype_name = get_setting("model_torch_dtype", "MODEL_TORCH_DTYPE", default="float16")
                load_in_4bit = bool(get_setting("MODEL_LOAD_IN_4BIT", "model_load_in_4bit", default=False))
                trust_remote_code = True

            if self._pipeline and not force_reload and self._active_model_name == model_name:
                logger.info("LLM already loaded with active DB config; skip reload.")
                return

            if use_mock_llm:
                logger.warning("USE_MOCK_LLM=true: using mock LLMService.")
                self._pipeline = "mock"
                self._active_model_name = model_name
                self._mark_loaded(True)
                return

            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            if torch_dtype_name == "bfloat16":
                torch_dtype = torch.bfloat16
            elif torch_dtype_name == "float32":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16

            logger.info(f"Loading LLM from DB config: {model_name} ...")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )

            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": "auto",
                "trust_remote_code": trust_remote_code,
            }

            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                compute_dtype = torch.bfloat16 if torch_dtype_name == "bfloat16" else torch.float16

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._do_sample,
                repetition_penalty=self._repetition_penalty,
                return_full_text=False,
            )
            self._active_model_name = model_name
            self._mark_loaded(True)
            logger.info("LLM loaded successfully from DB config.")
        except Exception:
            self._mark_loaded(False)
            logger.exception("Failed to load LLM")
            raise

    def _mark_loaded(self, loaded: bool):
        if not self._active_config_id:
            return
        try:
            from app.db.database import SessionLocal
            from app.db.models import AIModelConfig

            db = SessionLocal()
            try:
                row = db.query(AIModelConfig).filter_by(id=self._active_config_id).first()
                if row:
                    row.is_loaded = loaded
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Could not update model loaded flag: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def tokenizer(self):
        return self._tokenizer

    def reload_from_db(self):
        self.unload()
        self.load(force_reload=True)

    def unload(self):
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._mark_loaded(False)

    def generate(self, prompt: str) -> tuple[str, float]:
        if not self._pipeline:
            logger.warning("LLM not loaded. Loading from active DB config now...")
            self.load()

        t0 = time.perf_counter()
        if self._pipeline == "mock":
            latency_ms = (time.perf_counter() - t0) * 1000
            return "Mocked LLM answer.", round(latency_ms, 1)

        result = self._pipeline(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        return result[0]["generated_text"].strip(), round(latency_ms, 1)

    def invoke(self, prompt: str):
        """LangChain-like interface used by SafeVoiceMedicalChatbot.

        The medical chatbot calls llm.invoke(prompt). Returning a plain string is
        enough because app.ai_core.models.llm_loader.get_response_text() accepts
        both strings and objects with a .content attribute.
        """
        answer, _ = self.generate(prompt)
        return answer

    @property
    def active_model_name(self) -> str | None:
        return self._active_model_name

    def stream(self, prompt: str, should_stop: Callable[[], bool] | None = None) -> Generator[str, None, None]:
        if not self._pipeline:
            logger.warning("LLM not loaded. Loading from active DB config now...")
            self.load()

        if self._pipeline == "mock":
            for token in "Mocked LLM answer.".split():
                if should_stop and should_stop():
                    break
                yield token + " "
            return

        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
        import threading
        import torch

        class StopOnCancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return bool(should_stop and should_stop())

        streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True, skip_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            do_sample=self._do_sample,
            repetition_penalty=self._repetition_penalty,
            stopping_criteria=StoppingCriteriaList([StopOnCancel()]),
        )
        thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            if should_stop and should_stop():
                break
            yield token

        thread.join(timeout=1)


llm_service = LLMService()
