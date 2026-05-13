"""
Speech-to-Text engines.

Priority order:
  1. WhisperSTT       — uses openai-whisper (supports Vietnamese natively, easy install)
  2. FasterWhisperSTT — uses faster-whisper/CTranslate2 (lighter, int8, production-ready)
  3. ZipformerSTT     — uses sherpa-onnx (Vietnamese-specific offline model)
  4. NullSTT          — raises RuntimeError (STT disabled)

Set ENABLE_STT=true in .env and choose a backend via STT_BACKEND=whisper|faster_whisper|zipformer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


# ── Whisper STT (openai-whisper, recommended for ease of use) ─────────────────

class WhisperSTT:
    """
    Wrapper around openai-whisper for Vietnamese speech recognition.

    Install : pip install openai-whisper
    Models  : tiny, base, small, medium, large-v3
              'small' is a good trade-off on CPU; 'large-v3' for best accuracy on GPU.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", language: str = "vi"):
        import whisper  # openai-whisper

        self.language = language
        self.device = device
        self._model = whisper.load_model(model_size, device=device)

    def transcribe_file(self, wav_path: str | Path) -> str:
        result = self._model.transcribe(
            str(wav_path),
            language=self.language,
            fp16=(self.device != "cpu"),
        )
        return result["text"].strip()

    def transcribe_array(self, samples: Any, sample_rate: int = 16000) -> str:
        import numpy as np
        import whisper

        arr = np.array(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[:, 0]

        if sample_rate != 16000:
            try:
                import resampy
                arr = resampy.resample(arr, sample_rate, 16000)
            except ImportError:
                import scipy.signal as ss
                num = int(len(arr) * 16000 / sample_rate)
                arr = ss.resample(arr, num).astype(np.float32)

        arr = whisper.pad_or_trim(arr)
        mel = whisper.log_mel_spectrogram(arr).to(self.device)
        options = whisper.DecodingOptions(language=self.language, fp16=(self.device != "cpu"))
        result = whisper.decode(self._model, mel, options)
        return result.text.strip()


# ── Faster-Whisper (CTranslate2, lighter & faster on CPU) ────────────────────

class FasterWhisperSTT:
    """
    Wrapper around faster-whisper (CTranslate2 quantized Whisper).

    Install : pip install faster-whisper
    Models  : tiny, base, small, medium, large-v3
              int8 quantization runs well on CPU with low memory usage.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "vi",
    ):
        from faster_whisper import WhisperModel

        self.language = language
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_file(self, wav_path: str | Path) -> str:
        segments, _info = self._model.transcribe(str(wav_path), language=self.language)
        return " ".join(seg.text for seg in segments).strip()

    def transcribe_array(self, samples: Any, sample_rate: int = 16000) -> str:
        import numpy as np

        arr = np.array(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[:, 0]
        segments, _info = self._model.transcribe(arr, language=self.language, sampling_rate=sample_rate)
        return " ".join(seg.text for seg in segments).strip()


# ── Zipformer / sherpa-onnx (legacy Vietnamese-specific offline model) ────────

class ZipformerSTT:
    """Wrapper for sherpa-onnx Zipformer Vietnamese STT."""

    def __init__(self, model_dir: str | Path, num_threads: int = 4):
        import sherpa_onnx

        self.model_dir = Path(model_dir)
        required = ['encoder.int8.onnx', 'decoder.onnx', 'joiner.int8.onnx', 'tokens.txt', 'bpe.model']
        missing = [name for name in required if not (self.model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f'Missing STT files in {self.model_dir}: {missing}. Run scripts/download_stt_model.py first.'
            )
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(self.model_dir / 'encoder.int8.onnx'),
            decoder=str(self.model_dir / 'decoder.onnx'),
            joiner=str(self.model_dir / 'joiner.int8.onnx'),
            tokens=str(self.model_dir / 'tokens.txt'),
            bpe_vocab=str(self.model_dir / 'bpe.model'),
            num_threads=num_threads,
            decoding_method='greedy_search',
            debug=False,
        )

    def transcribe_file(self, wav_path: str | Path) -> str:
        import soundfile as sf

        samples, sample_rate = sf.read(str(wav_path), dtype='float32', always_2d=False)
        if getattr(samples, 'ndim', 1) > 1:
            samples = samples[:, 0]
        return self._recognize(samples, int(sample_rate))

    def transcribe_array(self, samples: Any, sample_rate: int = 16000) -> str:
        import numpy as np

        arr = np.array(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[:, 0]
        return self._recognize(arr, sample_rate)

    def _recognize(self, samples: Any, sample_rate: int) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


class NullSTT:
    def transcribe_file(self, wav_path: str | Path) -> str:
        raise RuntimeError('STT is disabled. Set ENABLE_STT=true and provide STT_MODEL_DIR.')

    def transcribe_array(self, samples: Any, sample_rate: int = 16000) -> str:
        raise RuntimeError('STT is disabled. Set ENABLE_STT=true and provide STT_MODEL_DIR.')
