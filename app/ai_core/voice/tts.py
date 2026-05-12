"""
Text-to-Speech engines for Vietnamese.

Available engines:
  1. EdgeTTSEngine    — Microsoft Edge TTS (free, online, high quality, lightweight)
                        pip install edge-tts  |  no GPU needed, ~1 MB download
  2. VietTTS          — VietTTS (offline, neural, good Vietnamese quality)
                        pip install viettts
  3. VieNeuTTSEngine  — VieNeu-TTS (original, requires vieneu package)
  4. GTTSEngine       — Google TTS (online, simple, acceptable quality)
                        pip install gtts
  5. NullTTS          — raises RuntimeError (TTS disabled)

Recommendation for Vietnamese:
  - Online / fast  → EdgeTTSEngine (best quality, zero GPU)
  - Offline / light → VietTTS (small model ~200 MB, good quality)
"""
from __future__ import annotations

import asyncio
from pathlib import Path


# ── Edge TTS (recommended: free, online, very high quality, lightweight) ──────

class EdgeTTSEngine:
    """
    Microsoft Edge TTS via edge-tts.

    Install : pip install edge-tts
    Voices  : vi-VN-HoaiMyNeural (female), vi-VN-NamMinhNeural (male)
    Usage   : No API key needed. Requires internet connection.
    """

    VOICE_MAP = {
        "female": "vi-VN-HoaiMyNeural",
        "male": "vi-VN-NamMinhNeural",
        "HoaiMy": "vi-VN-HoaiMyNeural",
        "NamMinh": "vi-VN-NamMinhNeural",
    }
    DEFAULT_VOICE = "vi-VN-HoaiMyNeural"

    def __init__(self, voice_id: str | None = "HoaiMy"):
        import edge_tts  # noqa: F401  (ensure it's installed)
        self._edge_tts = edge_tts
        self.default_voice = self.VOICE_MAP.get(voice_id or "", self.DEFAULT_VOICE)

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice_id: str | None = None,
        **kwargs,
    ) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        voice = self.VOICE_MAP.get(voice_id or "", self.default_voice)
        asyncio.run(self._async_synthesize(text, str(output), voice))
        return str(output)

    async def _async_synthesize(self, text: str, output_path: str, voice: str) -> None:
        communicate = self._edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def list_voices(self) -> dict[str, str]:
        return dict(self.VOICE_MAP)


# ── VietTTS (offline, neural, good quality) ───────────────────────────────────

class VietTTSEngine:
    """
    VietTTS offline neural TTS.

    Install : pip install viettts
    Models  : downloaded automatically on first use (~200 MB)
    Docs    : https://github.com/NTT123/viet-tts
    """

    def __init__(self, voice: str = "0"):
        from viettts import tts as _tts_fn
        self._tts_fn = _tts_fn
        self.voice = voice

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice_id: str | None = None,
        **kwargs,
    ) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        voice = voice_id or self.voice
        self._tts_fn(text=text, output=str(output), voice=voice)
        return str(output)

    def list_voices(self) -> dict[str, str]:
        return {"0": "Female (default)", "1": "Male"}


# ── gTTS (Google TTS, simple online fallback) ─────────────────────────────────

class GTTSEngine:
    """
    Google TTS via gtts.

    Install : pip install gtts
    Quality : Acceptable; robotic compared to Edge TTS.
    """

    def __init__(self, lang: str = "vi", slow: bool = False):
        from gtts import gTTS  # noqa: F401
        self._gTTS = gTTS
        self.lang = lang
        self.slow = slow

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        **kwargs,
    ) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tts_obj = self._gTTS(text=text, lang=self.lang, slow=self.slow)
        # gTTS saves mp3; convert to wav if needed
        mp3_path = output.with_suffix(".mp3")
        tts_obj.save(str(mp3_path))
        if output.suffix.lower() == ".wav":
            try:
                import pydub
                audio = pydub.AudioSegment.from_mp3(str(mp3_path))
                audio.export(str(output), format="wav")
                mp3_path.unlink(missing_ok=True)
            except ImportError:
                # Return mp3 if pydub not available
                return str(mp3_path)
        return str(output)

    def list_voices(self) -> dict[str, str]:
        return {"vi": "Vietnamese (Google TTS)"}


# ── VieNeu TTS (original engine) ──────────────────────────────────────────────

class VieNeuTTSEngine:
    """Wrapper for VieNeu-TTS preset voices."""

    def __init__(self, voice_id: str | None = 'Ly'):
        from vieneu import Vieneu

        self.tts = Vieneu()
        self._voices: dict[str, str] = {}
        for desc, name in self.tts.list_preset_voices():
            self._voices[name] = desc

        if voice_id and voice_id in self._voices:
            self.default_voice_id = voice_id
        elif self._voices:
            self.default_voice_id = next(iter(self._voices))
        else:
            self.default_voice_id = None
        self.default_voice_data = (
            self.tts.get_preset_voice(self.default_voice_id) if self.default_voice_id else None
        )

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice_id: str | None = None,
        ref_audio: str | Path | None = None,
    ) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if ref_audio:
            voice_data = self.tts.load_ref_audio(str(ref_audio))
        elif voice_id:
            voice_data = self.tts.get_preset_voice(voice_id)
        else:
            voice_data = self.default_voice_data
        audio = self.tts.infer(text=text, voice=voice_data)
        self.tts.save(audio, str(output))
        return str(output)

    def list_voices(self) -> dict[str, str]:
        return dict(self._voices)


class NullTTS:
    def synthesize(self, text: str, output_path: str | Path, **kwargs) -> str:
        raise RuntimeError('TTS is disabled. Set ENABLE_TTS=true to enable VieNeu-TTS.')
