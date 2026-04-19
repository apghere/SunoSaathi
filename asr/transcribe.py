"""
Whisper ASR wrapper — standalone module, no FastAPI dependency.

Usage:
    from asr.transcribe import load_model, transcribe

    model = load_model("small")
    result = transcribe(model, "path/to/audio.wav", language="hi")
    print(result["text"])        # raw transcript
    print(result["language"])    # detected language code
"""

from __future__ import annotations

import functools
import numpy as np
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Model loader (module-level cache via functools.lru_cache)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def load_model(name: str = "small"):
    """Load and cache a Whisper model by name.

    Parameters
    ----------
    name : {"tiny", "base", "small", "medium", "large"}
    """
    import whisper
    return whisper.load_model(name)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(
    model,
    audio: Union[str, Path, np.ndarray],
    language: str | None = None,
    fp16: bool = False,
) -> dict:
    """Transcribe audio using a pre-loaded Whisper model.

    Parameters
    ----------
    model     : Whisper model loaded with load_model()
    audio     : file path (str/Path) or float32 numpy array (16 kHz, mono)
    language  : ISO-639 code ("hi", "en", …) or None for auto-detect
    fp16      : Use fp16 inference (False for CPU; True only with CUDA)

    Returns
    -------
    dict with keys:
        text      : str  — full transcript
        language  : str  — detected/forced language code
        segments  : list — Whisper segment dicts (with start/end/text)
    """
    kwargs: dict = dict(fp16=fp16, without_timestamps=False)
    if language and language != "auto":
        kwargs["language"] = language

    if isinstance(audio, (str, Path)):
        result = model.transcribe(str(audio), **kwargs)
    else:
        # numpy array — must be float32 at 16 kHz
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        result = model.transcribe(audio, **kwargs)

    return {
        "text":     result.get("text", "").strip(),
        "language": result.get("language", "unknown"),
        "segments": result.get("segments", []),
    }


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Convert raw 16-bit little-endian PCM bytes → float32 in [-1, 1]."""
    audio = np.frombuffer(raw, dtype=np.int16)
    return audio.astype(np.float32) / 32_768.0
