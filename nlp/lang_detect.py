"""
Language identification for SunoSaathi.

Detection priority:
  1. fastText lid.176.bin   — 176-language model, ~130 MB download
  2. langdetect             — Python-only, ~200 KB, less accurate
  3. Passthrough            — returns {"language": "unknown", "confidence": 0.0}

Download fastText model manually (130 MB, run once):
    import urllib.request
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        "lid.176.bin"
    )
Or use the helper:
    from nlp.lang_detect import download_fasttext_model
    download_fasttext_model()
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from utils.config import ROOT_DIR, LANG_NATIVE, LANG_SCRIPT

_FASTTEXT_MODEL_PATH = ROOT_DIR / "lid.176.bin"

# ---------------------------------------------------------------------------
# Loader helpers (cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_fasttext():
    try:
        import fasttext
        if _FASTTEXT_MODEL_PATH.exists():
            model = fasttext.load_model(str(_FASTTEXT_MODEL_PATH))
            return ("fasttext", model)
    except ImportError:
        pass
    return None


@functools.lru_cache(maxsize=1)
def _load_langdetect():
    try:
        from langdetect import detect, detect_langs
        return ("langdetect", detect, detect_langs)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(text: str) -> dict:
    """Detect the language of *text* at the sentence level.

    Returns
    -------
    dict with keys:
        language   : str  — ISO-639-1 code ("hi", "en", …) or "unknown"
        confidence : float — 0.0–1.0
        native     : str  — native script name of the language, or ""
        script     : str  — writing system ("Devanagari", "Latin", …), or ""
        backend    : str  — which backend was used
    """
    text = text.strip()
    if not text or len(text.split()) < 2:
        return _unknown("too short")

    # 1 — fastText
    ft = _load_fasttext()
    if ft:
        _, model = ft
        labels, probs = model.predict(text.replace("\n", " "), k=1)
        lang = labels[0].replace("__label__", "")
        return _result(lang, float(probs[0]), "fasttext")

    # 2 — langdetect
    ld = _load_langdetect()
    if ld:
        _, detect_fn, detect_langs_fn = ld
        try:
            langs = detect_langs_fn(text)
            if langs:
                top = langs[0]
                return _result(top.lang, float(top.prob), "langdetect")
        except Exception:
            pass

    # 3 — script heuristic (very rough fallback)
    lang = _script_heuristic(text)
    if lang:
        return _result(lang, 0.6, "heuristic")

    return _unknown("no backend")


def detect_segments(text: str) -> list[dict]:
    """Detect language per sentence — useful for code-mixed speech.

    Splits *text* at sentence boundaries, classifies each segment,
    and returns a list of dicts (same schema as detect_language).
    """
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    return [
        {**detect_language(s), "text": s}
        for s in sentences if s.strip()
    ]


def download_fasttext_model(dest: Path | None = None) -> Path:
    """Download lid.176.bin if not already present."""
    import urllib.request
    path = dest or _FASTTEXT_MODEL_PATH
    if path.exists():
        print(f"fastText model already at {path}")
        return path
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    print(f"Downloading fastText model to {path} (~130 MB) …")
    urllib.request.urlretrieve(url, str(path))
    print("Done.")
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _result(lang: str, confidence: float, backend: str) -> dict:
    # Normalise: fastText uses "zh-Hans" etc.; strip subtag for lookup
    base = lang.split("-")[0].lower()
    return {
        "language":   base,
        "confidence": round(confidence, 4),
        "native":     LANG_NATIVE.get(base, ""),
        "script":     LANG_SCRIPT.get(base, ""),
        "backend":    backend,
    }


def _unknown(reason: str) -> dict:
    return {
        "language": "unknown", "confidence": 0.0,
        "native": "", "script": "", "backend": reason,
    }


def _script_heuristic(text: str) -> str | None:
    """Rough language guess based on Unicode block presence."""
    ranges = {
        "hi": (0x0900, 0x097F),   # Devanagari
        "bn": (0x0980, 0x09FF),   # Bengali
        "ta": (0x0B80, 0x0BFF),   # Tamil
        "te": (0x0C00, 0x0C7F),   # Telugu
        "ml": (0x0D00, 0x0D7F),   # Malayalam
    }
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in ranges.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.__getitem__)
