"""
SunoSaathi — central configuration.
All constants referenced in the methodology document are declared here.
Import this module everywhere instead of scattering magic numbers.
"""

from pathlib import Path

# ── Repository root ──────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
LANDMARKS_DIR  = ROOT_DIR / "avatar" / "landmarks"
GLOSSARY_PATH  = DATA_DIR / "sign_glossary.json"

# ── Audio / VAD ──────────────────────────────────────────────────────────────
SAMPLE_RATE          = 16_000   # Hz — required by both webrtcvad and Whisper
CHANNELS             = 1
VAD_AGGRESSIVENESS   = 2        # 0 = permissive … 3 = aggressive
VAD_FRAME_MS         = 30       # Must be 10, 20, or 30 (webrtcvad constraint)
VAD_FRAME_SAMPLES    = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480
VAD_FRAME_BYTES      = VAD_FRAME_SAMPLES * 2                   # 960 (int16)
SILENCE_THRESHOLD_MS = 800      # flush buffer after this much trailing silence
RMS_TARGET_DBFS      = -20.0    # normalise audio to −20 dBFS
MAX_AUDIO_SECONDS    = 28       # hard cap — Whisper context window is 30 s

# ── ASR ──────────────────────────────────────────────────────────────────────
WHISPER_MODEL        = "small"  # tiny | base | small | medium | large
BEAM_WIDTH_ASR       = 5        # beam width for Whisper decoding

# ── Caption simplification ───────────────────────────────────────────────────
MAX_CAPTION_TOKENS   = 40       # split sentences exceeding this token count
MAX_CAPTION_WORDS    = 10       # hard cap per output chunk
BEAM_WIDTH_CAPTION   = 1        # greedy for rule-based; no beam needed

# ── Display ──────────────────────────────────────────────────────────────────
DISPLAY_WPS          = 3        # words per second (NCI guideline: 3–5)
FONT_SIZE_DEFAULT    = 20       # caption font size in px
FONT_SIZE_MIN        = 14
FONT_SIZE_MAX        = 36
READING_SPEED_MIN    = 1
READING_SPEED_MAX    = 7

# ── Session ──────────────────────────────────────────────────────────────────
MAX_TRANSCRIPT_HISTORY = 500    # cap memory usage in long sessions

# ── Avatar / sign animation ──────────────────────────────────────────────────
SIGN_PAUSE_MS        = 400      # pause between consecutive sign animations
SIGN_CANVAS_W        = 260
SIGN_CANVAS_H        = 360

# ── Supported languages ──────────────────────────────────────────────────────
LANGUAGES: dict[str, str] = {
    "Auto-detect": "auto",
    "Hindi":       "hi",
    "Bengali":     "bn",
    "Tamil":       "ta",
    "Telugu":      "te",
    "Marathi":     "mr",
    "Malayalam":   "ml",
    "English":     "en",
}

LANG_NATIVE: dict[str, str] = {
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "ml": "മലയാളം",
    "en": "English",
}

LANG_SCRIPT: dict[str, str] = {
    "hi": "Devanagari",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Devanagari",
    "ml": "Malayalam",
    "en": "Latin",
}

# ── Feature flags ─────────────────────────────────────────────────────────────
ENABLE_LANG_DETECT   = True    # use fastText / langdetect for language ID
ENABLE_AVATAR        = True    # show sign-language avatar panel
ENABLE_WEBVTT        = True    # generate downloadable WebVTT captions
