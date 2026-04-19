"""
SunoSaathi — Real-Time Multilingual Captioning System
Streamlit MVP: record → Whisper ASR → language detection →
              caption simplification → sign-language avatar → WebVTT export.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="SunoSaathi · Multilingual Captions",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_whisper(name: str):
    import whisper
    return whisper.load_model(name)


@st.cache_resource(show_spinner=False)
def load_simplifier():
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True, capture_output=True,
            )
        from simplifier import get_simplifier
        return get_simplifier("rules")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_glossary_cached() -> dict:
    try:
        from avatar.renderer import load_glossary
        return load_glossary()
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def _lang_detect_fn():
    """Return (detect_fn, backend_name) or (None, 'unavailable')."""
    try:
        from nlp.lang_detect import detect_language
        return detect_language, "available"
    except Exception:
        return None, "unavailable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_lang(text: str) -> dict:
    fn, status = _lang_detect_fn()
    if fn is None or status == "unavailable":
        return {"language": "unknown", "confidence": 0.0, "native": "", "script": "", "backend": "unavailable"}
    try:
        return fn(text)
    except Exception:
        return {"language": "unknown", "confidence": 0.0, "native": "", "script": "", "backend": "error"}


def _extract_kw(captions: list[str]) -> list[str]:
    try:
        from nlp.simplify import extract_keywords_from_captions
        return extract_keywords_from_captions(captions)
    except Exception:
        return []


def _naive_chunks(text: str, limit: int = 10) -> list[str]:
    words = text.split()
    return [" ".join(words[i: i + limit]) for i in range(0, len(words), limit)]


def _is_meaningful(text: str) -> bool:
    words = re.sub(r"[^\w\s]", "", text).strip().split()
    return len(words) >= 2


def _est_duration(text: str, wps: float = 3.0) -> float:
    return max(1.0, len(text.split()) / wps)


# ---------------------------------------------------------------------------
# WebVTT
# ---------------------------------------------------------------------------

def _fmt_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _build_webvtt(history: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    cursor = 0.0
    for i, entry in enumerate(history, 1):
        dur = _est_duration(entry["text"])
        start = cursor
        end = cursor + dur
        caption = entry["simplified"][0] if entry["simplified"] else entry["text"]
        lines += [
            str(i),
            f"{_fmt_vtt_time(start)} --> {_fmt_vtt_time(end)}",
            caption,
            "",
        ]
        cursor = end + 0.3
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sign-language avatar
# ---------------------------------------------------------------------------

def _render_sign_section(keywords: list[str], glossary: dict) -> None:
    from utils.config import LANDMARKS_DIR
    try:
        from avatar.renderer import build_sign_queue, render_avatar, render_setup_prompt
    except Exception:
        st.caption("Avatar module not available.")
        return

    has_landmarks = LANDMARKS_DIR.exists() and any(LANDMARKS_DIR.glob("*.json"))
    if not has_landmarks:
        html_str = render_setup_prompt()
        components.html(html_str, height=440)
        st.caption(
            "Run `python avatar/generate_sample_landmarks.py` once to enable sign animations."
        )
        return

    if not keywords:
        st.caption("No content words detected for sign lookup.")
        return

    queue = build_sign_queue(keywords, glossary)
    if not queue:
        st.caption("No matching signs found in the glossary.")
        return

    html_str = render_avatar(queue)
    components.html(html_str, height=440)

    found = [q["word"] for q in queue if q["frames"]]
    missing = [q["word"] for q in queue if not q["frames"]]
    if found:
        st.caption(f"Signing: {', '.join(found)}")
    if missing:
        st.caption(f"No sign for: {', '.join(missing)} — showing text fallback")


# ---------------------------------------------------------------------------
# Dynamic CSS (accessibility-aware)
# ---------------------------------------------------------------------------

def _inject_css(font_size: int, high_contrast: bool, caption_mode: str) -> None:
    bg_cap   = "#000000" if not high_contrast else "#1a1a00"
    fg_cap   = "#ffffff" if not high_contrast else "#ffff00"
    bg_card  = "#1c1c1e"
    border   = "#2c2c2e"
    accent   = "#0a84ff"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}
#MainMenu, footer {{ visibility: hidden; }}

.suno-header {{ text-align:center; padding:1rem 0 1.5rem; }}
.suno-icon   {{ font-size:2.4rem; line-height:1; }}
.suno-title  {{ font-size:2rem; font-weight:700; letter-spacing:-0.03em; color:#f5f5f7; line-height:1; }}
.suno-sub    {{ font-size:0.85rem; color:#6e6e73; margin-top:.3rem; }}

.suno-label  {{
    font-size:.65rem; font-weight:600; letter-spacing:.1em;
    text-transform:uppercase; color:#6e6e73; margin-bottom:.4rem;
}}
.suno-divider {{ border:none; border-top:1px solid {border}; margin:1.2rem 0; }}

.transcript-card {{
    background:{bg_card}; border:1px solid {border};
    border-radius:12px; padding:1rem 1.2rem; margin:.5rem 0 .8rem;
}}
.transcript-text {{
    font-size:.9rem; font-style:italic; color:#aeaeb2; line-height:1.7;
}}
.lang-badge {{
    display:inline-block; background:{accent}22; color:{accent};
    border:1px solid {accent}55; border-radius:100px;
    padding:.15rem .7rem; font-size:.72rem; font-weight:600;
    margin-left:.6rem; vertical-align:middle;
}}

.caption-screen {{
    background:{bg_cap}; border-radius:12px;
    padding:1.5rem; margin:.5rem 0; min-height:90px;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:.25rem;
}}
.caption-line {{
    display:block; font-size:{font_size}px; font-weight:600;
    color:{fg_cap}; line-height:1.5; text-align:center; max-width:100%;
}}
{'/* scrolling mode */' if caption_mode == "Scrolling" else ''}
.caption-screen {{ overflow-y:auto; max-height:200px; justify-content:flex-start; }}

.suno-meta {{ font-size:.72rem; color:#48484a; text-align:center; margin-top:.4rem; }}
.keyword-chip {{
    display:inline-block; background:#2c2c2e; color:#8e8e93;
    border-radius:100px; padding:.12rem .6rem;
    font-size:.7rem; margin:.15rem .1rem;
}}

/* Language radio pills */
div[data-testid="stRadio"] > div {{
    flex-direction:row !important; flex-wrap:wrap; gap:.35rem;
}}
div[data-testid="stRadio"] label {{
    background:transparent !important; border:1px solid #3a3a3c !important;
    border-radius:100px !important; padding:.25rem .85rem !important;
    font-size:.82rem !important; color:#8e8e93 !important;
    cursor:pointer; transition:.15s;
    white-space:nowrap;
}}
div[data-testid="stRadio"] label:hover {{
    border-color:#636366 !important; color:#e5e5ea !important;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    background:{accent} !important; border-color:{accent} !important;
    color:#fff !important; font-weight:500 !important;
}}
div[data-testid="stRadio"] label input {{ display:none !important; }}
div[data-testid="stAudioInput"] {{ border-radius:12px; overflow:hidden; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "history":         [],
        "session_start":   time.time(),
        "font_size":       20,
        "reading_speed":   3,
        "high_contrast":   False,
        "caption_mode":    "Static",
        "show_avatar":     True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[str, str]:
    """Render sidebar controls and return (model_name, language_code)."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")

        # Model
        model_choice = st.radio(
            "Whisper model",
            options=["tiny", "base"],
            index=1,
            captions=["75 MB · fastest", "142 MB · recommended"],
        )

        st.markdown("---")
        st.markdown("**Language**")
        from utils.config import LANGUAGES, LANG_NATIVE
        lang_name = st.selectbox(
            "Language",
            options=list(LANGUAGES.keys()),
            index=0,
            label_visibility="collapsed",
        )
        lang_code = LANGUAGES[lang_name]

        st.markdown("---")
        st.markdown("**Accessibility**")

        st.session_state.font_size = st.slider(
            "Caption font size (px)",
            min_value=14, max_value=36,
            value=st.session_state.font_size,
        )
        st.session_state.reading_speed = st.slider(
            "Reading speed (words/sec)",
            min_value=1, max_value=7,
            value=st.session_state.reading_speed,
        )
        st.session_state.high_contrast = st.toggle(
            "High-contrast mode",
            value=st.session_state.high_contrast,
        )
        st.session_state.caption_mode = st.radio(
            "Caption display",
            options=["Static", "Scrolling"],
            index=0 if st.session_state.caption_mode == "Static" else 1,
        )
        st.session_state.show_avatar = st.toggle(
            "Sign-language avatar",
            value=st.session_state.show_avatar,
        )

        st.markdown("---")
        st.caption(
            "**SunoSaathi** — real-time multilingual captioning "
            "for Deaf and hard-of-hearing users.\n\n"
            "All inference runs locally — no data leaves your device."
        )

    return model_choice, lang_code


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()

    model_choice, lang_code = _sidebar()

    _inject_css(
        st.session_state.font_size,
        st.session_state.high_contrast,
        st.session_state.caption_mode,
    )

    glossary = load_glossary_cached()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
<div class="suno-header">
    <div class="suno-icon">🤝</div>
    <div class="suno-title">SunoSaathi</div>
    <div class="suno-sub">Speak in your language — get clean captions and sign language instantly.</div>
</div>
""", unsafe_allow_html=True)

    # ── Language selector ─────────────────────────────────────────────────────
    from utils.config import LANGUAGES, LANG_NATIVE
    st.markdown('<div class="suno-label">Language hint</div>', unsafe_allow_html=True)
    lang_pills = {
        f"{k}  {LANG_NATIVE.get(v, '')}".strip(): v
        for k, v in LANGUAGES.items()
    }
    selected_pill = st.radio(
        "Language",
        options=list(lang_pills.keys()),
        horizontal=True,
        label_visibility="collapsed",
        index=list(lang_pills.values()).index(lang_code) if lang_code in lang_pills.values() else 0,
    )
    active_lang = lang_pills[selected_pill]

    st.markdown('<hr class="suno-divider">', unsafe_allow_html=True)

    # ── Record ────────────────────────────────────────────────────────────────
    st.markdown('<div class="suno-label">Record your voice</div>', unsafe_allow_html=True)
    st.caption("Click the mic to start — click again to stop. Speaks any supported language.")
    audio = st.audio_input("Record", label_visibility="collapsed")

    # ── Process ───────────────────────────────────────────────────────────────
    if audio is not None:
        with st.spinner("Transcribing …"):
            model      = load_whisper(model_choice)
            simplifier = load_simplifier()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
                f.write(audio.getvalue())
                tmp_path = f.name

            try:
                import whisper as _whisper_lib

                # Decode WebM → float32 numpy array at 16 kHz once.
                # This lets us inspect the audio before feeding Whisper, and
                # avoids a second ffmpeg decode inside model.transcribe().
                try:
                    raw_audio = _whisper_lib.load_audio(tmp_path)
                except Exception as dec_exc:
                    st.error(
                        f"Could not decode audio ({dec_exc}). "
                        "Ensure ffmpeg is installed and in PATH."
                    )
                    return

                # ── Energy gate ─────────────────────────────────────────────
                # Whisper hallucinates fluent, completely unrelated text when
                # fed near-silence.  This happens when the browser keeps using
                # the laptop mic while the user speaks into their headset mic.
                # Normal speech: RMS ≈ 0.02–0.10 (−34 to −20 dBFS).
                # Wrong/far mic: RMS < 0.001 (below −60 dBFS).
                rms = float(np.sqrt(np.mean(raw_audio ** 2)))
                _RMS_MIN = 0.001
                if rms < _RMS_MIN:
                    st.markdown("""
<div class="transcript-card">
  <div class="suno-label">Microphone issue detected</div>
  <div class="transcript-text">
    Audio signal is too quiet (your earphone/headset mic is not being used).<br><br>
    <b>Fix:</b> click the 🔒 lock icon in your browser address bar →
    <b>Microphone</b> → select your headset mic → refresh the page and try again.
  </div>
</div>
""", unsafe_allow_html=True)
                    return

                whisper_lang = None if active_lang == "auto" else active_lang
                result = model.transcribe(
                    raw_audio,                         # float32 array — no second decode
                    language=whisper_lang,
                    fp16=False,
                    without_timestamps=True,
                    # Prevent cascade hallucinations: don't let a bad first segment
                    # condition all subsequent segments.
                    condition_on_previous_text=False,
                    # Suppress segments Whisper itself marks as low-confidence.
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                )
                transcript = result["text"].strip()
                detected_whisper_lang = result.get("language", "unknown")
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                transcript = ""
                detected_whisper_lang = "unknown"
            finally:
                os.unlink(tmp_path)

        # Language detection
        lang_info = _detect_lang(transcript) if transcript else {}
        detected_lang = lang_info.get("language") or detected_whisper_lang

        if not transcript or not _is_meaningful(transcript):
            st.markdown("""
<div class="transcript-card">
    <div class="suno-label">Result</div>
    <div class="transcript-text">No clear speech detected — try speaking closer to the mic.</div>
</div>
""", unsafe_allow_html=True)
            return

        # ── Simplify & keywords ───────────────────────────────────────────────
        captions = simplifier.simplify(transcript) if simplifier else None
        captions = captions or _naive_chunks(transcript)
        keywords = _extract_kw(captions)

        # ── Save to history ───────────────────────────────────────────────────
        entry = {
            "text":       transcript,
            "language":   detected_lang,
            "simplified": captions,
            "keywords":   keywords,
            "ts":         time.time() - st.session_state.session_start,
        }
        st.session_state.history.append(entry)
        if len(st.session_state.history) > 500:
            st.session_state.history = st.session_state.history[-500:]

        # ── Layout ────────────────────────────────────────────────────────────
        if st.session_state.show_avatar:
            col_left, col_right = st.columns([0.6, 0.4])
        else:
            col_left = st.container()
            col_right = None

        with col_left:
            # Transcript card
            native = LANG_NATIVE.get(detected_lang, "")
            lang_badge = f'<span class="lang-badge">{detected_lang.upper()}{" · " + native if native else ""}</span>'
            safe_tx = html.escape(transcript)
            st.markdown(f"""
<div class="transcript-card">
    <div class="suno-label">Transcript {lang_badge}</div>
    <div class="transcript-text">{safe_tx}</div>
</div>
""", unsafe_allow_html=True)

            # Captions
            lines_html = "\n".join(
                f'<span class="caption-line">{html.escape(c)}</span>'
                for c in captions
            )
            st.markdown(
                f'<div class="suno-label" style="margin-top:.7rem;">Captions</div>'
                f'<div class="caption-screen">{lines_html}</div>',
                unsafe_allow_html=True,
            )

            # Meta
            st.markdown(
                f'<div class="suno-meta">'
                f'{len(captions)} chunk{"s" if len(captions)!=1 else ""} · '
                f'{len(transcript.split())} words · '
                f'model: {model_choice}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Keywords
            if keywords:
                chips = " ".join(f'<span class="keyword-chip">{html.escape(k)}</span>' for k in keywords)
                st.markdown(f'<div style="margin-top:.5rem;">{chips}</div>', unsafe_allow_html=True)

            # Audio playback
            st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
            st.audio(audio)
            st.markdown("</div>", unsafe_allow_html=True)

        # Avatar column
        if col_right is not None:
            with col_right:
                st.markdown('<div class="suno-label">Sign Language</div>', unsafe_allow_html=True)
                _render_sign_section(keywords, glossary)

        # ── Download & History ────────────────────────────────────────────────
        st.markdown('<hr class="suno-divider">', unsafe_allow_html=True)

        dl_col, hist_col = st.columns([1, 2])

        with dl_col:
            if st.session_state.history:
                vtt = _build_webvtt(st.session_state.history)
                st.download_button(
                    label="⬇ Download WebVTT",
                    data=vtt,
                    file_name="sunosaathi_captions.vtt",
                    mime="text/vtt",
                )

        with hist_col:
            if len(st.session_state.history) > 1:
                with st.expander(f"Session history ({len(st.session_state.history)} entries)"):
                    for i, h in enumerate(reversed(st.session_state.history[-20:]), 1):
                        lang_tag = f"[{h['language'].upper()}]" if h.get("language") else ""
                        cap_preview = h["simplified"][0][:60] if h["simplified"] else h["text"][:60]
                        st.markdown(
                            f"`{i}` {lang_tag} **{html.escape(cap_preview)}{'…' if len(cap_preview)==60 else ''}**",
                            unsafe_allow_html=True,
                        )

    else:
        # No audio yet — show sign avatar with any existing history keywords
        if st.session_state.show_avatar and st.session_state.history:
            last = st.session_state.history[-1]
            if last.get("keywords"):
                st.markdown("---")
                st.markdown('<div class="suno-label">Last sign animation</div>', unsafe_allow_html=True)
                _render_sign_section(last["keywords"], glossary)


if __name__ == "__main__":
    main()
