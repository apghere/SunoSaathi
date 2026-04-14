"""
Suno — Speech to Captions
Streamlit demo: record → Whisper → simplified captions.
"""

import html
import os
import re
import sys
import tempfile

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Suno · Speech to Captions",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Layout ── */
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 660px;
}

/* ── Header ── */
.suno-header {
    text-align: center;
    padding: 1.5rem 0 2.25rem;
}
.suno-icon   { font-size: 2.6rem; line-height: 1; }
.suno-title  {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    color: #F5F5F7;
    margin-top: 0.15rem;
    line-height: 1;
}
.suno-sub {
    font-size: 0.9rem;
    color: #6E6E73;
    margin-top: 0.35rem;
    font-weight: 400;
}

/* ── Section label ── */
.suno-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6E6E73;
    margin-bottom: 0.45rem;
}

/* ── Divider ── */
.suno-divider {
    border: none;
    border-top: 1px solid #2C2C2E;
    margin: 1.5rem 0;
}

/* ── Transcript card ── */
.transcript-card {
    background: #1C1C1E;
    border: 1px solid #2C2C2E;
    border-radius: 14px;
    padding: 1.15rem 1.4rem;
    margin: 0.6rem 0 0.9rem;
}
.transcript-text {
    font-size: 0.95rem;
    font-style: italic;
    color: #AEAEB2;
    line-height: 1.7;
}

/* ── Caption screen ── */
.caption-screen {
    background: #000000;
    border-radius: 14px;
    padding: 1.75rem 1.75rem;
    margin: 0.6rem 0;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.2rem;
}
.caption-line {
    display: block;
    font-size: 1.2rem;
    font-weight: 600;
    color: #FFFFFF;
    line-height: 1.5;
    text-align: center;
    max-width: 520px;
}

/* ── Meta line ── */
.suno-meta {
    font-size: 0.75rem;
    color: #48484A;
    text-align: center;
    margin-top: 0.4rem;
}

/* ── Language radio — pill style ── */
div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    flex-wrap: wrap;
    gap: 0.4rem;
}
div[data-testid="stRadio"] label {
    background: transparent !important;
    border: 1px solid #3A3A3C !important;
    border-radius: 100px !important;
    padding: 0.28rem 0.9rem !important;
    font-size: 0.85rem !important;
    color: #8E8E93 !important;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s, background 0.15s;
    white-space: nowrap;
}
div[data-testid="stRadio"] label:hover {
    border-color: #636366 !important;
    color: #E5E5EA !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #0A84FF !important;
    border-color: #0A84FF !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
}
div[data-testid="stRadio"] label input { display: none !important; }

/* ── Audio input — subtle ── */
div[data-testid="stAudioInput"] {
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_whisper_model(name: str):
    import whisper
    return whisper.load_model(name)


@st.cache_resource(show_spinner=False)
def load_simplifier():
    """Load the rule-based simplifier; auto-download the spaCy model if absent."""
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
                capture_output=True,
            )
        from simplifier import get_simplifier
        return get_simplifier("rules")
    except Exception:
        return None   # degrade to naive chunking


# ── Constants ─────────────────────────────────────────────────────────────────

LANGUAGES: dict[str, str] = {
    "Hindi":   "hi",
    "Tamil":   "ta",
    "Bengali": "bn",
}
LANG_SCRIPT: dict[str, str] = {
    "Hindi":   "हिन्दी",
    "Tamil":   "தமிழ்",
    "Bengali": "বাংলা",
}

MODEL_INFO: dict[str, str] = {
    "tiny": "75 MB · fastest",
    "base": "142 MB · recommended",
}


def _is_meaningful(text: str) -> bool:
    """Filter out Whisper silence/noise artifacts ('...', ' you.', etc.)."""
    words = re.sub(r'[^\w\s]', '', text).strip().split()
    return len(words) >= 2


def _naive_chunks(text: str, limit: int = 10) -> list[str]:
    words = text.split()
    return [' '.join(words[i:i+limit]) for i in range(0, len(words), limit)]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Settings")
    st.markdown("---")
    model_choice = st.radio(
        "Whisper model",
        options=list(MODEL_INFO.keys()),
        index=1,
        captions=list(MODEL_INFO.values()),
    )
    st.markdown("---")
    st.caption(
        "Models download on first use and are cached locally.  \n"
        "All inference runs on CPU — no GPU needed."
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="suno-header">
    <div class="suno-icon">🎙</div>
    <div class="suno-title">Suno</div>
    <div class="suno-sub">Speak in your language — get clean captions instantly.</div>
</div>
""", unsafe_allow_html=True)


# ── Language selector ─────────────────────────────────────────────────────────

st.markdown('<div class="suno-label">Language</div>', unsafe_allow_html=True)
lang_key = st.radio(
    "Language",
    options=list(LANGUAGES.keys()),
    format_func=lambda k: f"{k}  {LANG_SCRIPT[k]}",
    horizontal=True,
    label_visibility="collapsed",
)
lang_code = LANGUAGES[lang_key]

st.markdown('<hr class="suno-divider">', unsafe_allow_html=True)


# ── Record ────────────────────────────────────────────────────────────────────

st.markdown('<div class="suno-label">Record your voice</div>', unsafe_allow_html=True)
st.caption("Click the mic to start recording — click again to stop.")
audio = st.audio_input("Record", label_visibility="collapsed")


# ── Process & display ─────────────────────────────────────────────────────────

if audio is not None:
    with st.spinner("Transcribing …"):
        model     = load_whisper_model(model_choice)
        simplifier = load_simplifier()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
            f.write(audio.getvalue())
            tmp_path = f.name

        try:
            result     = model.transcribe(
                tmp_path,
                language=lang_code,
                fp16=False,
                without_timestamps=True,
            )
            transcript = result["text"].strip()
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            transcript = ""
        finally:
            os.unlink(tmp_path)

    # ── No speech ────────────────────────────────────────────────────────────
    if not transcript or not _is_meaningful(transcript):
        st.markdown("""
        <div class="transcript-card">
            <div class="suno-label">Result</div>
            <div class="transcript-text">
                No clear speech detected — try speaking a bit closer to the mic,
                or switch to <strong>tiny</strong> for shorter recordings.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        word_count = len(transcript.split())

        # ── Transcript card ───────────────────────────────────────────────────
        safe_tx = html.escape(transcript)
        st.markdown(f"""
        <div class="transcript-card">
            <div class="suno-label">Transcript</div>
            <div class="transcript-text">{safe_tx}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Caption screen ────────────────────────────────────────────────────
        captions = simplifier.simplify(transcript) if simplifier else _naive_chunks(transcript)
        captions = captions or _naive_chunks(transcript)  # ultimate fallback

        lines_html = "\n".join(
            f'<span class="caption-line">{html.escape(c)}</span>'
            for c in captions
        )
        st.markdown(f"""
        <div class="suno-label" style="margin-top:0.75rem;">Captions</div>
        <div class="caption-screen">{lines_html}</div>
        """, unsafe_allow_html=True)

        # ── Meta ──────────────────────────────────────────────────────────────
        n = len(captions)
        st.markdown(
            f'<div class="suno-meta">'
            f'{n} caption chunk{"s" if n != 1 else ""}  ·  {word_count} words'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Audio playback ────────────────────────────────────────────────────
        st.markdown("<div style='margin-top:1.25rem;'>", unsafe_allow_html=True)
        st.audio(audio)
        st.markdown("</div>", unsafe_allow_html=True)
