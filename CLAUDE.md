# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

**Start the server (Terminal 1):**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Start the mic client (Terminal 2):**
```bash
python client.py
```

**Switch Whisper model** (tiny/base/small/medium/large — default: small):
```bash
python -m uvicorn server:app -- --model medium
```

**Switch simplifier backend** (none/rules/bart — default: rules):
```bash
python -m uvicorn server:app -- --simplifier none    # raw transcript only
python -m uvicorn server:app -- --simplifier rules   # spaCy rule-based (default)
python -m uvicorn server:app -- --simplifier bart    # facebook/bart-large-cnn
```

**Tune VAD sensitivity** (0=permissive, 3=aggressive — default: 2):
```bash
python client.py --aggressiveness 3
```

**Install spaCy model** (required for `--simplifier rules`):
```bash
python -m spacy download en_core_web_sm
```

## Architecture

The pipeline is split into two processes that communicate over a single WebSocket connection.

```
Mic → PyAudio → webrtcvad (VAD state machine) → WebSocket → Whisper → Simplifier → JSON response
     [client.py]                                            [server.py]
```

**`client.py` — capture + chunking**
- Reads 30 ms frames of 16-bit PCM at 16 kHz from the default mic via PyAudio.
- Runs a ring-buffer VAD state machine: waits until 90% of a 300 ms window is voiced (triggered), then accumulates frames until 90% of the trailing 300 ms window is silent (end of utterance). Hard-flushes at 28 s to stay within Whisper's 30 s context window.
- Sends only voiced chunks (raw PCM bytes) over the WebSocket — silence is discarded locally.
- A separate asyncio task (`receive_loop`) receives and displays JSON messages from the server.

**`server.py` — inference**
- Loads the Whisper model and simplifier once at startup via FastAPI's `lifespan` context.
- `/ws/transcribe`: receives PCM bytes → `pcm16_to_float32` → `run_whisper` (thread-pool) → `_simplifier.simplify` (thread-pool) → sends JSON: `{"transcript": "...", "captions": [...]}`.
- `fp16=False` is hardcoded — required for CPU; Whisper's default `True` crashes on non-CUDA.

**`simplifier.py` — caption simplification**
- `get_simplifier(backend)` factory returns one of three `BaseSimplifier` subclasses.
- `RuleBasedSimplifier`: spaCy `en_core_web_sm` pipeline — removes parentheticals, splits sentences > 40 tokens at dependency clause boundaries (`advcl`, `relcl`, `conj`, `ccomp`), applies a ~50-entry vocabulary lookup table, caps each chunk at 10 words.
- `BARTSimplifier`: HuggingFace `facebook/bart-large-cnn` summarisation (optional, ~1.6 GB).
- `PassthroughSimplifier`: returns the raw transcript unchanged (backend `none`).

## Streamlit frontend

**Run locally:**
```bash
streamlit run app.py
```

**Deploy to Streamlit Community Cloud:**
1. Push repo to GitHub.
2. Go to share.streamlit.io → New app → select `app.py`.
3. The root `requirements.txt` and `packages.txt` are picked up automatically.
4. First load downloads the Whisper model (~142 MB for `base`) — subsequent loads use the cache.

`packages.txt` provides `ffmpeg` to the Cloud environment (Whisper needs it to decode WebM audio from the browser).

The Streamlit app does **not** use `pyaudio`, `webrtcvad`, `fastapi`, or `uvicorn` — those are local-server-only packages. Install them separately with `requirements-server.txt` for the local pipeline.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend — record → Whisper → captions |
| `server.py` | FastAPI app, Whisper inference, simplifier dispatch |
| `client.py` | PyAudio mic capture, webrtcvad chunking, WebSocket client |
| `simplifier.py` | Caption simplification — three swappable backends |
| `requirements.txt` | Streamlit Cloud deps (no pyaudio/webrtcvad) |
| `requirements-server.txt` | Local server extra deps (FastAPI + audio) |
| `packages.txt` | Streamlit Cloud system packages (ffmpeg) |
| `.streamlit/config.toml` | Dark theme + color palette |
| `walkthrough.md` | User-facing guide — one section per pipeline phase |

## Critical constraints

- **Audio format is fixed:** 16 kHz, 16-bit signed little-endian, mono. Both webrtcvad and Whisper require this; changing it in one place requires changing it in both.
- **webrtcvad frame sizes:** The VAD only accepts exactly 10, 20, or 30 ms frames. `FRAME_DURATION = 30` ms → 480 samples → 960 bytes. Frames of any other size will raise at `vad.is_speech()`.
- **Windows dependency:** Use `webrtcvad-wheels` (not `webrtcvad`) — the plain package requires MSVC C++ build tools to compile.
- **Model is global:** `_whisper_model` and `_simplifier` in `server.py` are module-level singletons loaded during lifespan. Not thread-safe for concurrent writes; the current design serializes requests naturally through the executor.
- **WebRTC-VAD frame sizes:** the VAD only accepts exactly 10, 20, or 30 ms frames. `FRAME_DURATION = 30` ms → 480 samples → 960 bytes. Any other size raises at `vad.is_speech()`.
- **Vocabulary pattern ordering:** in `simplifier.py`, `_VOCAB_PAIRS` are sorted by descending string length so multi-word phrases match before their component words. Don't change the sort order.
- **spaCy model required at runtime:** `RuleBasedSimplifier` raises with an install instruction if `en_core_web_sm` is not found. Always run `python -m spacy download en_core_web_sm` after a fresh install.
