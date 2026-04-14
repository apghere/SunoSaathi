# Pipeline Walkthrough

This document explains what each phase of the pipeline does, how to run it, and what to expect. It grows with the project — each new phase gets its own section.

---

## Overview

```
Mic → [VAD chunker] → WebSocket → [Whisper] → [Simplifier] → caption chunks
       client.py                   server.py    server.py
```

The pipeline is split into two processes connected by a single WebSocket:

| Process | File | Responsibility |
|---|---|---|
| Client | `client.py` | Mic capture, VAD segmentation, sending audio |
| Server | `server.py` | Whisper transcription, caption simplification |

---

## Phase 1 — Real-time Audio Transcription

### What it does

`client.py` captures audio from your default microphone in 30 ms frames. A Voice Activity Detection (VAD) state machine accumulates voiced frames and discards silence. When a speech segment ends, the raw PCM chunk is sent to the server over WebSocket. The server decodes it with OpenAI Whisper and returns a transcript.

Nothing here is trained or fine-tuned. Whisper `small` handles English, Hindi, Tamil, Bengali, and most other languages out of the box.

### Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Windows note:** `requirements.txt` uses `webrtcvad-wheels`, not `webrtcvad`. The plain package requires MSVC to compile.

### Running

Open two terminals in `E:/Projects/suno/`.

**Terminal 1 — server:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

You should see:
```
Loading Whisper 'small' model …
Whisper ready.
Loading simplifier backend 'rules' …
Simplifier ready.
```

Whisper downloads ~461 MB on first launch (cached in `~/.cache/whisper` after that).

**Terminal 2 — client:**
```bash
python client.py
```

You should see:
```
Connecting to ws://localhost:8000/ws/transcribe …
Connected. Speak now — Ctrl-C to quit.
```

### What you'll see when you speak

```
[recording …]               ← VAD detected speech onset
[sent 2.3s — waiting …]     ← chunk sent, Whisper is running
                             ← blank line
  [raw]     Hello, my name is Arjun and I am currently working on this feature.
  [cap 1]   Hello, my name is Arjun and I am now working
```

- `[raw]` is the unmodified Whisper output.
- `[cap N]` is the simplified caption chunk (Phase 2).

### Tuning VAD

The VAD aggressiveness controls how aggressively non-speech is filtered:

| Value | Best for |
|---|---|
| `0` | Very quiet rooms |
| `1` | Normal indoors |
| `2` | Default — works for most environments |
| `3` | Noisy rooms, music in background |

```bash
python client.py --aggressiveness 3
```

**If short utterances are getting dropped** — lower aggressiveness or reduce `MIN_SPEECH_MS` in `client.py` (default 400 ms).

**If Whisper receives lots of empty or near-silence chunks** — raise aggressiveness.

### Switching Whisper models

```bash
# Pass args after the uvicorn separator '--'
python -m uvicorn server:app -- --model medium
```

| Model | Size | Notes |
|---|---|---|
| `tiny` | 75 MB | Fast, lower accuracy |
| `base` | 142 MB | Good for English-only |
| `small` | 461 MB | **Default** — solid multilingual |
| `medium` | 1.5 GB | Better accuracy, slower on CPU |
| `large` | 3 GB | Best accuracy, needs ≥8 GB RAM |

---

## Phase 2 — Caption Simplification

### What it does

After Whisper produces a raw transcript, the server passes it through a simplifier before sending the response. The simplifier breaks long sentences into short caption-friendly chunks (≤ 10 words) that are easier to read on screen.

Three backends are available, selectable at server startup.

### Backend A: Rule-based (default)

**Flag:** `--simplifier rules`

A lightweight spaCy pipeline. No model weights beyond the 12 MB `en_core_web_sm`.

**Steps applied to each sentence:**

1. **Parenthetical removal** — strips `(…)` and `[…]` blocks.
2. **Dependency split** — if a sentence exceeds 40 tokens, it is split at clause boundaries detected in the spaCy dependency tree (`advcl`, `relcl`, `conj`, `ccomp`). Each clause becomes a separate chunk.
3. **Vocabulary lookup** — replaces ~50 complex words/phrases with simpler equivalents using a case-insensitive regex table.

   Selected substitutions:

   | Complex | Simple |
   |---|---|
   | `demonstrated` | `showed` |
   | `significant` | `big` |
   | `approximately` | `about` |
   | `subsequently` | `then` |
   | `in order to` | `to` |
   | `due to the fact that` | `because` |
   | `facilitate` | `help` |
   | `numerous` | `many` |

   Full table: see `_VOCAB_RAW` in `simplifier.py`.

4. **Word cap** — each chunk is hard-truncated to 10 words.

**Example:**

```
IN  : The patient demonstrated significant improvement after the new treatment
      was administered by the medical team (as noted in the report).

OUT : The patient showed big improvement after the new treatment was
```

**Non-English text** passes through the vocab step unchanged (no matches) and is not dependency-split if under 40 tokens. For short Hindi or Tamil utterances Whisper produces, the rule-based output is the same as the raw transcript, which is acceptable at this stage.

### Backend B: BART (facebook/bart-large-cnn)

**Flag:** `--simplifier bart`

Uses HuggingFace `transformers` to summarise each chunk. Higher quality than rules on English. Slower on CPU (~2–5 s per chunk). Downloads ~1.6 GB on first use.

**Extra install required:**
```bash
pip install transformers sentencepiece
```

**Run:**
```bash
python -m uvicorn server:app -- --simplifier bart
```

BART targets a 8–30 token output per chunk. It does not have the 10-word cap the rule-based path applies; adjust `max_length` in `BARTSimplifier.__init__` if you want shorter output.

**When to use:** if you want noticeably better English simplification without training anything. Not recommended for non-English text — BART-CNN was trained on English news.

### Backend C: None (passthrough)

**Flag:** `--simplifier none`

Disables simplification entirely. The server returns the raw Whisper transcript as a single caption string. Useful for debugging the audio pipeline in isolation.

```bash
python -m uvicorn server:app -- --simplifier none
```

### Running with a specific backend

```bash
# Terminal 1
python -m uvicorn server:app -- --model small --simplifier rules

# Terminal 2
python client.py --aggressiveness 2
```

### What you'll see (rule-based example)

Speak: *"The system will subsequently facilitate the utilization of numerous additional resources in order to commence the implementation."*

```
  [raw]     The system will subsequently facilitate the utilization of numerous additional resources in order to commence the implementation.
  [cap 1]   The system will then help the utilization of many more
```

### Tuning

**Shorter captions:** decrease `MAX_WORDS` in `simplifier.py` (default `10`).

**Less aggressive splitting:** increase `LONG_SENT_TOKENS` (default `40`) to require longer sentences before dependency splitting activates.

**Adding vocab entries:** edit `_VOCAB_RAW` in `simplifier.py`. Multi-word phrases must be listed before their component single words — the regex loop processes them in descending length order.

**Cap behaviour:** the 10-word cap is a hard truncation at whitespace boundaries. If truncated chunks feel cut off, raise `MAX_WORDS` to `12–15` and evaluate visually.

---

## Phase 3 — Streamlit Frontend

### What it does

`app.py` is a self-contained Streamlit app that replaces the two-terminal local pipeline with a single browser tab. Audio recording happens in the browser via `st.audio_input` (WebM format). The recorded chunk is decoded by Whisper on the server (or Streamlit Cloud), and the transcript + captions are displayed immediately.

```
Browser mic → st.audio_input (WebM) → Whisper (CPU) → Simplifier → caption screen
                               [app.py — single process]
```

### Prerequisites

```bash
pip install -r requirements.txt      # streamlit, openai-whisper, torch, spacy, numpy
streamlit run app.py                 # first run downloads the Whisper model
```

The app auto-downloads `en_core_web_sm` on first launch if the spaCy model is missing.

### Running locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. No separate server or client terminal needed.

### Deploying to Streamlit Community Cloud (free)

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo → set **Main file path** to `app.py`.
3. Click **Deploy**. The platform picks up `requirements.txt` and `packages.txt` automatically.

`packages.txt` installs `ffmpeg` on the Cloud host — Whisper needs it to decode the WebM audio the browser sends.

**First cold start:** the Whisper model downloads on first use (~142 MB for `base`). After that it's cached across restarts.

### What you'll see

The UI has three zones:

1. **Language pills** — select Hindi, Tamil, or Bengali. This passes a language hint to Whisper, improving accuracy and speed.

2. **Record button** — a single click starts recording; another click stops it. The recording is processed immediately after stopping.

3. **Results** — two cards appear:

   ```
   ┌──────────────────────────────────────────────┐
   │ TRANSCRIPT                                   │
   │ "The system demonstrated significant         │
   │  improvement after the new module..."        │
   └──────────────────────────────────────────────┘

   CAPTIONS
   ┌──────────────────────────────────────────────┐
   │                                              │
   │   The system showed big improvement          │
   │   after the new module was                   │
   │                                              │
   └──────────────────────────────────────────────┘
   ```

   The black caption screen mimics how subtitles appear on a TV or video player.

### Model selection

Open the sidebar (chevron on the top-left) to switch between:

| Model | Size | Notes |
|---|---|---|
| `tiny` | 75 MB | Fastest; good for clear English; less reliable for Indian languages |
| `base` | 142 MB | **Default** — good balance for Hindi/Tamil/Bengali on CPU |

Larger models (`small`, `medium`) can be added to `MODEL_INFO` in `app.py` for local use, but may hit memory limits on Streamlit Community Cloud's free tier (1 GB RAM).

### Indian language behaviour

Whisper natively transcribes Hindi, Tamil, and Bengali — select the right language to give it a hint. The rule-based simplifier processes these as-is (no vocabulary substitutions match, and spaCy's English dependency parser doesn't apply), so each caption chunk is simply the first 10 words of a sentence. That's fine — the goal here is the end-to-end pipeline, not NLP quality for non-English text.

### Tuning

**Captions too long:** lower `MAX_WORDS` in `simplifier.py` (default `10`).  
**Captions cut off awkwardly:** raise `MAX_WORDS` to `12` — 10 is conservative.  
**Silence detected as speech:** Whisper's silence artifact ("... you.", " Thank you.") is filtered by `_is_meaningful()` in `app.py`, which requires at least 2 real words. Adjust the threshold there if needed.

---

*More phases to follow — subtitle file export (.srt), live overlay rendering, speaker diarization.*
