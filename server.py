"""
FastAPI + WebSocket Whisper transcription server.

Accepts raw 16-bit PCM audio chunks (16 kHz, mono) over a WebSocket
connection and returns JSON messages:

    {"transcript": "<raw whisper output>", "captions": ["chunk1", "chunk2", ...]}

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000

With options (pass after '--'):
    uvicorn server:app -- --model medium --simplifier rules
"""

import argparse
import asyncio
import json
import logging

import numpy as np
import whisper
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from simplifier import get_simplifier, BaseSimplifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # Hz — must match client

# ---------------------------------------------------------------------------
# CLI args — parsed before uvicorn swallows argv
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--model",
    default="small",
    choices=["tiny", "base", "small", "medium", "large"],
)
_parser.add_argument(
    "--simplifier",
    default="rules",
    choices=["none", "rules", "bart"],
    help="Caption simplification backend (default: rules)",
)
_args, _ = _parser.parse_known_args()

MODEL_NAME        : str = _args.model
SIMPLIFIER_BACKEND: str = _args.simplifier

# ---------------------------------------------------------------------------
# Globals — loaded once at startup
# ---------------------------------------------------------------------------
_whisper_model  = None
_simplifier: BaseSimplifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _whisper_model, _simplifier

    logger.info(f"Loading Whisper '{MODEL_NAME}' model …")
    _whisper_model = whisper.load_model(MODEL_NAME)
    logger.info("Whisper ready.")

    logger.info(f"Loading simplifier backend '{SIMPLIFIER_BACKEND}' …")
    _simplifier = get_simplifier(SIMPLIFIER_BACKEND)
    logger.info("Simplifier ready.")

    yield
    logger.info("Shutting down.")


app = FastAPI(title="Whisper Transcription Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Convert raw 16-bit little-endian PCM bytes → float32 in [-1, 1]."""
    audio = np.frombuffer(raw, dtype=np.int16)
    return audio.astype(np.float32) / 32_768.0


def run_whisper(audio: np.ndarray) -> str:
    """
    Transcribe a float32 audio array.
    fp16=False is required on CPU; Whisper's default True raises on non-CUDA.
    """
    result = _whisper_model.transcribe(
        audio,
        fp16=False,
        without_timestamps=True,
    )
    return result["text"].strip()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"Client connected: {websocket.client}")

    loop = asyncio.get_running_loop()
    try:
        while True:
            data: bytes = await websocket.receive_bytes()
            if len(data) < 2:
                continue

            audio      = pcm16_to_float32(data)
            duration_s = len(audio) / SAMPLE_RATE
            logger.info(f"Received chunk: {duration_s:.2f}s ({len(data)} bytes)")

            # Whisper in thread-pool — keeps event loop responsive
            transcript = await loop.run_in_executor(None, run_whisper, audio)

            if not transcript:
                logger.debug("Empty transcript — skipping")
                continue

            # Simplify in thread-pool (BART can be slow on CPU)
            captions = await loop.run_in_executor(None, _simplifier.simplify, transcript)

            logger.info(f"  raw : {transcript}")
            logger.info(f"  caps: {captions}")

            payload = json.dumps({"transcript": transcript, "captions": captions})
            await websocket.send_text(payload)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
    except Exception as exc:
        logger.exception(f"Unhandled error: {exc}")
        await websocket.close(code=1011)
