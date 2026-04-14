"""
Microphone → VAD → WebSocket → transcript + captions client.

Captures audio from the default mic with PyAudio, splits it into speech
segments using WebRTC-VAD, sends each segment to the server, and prints
the JSON response it receives back:

    {"transcript": "...", "captions": ["...", "..."]}

Run:
    python client.py [--server ws://localhost:8000/ws/transcribe] [--aggressiveness 2]
"""

import argparse
import asyncio
import collections
import json

import pyaudio
import webrtcvad
import websockets

# ---------------------------------------------------------------------------
# Audio / VAD constants
# ---------------------------------------------------------------------------
SAMPLE_RATE       = 16_000  # Hz — webrtcvad and Whisper both prefer 16 kHz
CHANNELS          = 1
FRAME_DURATION    = 30      # ms — webrtcvad accepts 10 / 20 / 30
FRAME_SAMPLES     = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # 480
FRAME_BYTES       = FRAME_SAMPLES * 2                         # int16 → 2 bytes/sample

PADDING_MS        = 300     # ring-buffer length (onset + offset padding)
PADDING_FRAMES    = PADDING_MS // FRAME_DURATION              # 10

MIN_SPEECH_MS     = 400     # ignore blips shorter than this
MIN_SPEECH_FRAMES = MIN_SPEECH_MS // FRAME_DURATION           # 13

MAX_SPEECH_S      = 28      # hard cap — Whisper context is 30 s
MAX_SPEECH_FRAMES = int(MAX_SPEECH_S * 1000 / FRAME_DURATION)

SPEECH_RATIO      = 0.9     # fraction of ring buffer voiced  → trigger ON
SILENCE_RATIO     = 0.9     # fraction of ring buffer silent  → trigger OFF


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def _print_response(raw: str) -> None:
    """Parse server JSON and pretty-print transcript + captions."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: server sent plain text (e.g. older version)
        print(f"\r\033[K[transcript] {raw}")
        return

    transcript = data.get("transcript", "")
    captions   = data.get("captions", [])

    print(f"\r\033[K")                           # clear the "recording…" line
    print(f"  [raw]     {transcript}")
    if captions:
        for i, cap in enumerate(captions, 1):
            print(f"  [cap {i}]   {cap}")
    else:
        print("  [caps]    (none)")
    print()


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------

async def run(server_url: str, aggressiveness: int) -> None:
    vad = webrtcvad.Vad(aggressiveness)

    pa     = pyaudio.PyAudio()
    stream = pa.open(
        rate=SAMPLE_RATE,
        channels=CHANNELS,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=FRAME_SAMPLES,
    )

    print(f"Connecting to {server_url} …")
    async with websockets.connect(server_url) as ws:
        print("Connected. Speak now — Ctrl-C to quit.\n")

        # Background task: receive and display server responses
        async def receive_loop() -> None:
            async for message in ws:
                _print_response(message)

        recv_task = asyncio.create_task(receive_loop())

        # VAD state machine
        ring    = collections.deque(maxlen=PADDING_FRAMES)
        voiced: list[bytes] = []
        triggered = False
        loop = asyncio.get_running_loop()

        try:
            while True:
                # PyAudio read is blocking — run in executor to not stall the loop
                frame: bytes = await loop.run_in_executor(
                    None,
                    lambda: stream.read(FRAME_SAMPLES, exception_on_overflow=False),
                )

                if len(frame) != FRAME_BYTES:
                    continue  # skip rare incomplete frames at startup

                is_speech = vad.is_speech(frame, SAMPLE_RATE)

                if not triggered:
                    ring.append((frame, is_speech))
                    voiced_in_ring = sum(1 for _, s in ring if s)
                    if voiced_in_ring >= SPEECH_RATIO * ring.maxlen:
                        triggered = True
                        voiced.extend(f for f, _ in ring)
                        ring.clear()
                        print("\r\033[K[recording …]", end="", flush=True)
                else:
                    voiced.append(frame)
                    ring.append((frame, is_speech))
                    silent_in_ring = sum(1 for _, s in ring if not s)

                    force_flush  = len(voiced) >= MAX_SPEECH_FRAMES
                    end_of_speech = silent_in_ring >= SILENCE_RATIO * ring.maxlen

                    if end_of_speech or force_flush:
                        if len(voiced) >= MIN_SPEECH_FRAMES:
                            chunk      = b"".join(voiced)
                            duration_s = len(voiced) * FRAME_DURATION / 1000
                            await ws.send(chunk)
                            print(f"\r\033[K[sent {duration_s:.1f}s — waiting …]",
                                  end="", flush=True)

                        voiced    = []
                        ring      = collections.deque(maxlen=PADDING_FRAMES)
                        triggered = False

                await asyncio.sleep(0)  # yield so receive_loop can run

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nStopping …")
        finally:
            recv_task.cancel()
            stream.stop_stream()
            stream.close()
            pa.terminate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mic → VAD → Whisper client")
    parser.add_argument(
        "--server",
        default="ws://localhost:8000/ws/transcribe",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--aggressiveness",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="WebRTC-VAD aggressiveness (0=permissive … 3=aggressive, default: 2)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.server, args.aggressiveness))


if __name__ == "__main__":
    main()
