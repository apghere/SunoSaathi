"""
Offline landmark extraction — processes ISL reference videos with
MediaPipe Holistic to produce the JSON files consumed by the renderer.

Requirements (install before running):
    pip install mediapipe opencv-python

Usage (single video):
    python avatar/extract_landmarks.py --video path/to/hello.mp4 --word hello

Usage (batch — all .mp4 files in a directory):
    python avatar/extract_landmarks.py --video_dir path/to/videos/ --output_dir avatar/landmarks/

The output JSON format per sign:
    {
      "word":   "hello",
      "fps":    int,
      "frames": [
        {
          "pose":       [[x, y, z], …],   // 33 landmarks
          "right_hand": [[x, y, z], …],   // 21 landmarks, or null
          "left_hand":  [[x, y, z], …]    // 21 landmarks, or null
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import LANDMARKS_DIR


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_from_video(
    video_path: str | Path,
    word: str,
    output_dir: Path | None = None,
    max_frames: int = 300,
) -> Path:
    """Extract MediaPipe Holistic landmarks from *video_path* and save as JSON.

    Parameters
    ----------
    video_path  : path to an MP4/AVI/MOV video of a single sign
    word        : the sign word label (used as filename)
    output_dir  : where to write the JSON (defaults to avatar/landmarks/)
    max_frames  : cap extraction at this many frames

    Returns
    -------
    Path to the written JSON file.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Install with:  pip install mediapipe opencv-python"
        ) from e

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[dict] = []

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        def _lm_list(lm_obj) -> list | None:
            if lm_obj is None:
                return None
            return [[round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                    for lm in lm_obj.landmark]

        frames.append({
            "pose":       _lm_list(result.pose_landmarks),
            "right_hand": _lm_list(result.right_hand_landmarks),
            "left_hand":  _lm_list(result.left_hand_landmarks),
        })

    cap.release()
    holistic.close()

    out_dir = output_dir or LANDMARKS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{word}.json"

    data = {"word": word, "fps": round(fps), "frames": frames}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"Wrote {len(frames)} frames → {out_path}")
    return out_path


def batch_extract(
    video_dir: str | Path,
    output_dir: Path | None = None,
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[Path]:
    """Process every video in *video_dir*.

    Filenames are used as word labels (e.g. hello.mp4 → "hello").
    """
    vdir = Path(video_dir)
    results: list[Path] = []
    for vpath in sorted(vdir.iterdir()):
        if vpath.suffix.lower() not in extensions:
            continue
        word = vpath.stem.lower()
        print(f"Processing {vpath.name} …")
        try:
            p = extract_from_video(vpath, word, output_dir)
            results.append(p)
        except Exception as exc:
            print(f"  SKIPPED: {exc}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from ISL reference videos."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",     help="Path to a single video file")
    group.add_argument("--video_dir", help="Directory of video files (batch mode)")
    parser.add_argument("--word",       help="Sign word label (required for --video)")
    parser.add_argument(
        "--output_dir",
        default=str(LANDMARKS_DIR),
        help=f"Output directory for JSON files (default: {LANDMARKS_DIR})",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Maximum frames to extract per video (default: 300)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = Path(args.output_dir)

    if args.video:
        if not args.word:
            print("Error: --word is required when using --video")
            sys.exit(1)
        extract_from_video(args.video, args.word, out, args.max_frames)
    else:
        batch_extract(args.video_dir, out)
