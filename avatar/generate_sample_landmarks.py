"""
Generate synthetic MediaPipe-format landmark sequences for demo purposes.

These are NOT real ISL signs — they are simplified arm/hand animations
that demonstrate the full pipeline (speech → text → stick figure) without
requiring CC-licensed ISL video footage.

Run once:
    python avatar/generate_sample_landmarks.py

Creates JSON files in avatar/landmarks/ for the 15 most common signs.
To add real ISL signs, use avatar/extract_landmarks.py on actual videos.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import LANDMARKS_DIR


# ---------------------------------------------------------------------------
# Base standing pose (33 MediaPipe Pose landmarks, normalised 0-1)
# ---------------------------------------------------------------------------
#   x: 0 = left edge of frame, 1 = right edge
#   y: 0 = top of frame,       1 = bottom
# ---------------------------------------------------------------------------

BASE_POSE_XY: dict[int, tuple[float, float]] = {
    0:  (0.500, 0.140),   # nose
    1:  (0.520, 0.124),   # left eye inner
    2:  (0.535, 0.121),   # left eye
    3:  (0.550, 0.124),   # left eye outer
    4:  (0.480, 0.124),   # right eye inner
    5:  (0.465, 0.121),   # right eye
    6:  (0.450, 0.124),   # right eye outer
    7:  (0.562, 0.134),   # left ear
    8:  (0.438, 0.134),   # right ear
    9:  (0.512, 0.171),   # mouth left
    10: (0.488, 0.171),   # mouth right
    11: (0.380, 0.340),   # left shoulder
    12: (0.620, 0.340),   # right shoulder
    13: (0.315, 0.508),   # left elbow
    14: (0.685, 0.508),   # right elbow
    15: (0.265, 0.648),   # left wrist
    16: (0.735, 0.648),   # right wrist
    17: (0.247, 0.672),   # left pinky
    18: (0.753, 0.672),   # right pinky
    19: (0.256, 0.660),   # left index
    20: (0.744, 0.660),   # right index
    21: (0.270, 0.666),   # left thumb
    22: (0.730, 0.666),   # right thumb
    23: (0.410, 0.720),   # left hip
    24: (0.590, 0.720),   # right hip
    25: (0.407, 0.848),   # left knee
    26: (0.593, 0.848),   # right knee
    27: (0.406, 0.954),   # left ankle
    28: (0.594, 0.954),   # right ankle
    29: (0.400, 0.963),   # left heel
    30: (0.600, 0.963),   # right heel
    31: (0.397, 0.981),   # left foot index
    32: (0.603, 0.981),   # right foot index
}


def _full_pose(overrides: dict[int, tuple[float, float]] | None = None) -> list:
    """Return all 33 pose landmarks as [[x, y, 0], …], with optional overrides."""
    base = dict(BASE_POSE_XY)
    if overrides:
        base.update(overrides)
    return [[round(base[i][0], 4), round(base[i][1], 4), 0.0] for i in range(33)]


# ---------------------------------------------------------------------------
# Hand shape generators (21 landmarks, all relative to wrist position)
# ---------------------------------------------------------------------------

def _open_right_hand(wx: float, wy: float) -> list:
    """Open palm, right hand, fingers spread upward."""
    offsets = [
        (0.000,  0.000),   # 0  wrist
        (-0.025, -0.020),  # 1  thumb mcp
        (-0.040, -0.045),  # 2  thumb pip
        (-0.052, -0.068),  # 3  thumb dip
        (-0.062, -0.090),  # 4  thumb tip
        (-0.008, -0.042),  # 5  index mcp
        (-0.008, -0.088),  # 6  index pip
        (-0.008, -0.118),  # 7  index dip
        (-0.008, -0.144),  # 8  index tip
        (0.010,  -0.044),  # 9  middle mcp
        (0.010,  -0.092),  # 10 middle pip
        (0.010,  -0.124),  # 11 middle dip
        (0.010,  -0.152),  # 12 middle tip
        (0.028,  -0.040),  # 13 ring mcp
        (0.028,  -0.085),  # 14 ring pip
        (0.028,  -0.114),  # 15 ring dip
        (0.028,  -0.140),  # 16 ring tip
        (0.045,  -0.028),  # 17 pinky mcp
        (0.045,  -0.064),  # 18 pinky pip
        (0.045,  -0.090),  # 19 pinky dip
        (0.045,  -0.112),  # 20 pinky tip
    ]
    return [[round(wx + dx, 4), round(wy + dy, 4), 0.0] for dx, dy in offsets]


def _fist_right_hand(wx: float, wy: float) -> list:
    """Closed fist, right hand."""
    offsets = [
        (0.000,  0.000),   # wrist
        (-0.020, -0.014),  # 1
        (-0.030, -0.028),  # 2
        (-0.030, -0.040),  # 3
        (-0.030, -0.050),  # 4 thumb tip
        (-0.008, -0.030),  # 5
        (-0.006, -0.038),  # 6
        (-0.005, -0.028),  # 7
        (-0.004, -0.018),  # 8
        (0.010,  -0.030),  # 9
        (0.012,  -0.038),  # 10
        (0.010,  -0.028),  # 11
        (0.008,  -0.018),  # 12
        (0.025,  -0.025),  # 13
        (0.026,  -0.034),  # 14
        (0.024,  -0.025),  # 15
        (0.022,  -0.016),  # 16
        (0.038,  -0.018),  # 17
        (0.038,  -0.026),  # 18
        (0.036,  -0.018),  # 19
        (0.034,  -0.010),  # 20
    ]
    return [[round(wx + dx, 4), round(wy + dy, 4), 0.0] for dx, dy in offsets]


def _point_right_hand(wx: float, wy: float) -> list:
    """Pointing hand (index extended, others curled)."""
    offsets = [
        (0.000,  0.000),
        (-0.020, -0.014), (-0.030, -0.028), (-0.032, -0.042), (-0.032, -0.054),  # thumb
        (-0.008, -0.040), (-0.008, -0.086), (-0.008, -0.118), (-0.008, -0.146),  # index (extended)
        (0.010,  -0.030), (0.012,  -0.038), (0.010,  -0.028), (0.008,  -0.018),  # middle curled
        (0.025,  -0.025), (0.026,  -0.034), (0.024,  -0.025), (0.022,  -0.016),  # ring
        (0.038,  -0.018), (0.038,  -0.026), (0.036,  -0.018), (0.034,  -0.010),  # pinky
    ]
    return [[round(wx + dx, 4), round(wy + dy, 4), 0.0] for dx, dy in offsets]


def _flat_right_hand(wx: float, wy: float) -> list:
    """Flat B-handshape, fingers together and horizontal."""
    offsets = [
        (0.000,  0.000),
        (-0.022, -0.016), (-0.036, -0.026), (-0.044, -0.036), (-0.050, -0.044),
        (-0.008, -0.038), (-0.012, -0.076), (-0.012, -0.104), (-0.012, -0.128),
        (0.006,  -0.040), (0.006,  -0.080), (0.006,  -0.110), (0.006,  -0.136),
        (0.020,  -0.038), (0.020,  -0.076), (0.020,  -0.104), (0.020,  -0.128),
        (0.033,  -0.032), (0.033,  -0.066), (0.033,  -0.092), (0.033,  -0.114),
    ]
    return [[round(wx + dx, 4), round(wy + dy, 4), 0.0] for dx, dy in offsets]


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interp_pose(pose_a: dict, pose_b: dict, t: float) -> dict:
    """Linearly interpolate between two pose override dicts."""
    keys = set(pose_a) | set(pose_b)
    result = {}
    for k in keys:
        ax, ay = pose_a.get(k, BASE_POSE_XY.get(k, (0.5, 0.5)))
        bx, by = pose_b.get(k, BASE_POSE_XY.get(k, (0.5, 0.5)))
        result[k] = (_lerp(ax, bx, t), _lerp(ay, by, t))
    return result


def _smooth(t: float) -> float:
    """Smooth-step easing."""
    return t * t * (3 - 2 * t)


def _build_frames(
    keyframes: list[dict],
    fps: int = 15,
) -> list[dict]:
    """Build animation frames by interpolating between keyframes.

    Each keyframe is a dict:
        pose_overrides : dict[int, (x, y)]  — which pose landmarks to move
        right_hand     : callable(wx, wy) → 21-landmark list, or None
        left_hand      : callable(wx, wy) → 21-landmark list, or None
        duration_s     : how long to hold / transition into this keyframe
    """
    frames = []
    for ki in range(len(keyframes) - 1):
        kf_a = keyframes[ki]
        kf_b = keyframes[ki + 1]
        n = max(1, round(kf_b.get("duration_s", 0.5) * fps))
        for fi in range(n):
            t = _smooth(fi / n)
            pose_ov = _interp_pose(
                kf_a.get("pose_overrides", {}),
                kf_b.get("pose_overrides", {}),
                t,
            )
            pose = _full_pose(pose_ov)

            # Wrist positions for hand placement
            wx_r, wy_r = pose[16][0], pose[16][1]
            wx_l, wy_l = pose[15][0], pose[15][1]

            rh_fn_a = kf_a.get("right_hand")
            rh_fn_b = kf_b.get("right_hand") or rh_fn_a
            lh_fn_a = kf_a.get("left_hand")
            lh_fn_b = kf_b.get("left_hand") or lh_fn_a

            rh = rh_fn_b(wx_r, wy_r) if rh_fn_b else None
            lh = lh_fn_b(wx_l, wy_l) if lh_fn_b else None

            frames.append({
                "pose":       pose,
                "right_hand": rh,
                "left_hand":  lh,
            })
    return frames


# ---------------------------------------------------------------------------
# Sign definitions
# ---------------------------------------------------------------------------

def _sign_hello(fps: int = 15) -> dict:
    """Wave: right arm raised, hand sweeping left–right."""
    arm_raised = {14: (0.690, 0.295), 16: (0.720, 0.145), 20: (0.720, 0.145), 18: (0.720, 0.145)}
    keyframes = [
        {"pose_overrides": {}, "right_hand": _open_right_hand, "duration_s": 0.3},
        {"pose_overrides": arm_raised, "right_hand": _open_right_hand, "duration_s": 0.3},
        {"pose_overrides": {**arm_raised, 16: (0.630, 0.140)}, "right_hand": _open_right_hand, "duration_s": 0.3},
        {"pose_overrides": {**arm_raised, 16: (0.800, 0.140)}, "right_hand": _open_right_hand, "duration_s": 0.3},
        {"pose_overrides": {**arm_raised, 16: (0.630, 0.140)}, "right_hand": _open_right_hand, "duration_s": 0.3},
        {"pose_overrides": {**arm_raised, 16: (0.800, 0.140)}, "right_hand": _open_right_hand, "duration_s": 0.3},
        {"pose_overrides": {}, "right_hand": _open_right_hand, "duration_s": 0.2},
    ]
    return {"word": "hello", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_thank_you(fps: int = 15) -> dict:
    """Flat hand from chin moving forward."""
    at_chin  = {14: (0.645, 0.320), 16: (0.520, 0.200)}
    extended = {14: (0.660, 0.345), 16: (0.555, 0.250)}
    keyframes = [
        {"pose_overrides": at_chin,  "right_hand": _flat_right_hand, "duration_s": 0.4},
        {"pose_overrides": extended, "right_hand": _flat_right_hand, "duration_s": 0.5},
        {"pose_overrides": at_chin,  "right_hand": _flat_right_hand, "duration_s": 0.3},
    ]
    return {"word": "thank_you", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_yes(fps: int = 15) -> dict:
    """Fist nodding down–up."""
    neutral = {14: (0.685, 0.380), 16: (0.685, 0.470)}
    down    = {14: (0.688, 0.400), 16: (0.688, 0.498)}
    up      = {14: (0.680, 0.355), 16: (0.682, 0.440)}
    keyframes = [
        {"pose_overrides": neutral, "right_hand": _fist_right_hand, "duration_s": 0.2},
        {"pose_overrides": down,    "right_hand": _fist_right_hand, "duration_s": 0.2},
        {"pose_overrides": up,      "right_hand": _fist_right_hand, "duration_s": 0.2},
        {"pose_overrides": down,    "right_hand": _fist_right_hand, "duration_s": 0.2},
        {"pose_overrides": neutral, "right_hand": _fist_right_hand, "duration_s": 0.2},
    ]
    return {"word": "yes", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_no(fps: int = 15) -> dict:
    """Index finger wagging left–right."""
    center = {14: (0.685, 0.370), 16: (0.685, 0.445)}
    left   = {14: (0.670, 0.370), 16: (0.655, 0.440)}
    right  = {14: (0.700, 0.370), 16: (0.715, 0.440)}
    keyframes = [
        {"pose_overrides": center, "right_hand": _point_right_hand, "duration_s": 0.15},
        {"pose_overrides": right,  "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": left,   "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": right,  "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": center, "right_hand": _point_right_hand, "duration_s": 0.15},
    ]
    return {"word": "no", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_help(fps: int = 15) -> dict:
    """Right flat hand placed on left fist, both hands rise."""
    start = {13: (0.330, 0.520), 15: (0.290, 0.640),
             14: (0.660, 0.520), 16: (0.640, 0.620)}
    rise  = {13: (0.328, 0.470), 15: (0.288, 0.580),
             14: (0.658, 0.470), 16: (0.638, 0.560)}
    keyframes = [
        {"pose_overrides": start, "right_hand": _flat_right_hand, "left_hand": _fist_right_hand, "duration_s": 0.4},
        {"pose_overrides": rise,  "right_hand": _flat_right_hand, "left_hand": _fist_right_hand, "duration_s": 0.5},
        {"pose_overrides": start, "right_hand": _flat_right_hand, "left_hand": _fist_right_hand, "duration_s": 0.3},
    ]
    return {"word": "help", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_please(fps: int = 15) -> dict:
    """Open right hand making circular motion on chest."""
    c = {14: (0.655, 0.390), 16: (0.585, 0.470)}
    # Circular path: right → down → left → up → right
    traj = [
        {14: (0.660, 0.390), 16: (0.600, 0.465)},
        {14: (0.668, 0.408), 16: (0.610, 0.488)},
        {14: (0.658, 0.418), 16: (0.592, 0.500)},
        {14: (0.644, 0.410), 16: (0.575, 0.490)},
        {14: (0.640, 0.396), 16: (0.572, 0.475)},
        {14: (0.648, 0.386), 16: (0.582, 0.462)},
        c,
    ]
    keyframes = [
        {**{"pose_overrides": p, "right_hand": _open_right_hand, "duration_s": 0.12}}
        for p in [c] + traj
    ]
    return {"word": "please", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_good(fps: int = 15) -> dict:
    """Thumbs-up moving slightly forward."""
    start = {14: (0.685, 0.390), 16: (0.710, 0.480)}
    fwd   = {14: (0.680, 0.380), 16: (0.700, 0.460)}
    # thumb-up hand: open palm rotated = just use fist pointing up
    keyframes = [
        {"pose_overrides": start, "right_hand": _fist_right_hand, "duration_s": 0.3},
        {"pose_overrides": fwd,   "right_hand": _fist_right_hand, "duration_s": 0.4},
        {"pose_overrides": start, "right_hand": _fist_right_hand, "duration_s": 0.2},
    ]
    return {"word": "good", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_water(fps: int = 15) -> dict:
    """W-shape hand touching chin, small tap."""
    at_chin  = {14: (0.648, 0.328), 16: (0.528, 0.210)}
    tapped   = {14: (0.650, 0.336), 16: (0.532, 0.220)}
    keyframes = [
        {"pose_overrides": at_chin, "right_hand": _open_right_hand, "duration_s": 0.3},
        {"pose_overrides": tapped,  "right_hand": _open_right_hand, "duration_s": 0.2},
        {"pose_overrides": at_chin, "right_hand": _open_right_hand, "duration_s": 0.2},
        {"pose_overrides": tapped,  "right_hand": _open_right_hand, "duration_s": 0.2},
        {"pose_overrides": at_chin, "right_hand": _open_right_hand, "duration_s": 0.2},
    ]
    return {"word": "water", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_eat(fps: int = 15) -> dict:
    """Flat hand tapping toward mouth."""
    near_mouth = {14: (0.650, 0.330), 16: (0.530, 0.215)}
    mid        = {14: (0.658, 0.358), 16: (0.542, 0.245)}
    keyframes = [
        {"pose_overrides": mid,        "right_hand": _flat_right_hand, "duration_s": 0.2},
        {"pose_overrides": near_mouth, "right_hand": _flat_right_hand, "duration_s": 0.2},
        {"pose_overrides": mid,        "right_hand": _flat_right_hand, "duration_s": 0.2},
        {"pose_overrides": near_mouth, "right_hand": _flat_right_hand, "duration_s": 0.2},
        {"pose_overrides": mid,        "right_hand": _flat_right_hand, "duration_s": 0.2},
    ]
    return {"word": "eat", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_stop(fps: int = 15) -> dict:
    """Flat hand pushing forward (stop gesture)."""
    neutral = {14: (0.685, 0.450), 16: (0.700, 0.550)}
    pushed  = {14: (0.680, 0.420), 16: (0.680, 0.500)}
    keyframes = [
        {"pose_overrides": neutral, "right_hand": _flat_right_hand, "duration_s": 0.3},
        {"pose_overrides": pushed,  "right_hand": _flat_right_hand, "duration_s": 0.4},
        {"pose_overrides": neutral, "right_hand": _flat_right_hand, "duration_s": 0.3},
    ]
    return {"word": "stop", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_go(fps: int = 15) -> dict:
    """Pointing index finger moving forward (outward arc)."""
    start = {14: (0.680, 0.420), 16: (0.680, 0.510)}
    fwd   = {14: (0.672, 0.395), 16: (0.662, 0.472)}
    keyframes = [
        {"pose_overrides": start, "right_hand": _point_right_hand, "duration_s": 0.3},
        {"pose_overrides": fwd,   "right_hand": _point_right_hand, "duration_s": 0.5},
        {"pose_overrides": start, "right_hand": _point_right_hand, "duration_s": 0.2},
    ]
    return {"word": "go", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_come(fps: int = 15) -> dict:
    """Beckoning — index finger curling toward body."""
    extended  = {14: (0.672, 0.395), 16: (0.658, 0.468)}
    beckoning = {14: (0.682, 0.405), 16: (0.672, 0.488)}
    keyframes = [
        {"pose_overrides": extended,  "right_hand": _point_right_hand, "duration_s": 0.3},
        {"pose_overrides": beckoning, "right_hand": _fist_right_hand,  "duration_s": 0.3},
        {"pose_overrides": extended,  "right_hand": _point_right_hand, "duration_s": 0.3},
        {"pose_overrides": beckoning, "right_hand": _fist_right_hand,  "duration_s": 0.3},
    ]
    return {"word": "come", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_understand(fps: int = 15) -> dict:
    """Index finger tapping forehead."""
    neutral  = {14: (0.685, 0.450), 16: (0.700, 0.545)}
    at_head  = {14: (0.648, 0.280), 16: (0.548, 0.162)}
    tapped   = {14: (0.650, 0.290), 16: (0.552, 0.172)}
    keyframes = [
        {"pose_overrides": neutral, "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": at_head, "right_hand": _point_right_hand, "duration_s": 0.3},
        {"pose_overrides": tapped,  "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": at_head, "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": neutral, "right_hand": _point_right_hand, "duration_s": 0.2},
    ]
    return {"word": "understand", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_learn(fps: int = 15) -> dict:
    """Flat hand at forehead, fingers fold toward palm."""
    at_head  = {14: (0.648, 0.285), 16: (0.545, 0.165)}
    keyframes = [
        {"pose_overrides": at_head, "right_hand": _open_right_hand, "duration_s": 0.4},
        {"pose_overrides": at_head, "right_hand": _fist_right_hand, "duration_s": 0.4},
        {"pose_overrides": at_head, "right_hand": _open_right_hand, "duration_s": 0.3},
    ]
    return {"word": "learn", "fps": fps, "frames": _build_frames(keyframes, fps)}


def _sign_name(fps: int = 15) -> dict:
    """H-handshape tapping twice (first two fingers extended)."""
    start = {14: (0.682, 0.402), 16: (0.686, 0.495)}
    tap   = {14: (0.685, 0.415), 16: (0.690, 0.512)}
    keyframes = [
        {"pose_overrides": start, "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": tap,   "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": start, "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": tap,   "right_hand": _point_right_hand, "duration_s": 0.2},
        {"pose_overrides": start, "right_hand": _point_right_hand, "duration_s": 0.1},
    ]
    return {"word": "name", "fps": fps, "frames": _build_frames(keyframes, fps)}


# ---------------------------------------------------------------------------
# Generator map & entry point
# ---------------------------------------------------------------------------

_GENERATORS = {
    "hello":       _sign_hello,
    "thank_you":   _sign_thank_you,
    "yes":         _sign_yes,
    "no":          _sign_no,
    "help":        _sign_help,
    "please":      _sign_please,
    "good":        _sign_good,
    "water":       _sign_water,
    "eat":         _sign_eat,
    "stop":        _sign_stop,
    "go":          _sign_go,
    "come":        _sign_come,
    "understand":  _sign_understand,
    "learn":       _sign_learn,
    "name":        _sign_name,
}


def generate_all(output_dir: Path | None = None, fps: int = 15, verbose: bool = True) -> None:
    """Generate all sample landmark files into *output_dir*."""
    d = output_dir or LANDMARKS_DIR
    d.mkdir(parents=True, exist_ok=True)

    for word, gen_fn in _GENERATORS.items():
        data = gen_fn(fps=fps)
        out_path = d / f"{word}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        if verbose:
            print(f"  wrote {out_path.name}  ({len(data['frames'])} frames)")

    if verbose:
        print(f"\nDone — {len(_GENERATORS)} sign files in {d}")


if __name__ == "__main__":
    print("Generating sample landmark files …\n")
    generate_all(verbose=True)
