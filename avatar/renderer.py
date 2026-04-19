"""
2D stick-figure sign language renderer for SunoSaathi.

Generates a self-contained HTML/JS Canvas animation from pre-extracted
MediaPipe Holistic landmark sequences. Designed to be embedded in Streamlit
via st.components.v1.html().

Usage:
    from avatar.renderer import build_sign_queue, render_avatar

    queue = build_sign_queue(["hello", "thank"], glossary, landmarks_dir)
    html  = render_avatar(queue)
    st.components.v1.html(html, height=440)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.config import GLOSSARY_PATH, LANDMARKS_DIR, SIGN_CANVAS_W, SIGN_CANVAS_H, SIGN_PAUSE_MS


# ---------------------------------------------------------------------------
# Glossary & landmark loading
# ---------------------------------------------------------------------------

def load_glossary(path: Path | None = None) -> dict[str, str]:
    """Load the keyword → filename mapping from sign_glossary.json."""
    p = path or GLOSSARY_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    # Strip _comment key
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_landmark_file(filename: str, landmarks_dir: Path | None = None) -> dict | None:
    """Load a landmark JSON file.  Returns None if the file doesn't exist."""
    d = landmarks_dir or LANDMARKS_DIR
    path = d / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_sign_queue(
    keywords: list[str],
    glossary: dict[str, str] | None = None,
    landmarks_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Map *keywords* to sign animations, returning an ordered queue.

    Each entry in the queue is one of:
      {"word": str, "fps": int, "frames": [...]}  — sign found
      {"word": str, "fps": 0, "frames": []}        — no sign, display word as text

    Parameters
    ----------
    keywords      : content words from caption simplification
    glossary      : loaded glossary dict (loaded if None)
    landmarks_dir : directory with landmark .json files (uses config default if None)
    """
    if glossary is None:
        glossary = load_glossary()

    queue: list[dict] = []
    seen: set[str] = set()

    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in seen:
            continue
        seen.add(kw_lower)

        filename = glossary.get(kw_lower)
        if filename:
            data = load_landmark_file(filename, landmarks_dir)
            if data and data.get("frames"):
                queue.append({
                    "word":   kw_lower,
                    "fps":    data.get("fps", 15),
                    "frames": data["frames"],
                })
                continue

        # No sign found — placeholder entry (renderer shows word as text)
        queue.append({"word": kw_lower, "fps": 0, "frames": []})

    return queue


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{
  background:#0a0a0a;
  display:flex;flex-direction:column;align-items:center;
  padding:10px;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}}
canvas{{border-radius:10px;background:#1c1c1e;display:block}}
#word-label{{
  color:#f5f5f7;font-size:15px;font-weight:600;
  margin-top:9px;min-height:22px;letter-spacing:.03em;
  text-transform:uppercase;text-align:center;
}}
#sub{{
  color:#6e6e73;font-size:11px;margin-top:3px;text-align:center;min-height:16px
}}
</style>
</head>
<body>
<canvas id="c" width="{W}" height="{H}"></canvas>
<div id="word-label"></div>
<div id="sub"></div>
<script>
// ─── Injected data ────────────────────────────────────────────────────────
const QUEUE   = {QUEUE_JSON};
const PAUSE   = {PAUSE_MS};

// ─── MediaPipe skeleton topology ─────────────────────────────────────────
const POSE_PAIRS=[
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28]
];
const HAND_PAIRS=[
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

// ─── Canvas setup ─────────────────────────────────────────────────────────
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const W=canvas.width,H=canvas.height;

function pt(lm){{return{{x:lm[0]*W,y:lm[1]*H}};}}

function segs(lms,pairs,col,lw){{
  if(!lms||!lms.length)return;
  ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.lineCap='round';
  pairs.forEach(([a,b])=>{{
    if(!lms[a]||!lms[b])return;
    const A=pt(lms[a]),B=pt(lms[b]);
    ctx.beginPath();ctx.moveTo(A.x,A.y);ctx.lineTo(B.x,B.y);ctx.stroke();
  }});
}}

function dots(lms,col,r){{
  if(!lms||!lms.length)return;
  ctx.fillStyle=col;
  lms.forEach(lm=>{{
    if(!lm)return;
    const p=pt(lm);
    ctx.beginPath();ctx.arc(p.x,p.y,r,0,2*Math.PI);ctx.fill();
  }});
}}

function drawHead(pose){{
  if(!pose||!pose[0])return;
  const p=pt(pose[0]);
  ctx.strokeStyle='#c8c8c8';ctx.lineWidth=2;
  ctx.beginPath();ctx.arc(p.x,p.y,18,0,2*Math.PI);ctx.stroke();
  [1,4].forEach(i=>{{
    if(!pose[i])return;
    const e=pt(pose[i]);
    ctx.fillStyle='#c8c8c8';
    ctx.beginPath();ctx.arc(e.x,e.y,2.5,0,2*Math.PI);ctx.fill();
  }});
}}

function drawFrame(frame){{
  ctx.clearRect(0,0,W,H);
  const pose=frame.pose, lh=frame.left_hand, rh=frame.right_hand;
  segs(pose,POSE_PAIRS,'#666',3);
  drawHead(pose);
  segs(rh,HAND_PAIRS,'#4fc3f7',1.5);
  segs(lh,HAND_PAIRS,'#f48fb1',1.5);
  dots(pose,'#a0a0a0',3.5);
  if(rh)dots(rh,'#4fc3f7',2.5);
  if(lh)dots(lh,'#f48fb1',2.5);
}}

function noSign(word){{
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#2c2c2e';
  ctx.fillRect(0,0,W,H);
  ctx.fillStyle='#6e6e73';
  ctx.font='13px -apple-system';
  ctx.textAlign='center';
  ctx.fillText(word,W/2,H/2-8);
  ctx.fillStyle='#48484a';
  ctx.font='11px -apple-system';
  ctx.fillText('no sign available',W/2,H/2+10);
}}

// ─── Animation loop ───────────────────────────────────────────────────────
if(!QUEUE||QUEUE.length===0){{
  ctx.fillStyle='#2c2c2e';ctx.fillRect(0,0,W,H);
  ctx.fillStyle='#6e6e73';ctx.font='13px sans-serif';
  ctx.textAlign='center';
  ctx.fillText('No keywords detected',W/2,H/2);
}}else{{
  let qi=0,fi=0;
  function playQueue(){{
    if(qi>=QUEUE.length)qi=0;
    const sign=QUEUE[qi];
    document.getElementById('word-label').textContent=sign.word.toUpperCase();
    document.getElementById('sub').textContent=
      'Sign '+(qi+1)+' of '+QUEUE.length;
    if(!sign.frames||sign.frames.length===0){{
      noSign(sign.word);
      qi++;fi=0;
      setTimeout(playQueue,PAUSE);
      return;
    }}
    const interval=1000/(sign.fps||15);
    function tick(){{
      if(fi<sign.frames.length){{
        drawFrame(sign.frames[fi++]);
        setTimeout(()=>requestAnimationFrame(tick),interval);
      }}else{{
        fi=0;qi++;
        setTimeout(playQueue,PAUSE);
      }}
    }}
    requestAnimationFrame(tick);
  }}
  playQueue();
}}
</script>
</body>
</html>
"""


def render_avatar(
    sign_queue: list[dict],
    width: int = SIGN_CANVAS_W,
    height: int = SIGN_CANVAS_H,
    pause_ms: int = SIGN_PAUSE_MS,
) -> str:
    """Return a self-contained HTML string for the sign-language avatar.

    Pass the result to ``st.components.v1.html(html, height=height+80)``.
    """
    queue_json = json.dumps(sign_queue, separators=(",", ":"))
    return _HTML_TEMPLATE.format(
        W=width,
        H=height,
        QUEUE_JSON=queue_json,
        PAUSE_MS=pause_ms,
    )


def render_setup_prompt() -> str:
    """Placeholder HTML shown before any landmark data has been generated."""
    return f"""\
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{{margin:0;background:#0a0a0a;display:flex;align-items:center;
     justify-content:center;height:{SIGN_CANVAS_H + 80}px;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
.box{{background:#1c1c1e;border:1px solid #2c2c2e;border-radius:12px;
      padding:20px 24px;max-width:240px;text-align:center}}
p{{color:#6e6e73;font-size:12px;line-height:1.6;margin:0}}
code{{color:#0a84ff;font-size:11px}}
</style></head><body>
<div class="box">
  <p>No landmark data found.<br><br>
  Run once to generate sample signs:<br>
  <code>python avatar/generate_sample_landmarks.py</code></p>
</div>
</body></html>
"""
