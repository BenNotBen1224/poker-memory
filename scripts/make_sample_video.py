"""Render the synthetic session video for examples/home_game.yaml.

Each frame shows the video clock and the hand's events at their recorded times,
so an evidence link that seeks to 01:55.4 visibly lands on "Black shirt raises to 75".
No real people or recordings are involved.

    python scripts/make_sample_video.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from poker_memory.engine import replay
from poker_memory.model import SessionDoc, VideoTime

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS, LENGTH_S = 480, 270, 10, 300


def timeline(doc: SessionDoc) -> list[tuple[int, int, str]]:
    """(ms, hand, label) for every timed moment."""
    out = []
    for record in doc.hands:
        if record.dealt_at:
            out.append((VideoTime.parse(record.dealt_at).earliest_ms, record.index, f"Hand {record.index} dealt"))
        for step in replay(record, doc.session.stakes).steps:
            if step.time:
                out.append((step.time.earliest_ms, record.index, step.label))
    return sorted(out)


def frame(ms: int, moments: list[tuple[int, int, str]], fonts) -> bytes:
    big, small = fonts
    img = Image.new("RGB", (W, H), (18, 64, 40))
    d = ImageDraw.Draw(img)
    d.text((12, 8), f"{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms // 100 % 10}", font=big, fill=(255, 255, 255))
    d.text((W - 12, 14), "SYNTHETIC - fictional players", font=small, fill=(170, 200, 180), anchor="ra")
    past = [m for m in moments if m[0] <= ms]
    if past:
        hand = past[-1][1]
        d.text((12, 56), f"Hand {hand}", font=small, fill=(250, 220, 120))
        shown = [m for m in past if m[1] == hand][-6:]
        for row, (t, _, label) in enumerate(shown):
            fresh = ms - t < 2000
            d.text((12, 86 + row * 28), f"{t // 60000:02d}:{t // 1000 % 60:02d}  {label}", font=small,
                   fill=(255, 235, 60) if fresh else (220, 230, 225))
    return img.tobytes()


def main() -> None:
    doc = SessionDoc.load(ROOT / "examples" / "home_game.yaml")
    target = ROOT / "examples" / doc.session.video
    moments = timeline(doc)
    fonts = (ImageFont.load_default(size=40), ImageFont.load_default(size=18))
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryslow", "-crf", "32",
           "-tune", "stillimage", "-g", str(FPS), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for n in range(LENGTH_S * FPS):
            proc.stdin.write(frame(n * 1000 // FPS, moments, fonts))
        proc.stdin.close()
        if proc.wait():
            sys.exit(f"ffmpeg failed with code {proc.returncode}")
    print(f"wrote {target} ({target.stat().st_size // 1024} KB, {len(moments)} timed moments)")


if __name__ == "__main__":
    main()
