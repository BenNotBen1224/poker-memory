"""One frame per event of a session file, taken at the second the app's video buttons jump to.

    python scripts/event_frames.py examples/real/wpt_rolling_thunder.yaml --out events
    python scripts/event_frames.py examples/real/wpt_rolling_thunder.yaml --offset 1 --out events_plus1

Look at each picture. It should show the action itself (chips crossing the line) or sit a second or two
before the action label appears. --offset looks a little later. The overlay's pot figure can lag by several
seconds on a big bet, so judge the chips and the action label, not the pot.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from frames import clock, label
from poker_memory.engine import replay
from poker_memory.model import SessionDoc


def frame_at(video: Path, seconds: float, width: int, out: Path) -> Image.Image:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{seconds}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={width}:-1", str(out)],
        check=True,
    )
    return Image.open(out).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, help="session YAML with its video next to it")
    parser.add_argument("--hand", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--offset", type=float, default=0.0, help="seconds after each event's stated time")
    parser.add_argument("--out", type=Path, required=True, help="folder for the frames")
    args = parser.parse_args()

    doc = SessionDoc.load(args.session)
    if not doc.session.video:
        raise SystemExit("this session has no video")
    video = args.session.parent / doc.session.video
    hand = next((h for h in doc.hands if h.index == args.hand), None)
    if hand is None:
        raise SystemExit(f"no hand {args.hand} in {args.session}")
    result = replay(hand, doc.session.stakes)
    if result.status != "verified":
        raise SystemExit("hand does not replay: " + "; ".join(i.message for i in result.issues))

    args.out.mkdir(parents=True, exist_ok=True)
    for n, step in enumerate((s for s in result.steps if s.time), start=1):
        seconds = step.time.earliest_ms / 1000 + args.offset
        path = args.out / f"{n:02d}_{int(seconds) // 60:02d}-{int(seconds) % 60:02d}.png"
        frame = label(frame_at(video, seconds, args.width, path), f"{clock(seconds)} {step.label}")
        frame.save(path)
        print(f"{n:02d}  {clock(seconds)}  {step.label}  -> {path}")


if __name__ == "__main__":
    main()
