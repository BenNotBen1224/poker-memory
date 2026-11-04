"""Pull timestamped frames out of a real recording so a person can read the hand by eye.

Contact sheet of every 10 s in the first two minutes:
    python scripts/frames.py VIDEO --start 0 --end 120 --step 10 --cols 3 --width 480 --out sheet.png

One full-size frame at 95.5 s:
    python scripts/frames.py VIDEO --start 95.5 --end 95.5 --width 1280 --out frame.png

Zoom on the bottom-left scoreboard of a 1280x720 video, one frame per second:
    python scripts/frames.py VIDEO --start 30 --end 45 --step 1 --crop 0,470,420,250 --width 420 --out zoom.png
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def clock(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


def extract(video: Path, seconds: float, width: int, out: Path, crop: str | None = None) -> bool:
    """Write the frame shown at `seconds` to `out`. False if the video ends before that."""
    box = ""
    if crop:
        x, y, w, h = (int(n) for n in crop.split(","))
        box = f"crop={w}:{h}:{x}:{y},"
    out.unlink(missing_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", f"{seconds}", "-i", str(video),
            "-frames:v", "1", "-vf", f"{box}scale={width}:-1", str(out),
        ],
        check=True,
    )
    return out.exists()


def grab(video: Path, start: float, end: float, step: float, width: int, crop: str | None,
         folder: Path) -> list[tuple[float, Path]]:
    found = []
    for n in range(int((end - start) / step + 1e-6) + 1):
        at = start + n * step
        path = folder / f"f_{n:04d}.png"
        if extract(video, at, width, path, crop):
            found.append((at, path))
    return found


def label(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arialbd.ttf", max(16, image.width // 22))
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((6, 4), text, font=font)
    draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 235, 59), font=font)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--end", type=float, required=True, help="seconds")
    parser.add_argument("--step", type=float, default=10.0, help="seconds between frames (must be above 0)")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--width", type=int, default=480, help="pixel width of each frame")
    parser.add_argument("--crop", help="x,y,w,h in source pixels, applied before scaling")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("--step must be above 0")

    with tempfile.TemporaryDirectory() as tmp:
        found = grab(args.video, args.start, args.end, args.step, args.width, args.crop, Path(tmp))
        tiles = [label(Image.open(path).convert("RGB"), clock(at)) for at, path in found]
    if not tiles:
        raise SystemExit("no frames extracted; check --start/--end against the video length")

    cols = min(args.cols, len(tiles))
    rows = -(-len(tiles) // cols)
    tile_w, tile_h = tiles[0].size
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"{len(tiles)} frames -> {args.out}")


if __name__ == "__main__":
    main()
