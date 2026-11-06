import sys
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parents[1] / "examples" / "real" / "wpt_rolling_thunder_fold.webm"
OUT = Path(__file__).resolve().parent / "plan"
FONT = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 14)
BIG = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 22)


def frame_at(container, stream, t: float) -> Image.Image:
    container.seek(int(t / stream.time_base), stream=stream, backward=True)
    for frame in container.decode(stream):
        if float(frame.pts * stream.time_base) + 1e-6 >= t:
            return frame.to_image()
    raise RuntimeError(t)


def gridded(img: Image.Image, label: str) -> Image.Image:
    small = img.resize((640, 360), Image.BILINEAR)
    d = ImageDraw.Draw(small)
    for x in range(0, 1280, 80):
        d.line([(x / 2, 0), (x / 2, 360)], fill=(255, 255, 0) if x % 320 == 0 else (120, 120, 0), width=1)
        d.text((x / 2 + 2, 2), str(x), fill="yellow", font=FONT)
    for y in range(0, 720, 80):
        d.line([(0, y / 2), (640, y / 2)], fill=(0, 255, 255) if y % 320 == 0 else (0, 120, 120), width=1)
        d.text((2, y / 2 + 2), str(y), fill="cyan", font=FONT)
    d.rectangle([(400, 330), (640, 360)], fill="black")
    d.text((408, 332), label, fill="white", font=BIG)
    return small


def main(name: str, times: list[float]) -> None:
    tiles = []
    with av.open(str(SRC)) as c:
        s = c.streams.video[0]
        for t in times:
            tiles.append(gridded(frame_at(c, s, t), f"{int(t // 60):02d}:{t % 60:05.2f}"))
    sheet = Image.new("RGB", (1280, 360 * ((len(tiles) + 1) // 2)), "black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 640, (i // 2) * 360))
    sheet.save(OUT / f"{name}.jpg", quality=85)
    print(name, len(tiles))


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2:]])
