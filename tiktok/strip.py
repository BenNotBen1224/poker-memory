import sys
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parents[1] / "examples" / "real" / "wpt_rolling_thunder_fold.webm"
OUT = Path(__file__).resolve().parent / "plan"
FONT = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 18)
TW, TH, COLS = 320, 180, 6


def main(name: str, start: float, end: float, step: float) -> None:
    thumbs, next_t = [], start
    with av.open(str(SRC)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, start - 1.0) / s.time_base), stream=s, backward=True)
        for frame in c.decode(s):
            t = float(frame.pts * s.time_base)
            if t > end:
                break
            if t + 1e-6 >= next_t:
                thumbs.append((t, frame.to_image().resize((TW, TH), Image.BILINEAR)))
                next_t += step
    rows = (len(thumbs) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TW, rows * (TH + 24)), "black")
    d = ImageDraw.Draw(sheet)
    for k, (t, img) in enumerate(thumbs):
        x, y = (k % COLS) * TW, (k // COLS) * (TH + 24)
        sheet.paste(img, (x, y + 24))
        d.text((x + 6, y + 2), f"{int(t // 60)}:{t % 60:05.2f}", fill="yellow", font=FONT)
    sheet.save(OUT / f"{name}.jpg", quality=80)
    print(name, len(thumbs))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
