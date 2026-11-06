from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parents[1] / "examples" / "real" / "wpt_rolling_thunder_fold.webm"
OUT = Path(__file__).resolve().parent / "plan"
STEP = 4.0
TW, TH = 320, 180
COLS, ROWS = 5, 6
FONT = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 18)


def main() -> None:
    thumbs = []
    next_t = 0.0
    with av.open(str(SRC)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in c.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t + 1e-6 < next_t:
                continue
            img = frame.to_image().resize((TW, TH), Image.BILINEAR)
            thumbs.append((t, img))
            next_t += STEP
    per = COLS * ROWS
    for sheet_i in range(0, len(thumbs), per):
        sheet = Image.new("RGB", (COLS * TW, ROWS * (TH + 24)), "black")
        draw = ImageDraw.Draw(sheet)
        for k, (t, img) in enumerate(thumbs[sheet_i:sheet_i + per]):
            x, y = (k % COLS) * TW, (k // COLS) * (TH + 24)
            sheet.paste(img, (x, y + 24))
            draw.text((x + 6, y + 2), f"{int(t // 60):02d}:{t % 60:05.2f}", fill="yellow", font=FONT)
        sheet.save(OUT / f"sheet-{sheet_i // per:02d}.jpg", quality=82)
    print("thumbs", len(thumbs), "sheets", (len(thumbs) + per - 1) // per)


if __name__ == "__main__":
    main()
