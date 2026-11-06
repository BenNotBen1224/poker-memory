import json
from pathlib import Path

import av
import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "examples" / "real" / "wpt_rolling_thunder_fold.webm"
OUT = Path(__file__).resolve().parent / "plan"


def main() -> None:
    prev = None
    rows = []
    with av.open(str(SRC)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for frame in c.decode(s):
            t = float(frame.pts * s.time_base)
            g = cv2.resize(frame.to_ndarray(format="gray"), (128, 72), interpolation=cv2.INTER_AREA).astype(np.int16)
            g[40:72, 0:44] = 0  # ignore the lower-left broadcast graphics
            if prev is not None:
                rows.append((t, float(np.abs(g - prev).mean())))
            prev = g
    diffs = np.array([d for _, d in rows])
    cuts = []
    for i, (t, d) in enumerate(rows):
        lo, hi = max(0, i - 6), min(len(rows), i + 7)
        neighborhood = np.median(np.r_[diffs[lo:i], diffs[i + 1:hi]])
        if d > 18 and d > 3.5 * max(neighborhood, 1.0):
            cuts.append({"t": round(t, 3), "diff": round(d, 1)})
    (OUT / "cuts.json").write_text(json.dumps(cuts, indent=1), encoding="utf-8")
    print("cuts", len(cuts))
    print(" ".join(f"{int(c['t'] // 60)}:{c['t'] % 60:05.2f}" for c in cuts))


if __name__ == "__main__":
    main()
