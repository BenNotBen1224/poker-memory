"""Download the real WPT clip (CC BY 3.0) next to its session file and check it is the exact file the hand was read from.

    python scripts/fetch_real_video.py

Safe to re-run: a file that already matches is left alone.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://upload.wikimedia.org/wikipedia/commons/f/fd/Incredibly_Unbelievable_Fold_on_the_World_Poker_Tour.webm"
SIZE = 35_378_861
SHA1 = "9e4f94ad95e0ffc06a5960a03c60c7b381c0c1b6"
DEST = Path(__file__).resolve().parents[1] / "examples" / "real" / "wpt_rolling_thunder_fold.webm"


def digest(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if DEST.exists() and DEST.stat().st_size == SIZE and digest(DEST) == SHA1:
        print(f"already here and verified: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "poker-memory-demo/0.1 (private research)"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    if tmp.stat().st_size != SIZE or digest(tmp) != SHA1:
        tmp.unlink()
        print("download does not match the expected size/SHA-1; not keeping it", file=sys.stderr)
        return 1
    tmp.replace(DEST)
    print(f"downloaded and verified: {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
