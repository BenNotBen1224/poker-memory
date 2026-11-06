"""Word timings for the spliced hook clip, measured by Whisper on the exact audio the edit plays."""
from __future__ import annotations

import json
import re
import sys

import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

import mix
import timeline as tl

PAD = 0.3
EXPECT = "This is unreal that he's considering getting away from this."
OUT = tl.HERE / "plan" / "hook_words.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9']", "", s.lower())


def main() -> int:
    name, parts, _, _ = tl.VOICE[0]
    assert name == "hook"
    clip = mix.build_clip(mix.source_audio(), parts)
    a16 = resample_poly(clip.mean(axis=1).astype(np.float64), 1, 3).astype(np.float32)
    pad = np.zeros(int(PAD * 16000), np.float32)
    model = WhisperModel("medium.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(np.concatenate([pad, a16, pad]), language="en", beam_size=5, word_timestamps=True,
                               condition_on_previous_text=False, vad_filter=False)
    words = [w for s in segs for w in s.words]
    for w in words:
        print(f"{w.start - PAD:6.3f} {w.end - PAD:6.3f} p={w.probability:.2f} {w.word.strip()}")
    expect = EXPECT.split()
    if [norm(w.word) for w in words] != [norm(x) for x in expect]:
        print("MISMATCH: heard words differ from the expected line; keeping estimates", file=sys.stderr)
        return 1
    length = len(clip) / mix.SR
    out = [{"text": x, "start": round(max(0.0, w.start - PAD), 3), "end": round(min(length, w.end - PAD), 3)}
           for x, w in zip(expect, words)]
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(out)} words, clip {length:.3f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
