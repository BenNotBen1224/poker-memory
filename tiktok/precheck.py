"""Audio-only dry run of verify.py, before spending ~6 min on a full render.

Builds the mix in memory and runs verify's own checks that depend only on audio:
voice-clip overlaps, reveal crossing, spoiler words, caption timing, loudness.
Also asks Whisper large-v3 what it hears in the clips medium.en tends to mishear
(e.g. whether "he folded" survives the fold boom).

  python precheck.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import numpy as np
from scipy.signal import resample_poly

import mix
import scenes
import timeline as tl
import verify


LISTEN = ("tank_30", "beating", "icant")


def listen(a16: np.ndarray, spans: dict) -> dict:
    from faster_whisper import WhisperModel
    m = WhisperModel("large-v3", device="cpu", compute_type="int8", local_files_only=True)
    out = {}
    for name in LISTEN:
        t0, t1 = spans[name]
        segs, _ = m.transcribe(a16[int(t0 * 16000):int(t1 * 16000)], language="en", word_timestamps=True,
                               beam_size=5, condition_on_previous_text=False, vad_filter=False)
        out[name] = " ".join(f"{w.word.strip()}({w.probability:.2f})" for s in segs for w in s.words)
    return out


def main() -> int:
    spans = verify.clip_spans()
    order = sorted(spans.items(), key=lambda kv: kv[1][0])
    overlaps = [(a, b, round(sa[1] - sb[0], 3)) for (a, sa), (b, sb) in zip(order, order[1:]) if sa[1] > sb[0]]
    crossing = [n for n, (a, b) in spans.items() if a < verify.T_REVEAL < b - 0.01]
    src = verify.source_scan(spans)

    ref, stats = mix.mix(scenes.build().cues)
    ref = ref[:stats["end_sample"]]
    loud = verify.loudness(ref)
    a16 = resample_poly(ref.mean(axis=1).astype(np.float64), 1, 3).astype(np.float32)
    heard = verify.whisper(a16)
    pre = [w for w in heard if w["t"] < verify.T_REVEAL]
    mix_hits, name_hits = verify.phrase_hits(pre, verify.BANNED), verify.phrase_hits(heard, verify.NAMES)
    caps = verify.caption_checks(a16, heard, spans)
    late = [r["clip"] for r in caps if r["median_offset_s"] is None or abs(r["median_offset_s"]) > 0.2]

    checks = [("no voice clips overlap", not overlaps, str(overlaps or "none")),
              ("no voice clip crosses the reveal", not crossing, str(crossing or "none")),
              ("no spoiler words in source clips", not src["hits"], str(src["hits"] or "none")),
              ("no spoiler words heard before reveal", not mix_hits, str(mix_hits or "none")),
              ("no commentator names heard", not name_hits, str(name_hits or "none")),
              ("loudness", -15 <= loud["lufs"] <= -13 and loud["true_peak_dbtp"] <= -1.0, str(loud)),
              ("captions on time", not late, str(late or "all within 0.2s"))]
    for n, ok, d in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {n}: {d}")
    print("\ncaptions (clip, start, median offset s, ctc-only, max |offset|, text match):")
    for r in caps:
        print(f"  {r['clip']:9s} {r['start']:6.2f}  {r['median_offset_s']!s:>7s} {r['median_offset_ctc_s']!s:>7s} "
              f"{r['max_abs_offset_s']!s:>6s} {r['text_match']:5.2f}  heard: {r['heard']}")
    print("\nlarge-v3, clip by clip:")
    for name, text in listen(a16, spans).items():
        print(f"  {name:9s} {text}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
