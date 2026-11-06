"""Pick clean cut points for every voice clip and check what a listener (Whisper) hears.

For each clip: transcribe the source around it (large-v3, word times), take the words just
before and after the captioned line, force-align all of them with CTC, and cut in the quietest
spot of each gap.  Then transcribe the old and the proposed clip (medium.en, clean voice only)
and compare with the caption.  Writes out/edges.json; changes nothing else.
"""
from __future__ import annotations

import difflib
import json
import re
import sys

import numpy as np
from faster_whisper import WhisperModel

import align
import mix
import timeline as tl

SR = 48000
PAD = 1.2


def toks(words: list[str]) -> list[tuple[str, int]]:
    out = []
    for i, w in enumerate(words):
        s = align.NUMBERS.get(w.lower().strip(".,?!"), w).lower().replace("-", " ")
        out += [(t, i) for t in re.sub(r"[^a-z' ]", " ", s).split() if t.strip("'")]
    return out


def ratio(caption: list[str], text: str) -> float:
    a, b = [t for t, _ in toks(caption)], [t for t, _ in toks(text.split())]
    return round(difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio(), 2)


def hear(model, a16: np.ndarray, words: bool):
    segs, _ = model.transcribe(a16, language="en", beam_size=5, word_timestamps=words,
                               condition_on_previous_text=False, vad_filter=False)
    segs = list(segs)
    if words:
        return [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
    return " ".join(s.text.strip() for s in segs)


def envelope(mono: np.ndarray, t0: float, t1: float, hop: float = 0.01, win: float = 0.03):
    ts = np.arange(t0, t1, hop)
    e = np.array([10 * np.log10(np.mean(mono[int((t - win / 2) * SR):int((t + win / 2) * SR)] ** 2) + 1e-12)
                  for t in ts])
    return ts, e


def quiet(ts, e, lo, hi, last: bool) -> float:
    """Earliest (or latest) point in [lo, hi] within 3 dB of the quietest point there."""
    m = (ts >= lo - 1e-9) & (ts <= hi + 1e-9)
    if not m.any():
        return hi if last else lo
    idx = np.where(e[m] <= e[m].min() + 3.0)[0]
    return float(ts[m][idx[-1] if last else idx[0]])


def neighbours(heard, caption):
    """Words Whisper heard just before and just after the captioned line."""
    ht, ct = toks([h[0] for h in heard]), toks(caption)
    sm = difflib.SequenceMatcher(a=[t for t, _ in ct], b=[t for t, _ in ht], autojunk=False)
    blocks = [bl for bl in sm.get_matching_blocks() if bl.size]
    if not blocks:
        return [], []
    first = ht[blocks[0].b][1] - blocks[0].a
    last = ht[blocks[-1].b + blocks[-1].size - 1][1] + (len(ct) - blocks[-1].a - blocks[-1].size)
    return heard[max(0, first - 3):max(0, first)], heard[last + 1:last + 4]


def main() -> int:
    lines = json.loads((tl.HERE / "plan" / "lines.json").read_text(encoding="utf-8"))
    src = mix.source_audio()
    mono = src.mean(axis=1)
    big = WhisperModel("large-v3", device="cpu", compute_type="int8", local_files_only=True)
    med = WhisperModel("medium.en", device="cpu", compute_type="int8", local_files_only=True)
    pad = np.zeros(int(0.4 * 16000), np.float32)
    rows = []
    for name, parts, at, keys in tl.VOICE[1:]:
        a, b = parts[0][0], parts[-1][1]
        caption = [w["text"] for k in keys for w in lines[k]["words"]]
        w0 = a - PAD
        a16 = align.to16k(src[int(w0 * SR):int((b + PAD) * SR)])
        heard = hear(big, a16, words=True)
        prev, nxt = neighbours(heard, caption)
        words = [h[0] for h in prev] + caption + [h[0] for h in nxt]
        res = align.refine_starts(a16, align.align(a16, words))
        p_res, c_res, n_res = res[:len(prev)], res[len(prev):len(prev) + len(caption)], res[len(prev) + len(caption):]
        first = next(r["start"] for r in c_res if r["start"] is not None) + w0
        last = [r["end"] for r in c_res if r["end"] is not None][-1] + w0
        prev_end = max([r["end"] for r in p_res if r["end"] is not None], default=None)
        next_start = min([r["start"] for r in n_res if r["start"] is not None], default=None)
        prev_end = None if prev_end is None else prev_end + w0
        next_start = None if next_start is None else next_start + w0
        ts, e = envelope(mono, w0, b + PAD)
        hi = min(next_start - 0.02 if next_start is not None else last + 0.45, last + 0.45)
        lo = min(last + 0.02, hi)
        b_new = round(min(quiet(ts, e, lo, hi, last=False) + 0.04, hi + 0.02), 2)
        lo2 = max(prev_end + 0.02 if prev_end is not None else first - 0.45, first - 0.45)
        hi2 = max(first - 0.01, lo2)
        a_new = round(max(quiet(ts, e, lo2, hi2, last=True) - 0.01, lo2 - 0.02), 2)
        old = align.to16k(mix.build_clip(src, [(a, b)]))
        new = align.to16k(mix.build_clip(src, [(a_new, b_new)]))
        h_old = hear(med, np.concatenate([pad, old, pad]), words=False)
        h_new = hear(med, np.concatenate([pad, new, pad]), words=False)
        row = {"clip": name, "old": [a, b], "new": [a_new, b_new],
               "first_word_start": round(first, 3), "last_word_end": round(last, 3),
               "prev_end": None if prev_end is None else round(prev_end, 3),
               "next_start": None if next_start is None else round(next_start, 3),
               "prev_words": [h[0] for h in prev], "next_words": [h[0] for h in nxt],
               "large_v3_window": " ".join(f"{h[0]}@{h[1] + w0:.2f}" for h in heard),
               "caption": " ".join(caption), "heard_old": h_old, "heard_new": h_new,
               "match_old": ratio(caption, h_old), "match_new": ratio(caption, h_new)}
        rows.append(row)
        print(f"{name:8s} old {a:.2f}-{b:.2f}  new {a_new:.2f}-{b_new:.2f}  words {first:.2f}-{last:.2f}"
              f"  prev_end {row['prev_end']}  next_start {row['next_start']}", flush=True)
        print(f"   prev {row['prev_words']}  next {row['next_words']}")
        print(f"   old [{row['match_old']:.2f}] {h_old}")
        print(f"   new [{row['match_new']:.2f}] {h_new}", flush=True)
    (tl.HERE / "out" / "edges.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
