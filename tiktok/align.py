"""CTC forced alignment with wav2vec2-base-960h (ONNX): where each known word is spoken.

  python align.py hook    align the spliced hook clip and write plan/hook_words.json
"""
from __future__ import annotations

import json
import re
import shutil
import sys

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

REPO = "Xenova/wav2vec2-base-960h"
STEP = 320 / 16000
NUMBERS = {"800,000": "eight hundred thousand", "2.2": "two point two", "30": "thirty"}
# Source-time word starts measured by hand where the aligner is unstable: {clip: {word index: source s}}.
# whatis: five audio contexts gave four different alignments. Energy dips and vowel formants show
# "is" = the short burst at 427.20 (F1 ~360 Hz, an "ih"), "that?" = 427.46 after a 0.12 s pause (F1 ~740 Hz, an "a"),
# "That" = 427.98 after the t-release hiss (F1 ~750 Hz), "is" = 428.23 after the t-closure dip.
MEASURED = {"whatis": {1: 427.20, 2: 427.46, 3: 427.98, 4: 428.23}}
_model: dict = {}


def _load():
    if not _model:
        _model["vocab"] = json.loads(open(hf_hub_download(REPO, "vocab.json"), encoding="utf-8").read())
        _model["sess"] = ort.InferenceSession(hf_hub_download(REPO, "onnx/model.onnx"),
                                              providers=["CPUExecutionProvider"])
    return _model["sess"], _model["vocab"]


def to16k(x48: np.ndarray) -> np.ndarray:
    mono = x48.mean(axis=1) if x48.ndim == 2 else x48
    return resample_poly(mono.astype(np.float64), 1, 3).astype(np.float32)


def log_probs(a16: np.ndarray) -> np.ndarray:
    sess, _ = _load()
    x = (a16 - a16.mean()) / np.sqrt(a16.var() + 1e-7)
    z = sess.run(None, {sess.get_inputs()[0].name: x[None].astype(np.float32)})[0][0].astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def spell(word: str) -> str:
    """Letters the model knows, numbers spelled out, hyphens and spaces as word breaks."""
    w = NUMBERS.get(word.lower().strip(".,?!"), word)
    parts = (re.sub(r"[^A-Z']", "", s) for s in re.split(r"[\s-]+", w.upper()))
    return "|".join(p for p in parts if p)


def align(a16: np.ndarray, words: list[str]) -> list[dict]:
    """Viterbi CTC alignment of words (in order) to 16 kHz audio. Times in seconds from the audio start."""
    _, vocab = _load()
    blank, sep = vocab["<pad>"], vocab["|"]
    spelled = [spell(w) for w in words]
    labels, owner = [], []
    for i, sp in enumerate(spelled):
        if not sp:
            continue
        if labels:
            labels.append(sep)
            owner.append(-1)
        for ch in sp:
            labels.append(sep if ch == "|" else vocab[ch])
            owner.append(i)
    em = log_probs(a16)
    T, L = len(em), len(labels)
    ext = [blank]
    for lab in labels:
        ext += [lab, blank]
    S = len(ext)
    ext = np.array(ext)
    skip = np.zeros(S, bool)
    for s in range(2, S):
        skip[s] = ext[s] != blank and ext[s] != ext[s - 2]
    dp = np.full((T, S), -np.inf)
    bp = np.zeros((T, S), np.int8)
    dp[0, 0], dp[0, 1] = em[0, ext[0]], em[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        c0 = prev
        c1 = np.concatenate([[-np.inf], prev[:-1]])
        c2 = np.where(skip, np.concatenate([[-np.inf, -np.inf], prev[:-2]]), -np.inf)
        stack = np.stack([c0, c1, c2])
        bp[t] = stack.argmax(axis=0)
        dp[t] = stack.max(axis=0) + em[t, ext]
    s = S - 1 if dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    path = np.zeros(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    out = []
    for i, w in enumerate(words):
        frames = [t for t in range(T) if path[t] % 2 == 1 and owner[(path[t] - 1) // 2] == i]
        if not frames:
            out.append({"text": w, "start": None, "end": None, "score": None})
            continue
        score = float(np.mean([em[t, ext[path[t]]] for t in frames]))
        out.append({"text": w, "start": round(frames[0] * STEP, 3), "end": round((frames[-1] + 1) * STEP, 3),
                    "score": round(score, 3)})
    return out


def refine_starts(a16: np.ndarray, res: list[dict], max_back: float = 0.3, min_rise_db: float = 6.0) -> list[dict]:
    """CTC marks vowel-initial words late after a pause; move a start back to a sharp energy rise in that pause."""
    hop, win = 160, 320
    n = 1 + max(0, (len(a16) - win) // hop)
    db = np.array([10 * np.log10(np.mean(a16[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n)])
    db = np.maximum(db, db.max() - 60)
    out, prev_end = [], 0.0
    for r in res:
        r = dict(r, ctc_start=r["start"])
        peak = db[int(r["start"] / 0.01):max(int(r["end"] / 0.01), int(r["start"] / 0.01) + 1)].max()
        i0, i1 = int(max(prev_end, r["start"] - max_back) / 0.01), int(r["start"] / 0.01)
        best, best_j = 0.0, None
        for j in range(max(i0, 2), min(i1, n - 3) + 1):
            rise = db[j + 3] - db[j - 2]
            if rise > best and db[j + 3] >= peak - 15:
                best, best_j = rise, j
        if best_j is not None and best >= min_rise_db:
            r["start"] = round(best_j * 0.01, 3)
        out.append(r)
        prev_end = r["end"]
    return out


def hook() -> int:
    import mix
    import overlays
    import timeline as tl
    name, parts, _, _ = tl.VOICE[0]
    clip = mix.build_clip(mix.source_audio(), parts)
    est = dict(overlays.HOOK_EST)
    plan = tl.HERE / "plan"
    wh_path = plan / "hook_words_whisper.json"
    if not wh_path.exists() and (plan / "hook_words.json").exists():
        shutil.copy(plan / "hook_words.json", wh_path)
    whisper = json.loads(wh_path.read_text(encoding="utf-8")) if wh_path.exists() else []
    words = [w for w, _ in overlays.HOOK_EST]
    a16 = to16k(clip)
    res = refine_starts(a16, align(a16, words))
    print(f"{'word':12s} {'start':>6s} {'ctc':>6s} {'end':>6s} {'score':>6s} {'whisper':>8s} {'estimate':>8s}")
    for i, r in enumerate(res):
        ws = whisper[i]["start"] if i < len(whisper) else float("nan")
        print(f"{r['text']:12s} {r['start']:6.3f} {r['ctc_start']:6.3f} {r['end']:6.3f} {r['score']:6.2f} {ws:8.3f} {est[r['text']]:8.3f}")
    if any(r["start"] is None for r in res):
        print("some words were not aligned; not writing", file=sys.stderr)
        return 1
    length = len(clip) / mix.SR
    nxt = [r["start"] for r in res[1:]] + [length]
    out = [{"text": r["text"], "start": r["start"], "end": round(min(max(r["end"], r["start"] + 0.08), cap), 3)}
           for r, cap in zip(res, nxt)]
    (plan / "hook_words.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (plan / "hook_words_ctc.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote plan/hook_words.json from the forced alignment")
    return 0


def voice() -> int:
    """Clip-relative CTC word times for every non-hook clip, measured on the clip exactly as built.
    Starts listed in MEASURED replace the aligner's."""
    import mix
    import timeline as tl
    plan = tl.HERE / "plan"
    lines = json.loads((plan / "lines.json").read_text(encoding="utf-8"))
    src = mix.source_audio()
    out, failed = {}, []
    for name, parts, _, keys in tl.VOICE[1:]:
        words = [(k, w["text"]) for k in keys for w in lines[k]["words"]]
        a16 = to16k(mix.build_clip(src, parts))
        res = refine_starts(a16, align(a16, [w for _, w in words]))
        if any(r["start"] is None for r in res):
            failed.append(name)
            continue
        for i, st in MEASURED.get(name, {}).items():
            off = tl.clip_offset(parts, st)
            if off is None:
                raise SystemExit(f"{name}: measured start {st} is outside the clip {parts}")
            res[i] = dict(res[i], start=round(off, 3), measured=True)
        starts = [r["start"] for r in res]
        if starts != sorted(starts):
            raise SystemExit(f"{name}: word starts out of order {starts}")
        nxt = [r["start"] for r in res[1:]] + [len(a16) / 16000]
        out[name] = {"parts": [list(p) for p in parts], "words": [
            {"text": r["text"], "key": k, "start": r["start"],
             "end": round(min(max(r["end"], r["start"] + 0.08), cap), 3), "ctc_start": r["ctc_start"],
             "score": r["score"], **({"measured": True} if r.get("measured") else {})}
            for (k, _), r, cap in zip(words, res, nxt)]}
        ws = out[name]["words"]
        print(f"{name:8s} {ws[0]['start']:.2f}-{ws[-1]['end']:.2f} of {len(a16) / 16000:.2f}s  "
              + " ".join(f"{w['text']}@{w['start']:.2f}" for w in ws))
    if failed:
        print(f"words not aligned in {failed}; not writing", file=sys.stderr)
        return 1
    (plan / "voice_words.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote plan/voice_words.json")
    return 0


if __name__ == "__main__":
    cmd = {"hook": hook, "voice": voice}.get(sys.argv[1] if len(sys.argv) == 2 else "")
    sys.exit(cmd() if cmd else 2)
