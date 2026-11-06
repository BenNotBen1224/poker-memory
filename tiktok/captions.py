"""Align agreed caption text to model word timestamps.

Each line is shown only if at least two transcription runs agree on its words
(checked by hand against plan/transcript-*.txt and plan/windows*.json).
"""
import difflib
import json
import re
from pathlib import Path

PLAN = Path(__file__).resolve().parent / "plan"

LINES = {
    "unreal": ("This is unreal that he's considering getting away from this.", (327.5, 333.3), None),
    "cowboys": ("Couple of cowboys here for Ian.", (23.2, 25.7), None),
    "atleast": ("Joe is going to at least call.", (46.6, 48.9), None),
    "farbest": ("Ian's hand far and away the best still.", (66.4, 69.0), None),
    "hate": ("You just hate to lay down kings.", (125.2, 127.8), None),
    "bet800": ("800,000 is the bet.", (196.6, 198.75), None),
    "center": ("2.2 million already in the center.", (198.75, 201.0), None),
    "wow": ("Yes, wow. Yeah, wow indeed.", (232.3, 236.2), None),
    "stacks": ("Two biggest stacks at the table.", (236.2, 239.0), None),
    "considering": ("Wow, he's actually considering this.", (243.3, 245.9), None),
    "highfive": ("I would have high-fived people and called already.", (246.9, 250.1), None),
    "burning": ("He's burning through those time extensions.", (272.8, 275.2), None),
    "noway": ("There's no way he gets away from this, right?", (282.1, 284.3), None),
    "howbeat": ("How can he think he's beat here?", (297.0, 299.3), None),
    "thirty": ("I guess I can't wait 30 minutes to find out before I make a decision", (370.7, 375.3), "Ian"),
    "killed": ("I'm just going to keep using it until my hand's killed", (399.6, 402.0), "Ian"),
    "six": ("Six time extensions.", (412.0, 415.4), None),
    "beating": ("Why is he beating himself up? He folded.", (422.7, 425.4), None),
    "whatis": ("What is that? That is unbelievable.", (426.0, 429.4), None),
    "camo": ("Yeah, Joe's hand was perfectly camouflaged.", (439.9, 443.55), None),
    "figure": ("How did he figure that out?", (443.45, 444.8), None),
    "icant": ("I can't. I can't.", (446.0, 447.4), None),
}

SOURCES = [
    ("large-noprompt", "windows2-large-noprompt.json"),
    ("large-v3", "windows-large-v3.json"),
    ("medium.en", "transcript-medium.en.json"),
    ("small.en", "transcript-small.en.json"),
]


def norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def model_words(fname: str) -> list[dict]:
    data = json.loads((PLAN / fname).read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    out = []
    for s in segs:
        for w in s.get("words") or []:
            for piece in w["word"].strip().split():
                if out and piece[:1] in ",.-" and piece[1:2].isalnum():
                    out[-1]["w"] += piece
                    out[-1]["n"] = norm(out[-1]["w"])
                    out[-1]["end"] = w["end"]
                    continue
                out.append({"start": w["start"], "end": w["end"], "w": piece, "n": norm(piece)})
    return out


def align(text: str, words: list[dict]) -> tuple[float, list[dict]]:
    canon = text.split()
    a = [norm(t) for t in canon]
    b = [w["n"] for w in words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    timed: list[dict | None] = [None] * len(canon)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                timed[i1 + k] = {"start": words[j1 + k]["start"], "end": words[j1 + k]["end"]}
        elif op == "replace":
            t0, t1 = words[j1]["start"], words[j2 - 1]["end"]
            n = i2 - i1
            for k in range(n):
                timed[i1 + k] = {"start": t0 + (t1 - t0) * k / n, "end": t0 + (t1 - t0) * (k + 1) / n}
    known = [k for k, v in enumerate(timed) if v]
    for k, v in enumerate(timed):
        if v:
            continue
        prev = max([j for j in known if j < k], default=None)
        nxt = min([j for j in known if j > k], default=None)
        t0 = timed[prev]["end"] if prev is not None else (timed[nxt]["start"] - 0.3)
        t1 = timed[nxt]["start"] if nxt is not None else t0 + 0.3
        timed[k] = {"start": t0, "end": max(t1, t0 + 0.05)}
    ratio = sm.ratio()
    return ratio, [{"text": c, "start": round(t["start"], 3), "end": round(t["end"], 3)} for c, t in zip(canon, timed)]


def build() -> dict:
    pools = {name: model_words(f) for name, f in SOURCES if (PLAN / f).exists()}
    result = {}
    for key, (text, (lo, hi), speaker) in LINES.items():
        best = None
        for name, _ in SOURCES:
            ws = [w for w in pools.get(name, []) if lo <= w["start"] < hi]
            if not ws:
                continue
            ratio, timed = align(text, ws)
            if best is None or ratio > best[0] + 1e-9:
                best = (ratio, name, timed)
        ratio, name, timed = best
        result[key] = {"text": text, "speaker": speaker, "model": name, "match": round(ratio, 3),
                       "start": timed[0]["start"], "end": timed[-1]["end"], "words": timed}
    return result


if __name__ == "__main__":
    lines = build()
    (PLAN / "lines.json").write_text(json.dumps(lines, indent=1), encoding="utf-8")
    for k, v in lines.items():
        print(f"{k:9s} {v['start']:7.2f}-{v['end']:7.2f} match={v['match']:.2f} [{v['model']}] {v['text']}")
