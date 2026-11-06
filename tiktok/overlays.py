"""Drawable overlay types and karaoke captions."""
from __future__ import annotations

import json
import math

import gfx
import timeline as tl
from gfx import WHITE, clamp01, place

DARK = (18, 18, 18)


class Spr:
    """A sprite shown from t0 to t1 with an entrance animation. cy may be replaced by a top edge."""

    def __init__(self, spr, cx, cy, t0, t1, anim="pop", alpha=1.0, top=None, fade_out=0.0):
        self.spr, self.cx, self.t0, self.t1 = spr, cx, t0, t1
        self.cy = cy if top is None else top + spr.shape[0] / 2
        self.anim, self.alpha, self.fade_out = anim, alpha, fade_out

    def draw(self, canvas, t):
        if not self.t0 <= t < self.t1:
            return
        dt = t - self.t0
        scale, alpha, dy = 1.0, self.alpha, 0.0
        if self.anim == "pop":
            scale, alpha = gfx.pop(dt), alpha * clamp01(dt / 0.06)
        elif self.anim == "slam":
            p = clamp01(dt / 0.14)
            scale, alpha = 2.2 - 1.2 * (1 - (1 - p) ** 3), alpha * clamp01(dt / 0.05)
        elif self.anim == "up":
            dy = 40 * (1 - gfx.ease_in_out(dt / 0.25))
            alpha *= clamp01(dt / 0.2)
        elif self.anim == "fade":
            alpha *= clamp01(dt / 0.2)
        if self.fade_out and self.t1 - t < self.fade_out:
            alpha *= (self.t1 - t) / self.fade_out
        place(canvas, self.spr, self.cx, self.cy + dy, scale, alpha)


class Flip:
    """A card that pops in face down, then flips to its face."""

    def __init__(self, back, front, cx, cy, t0, tflip, t1, dur=0.22):
        self.back, self.front, self.cx, self.cy = back, front, cx, cy
        self.t0, self.tflip, self.t1, self.dur = t0, tflip, t1, dur

    def draw(self, canvas, t):
        if not self.t0 <= t < self.t1:
            return
        if t < self.tflip:
            place(canvas, self.back, self.cx, self.cy, gfx.pop(t - self.t0))
            return
        p = clamp01((t - self.tflip) / self.dur)
        spr = self.back if p < 0.5 else self.front
        place(canvas, spr, self.cx, self.cy, 1.0, 1.0, sx=max(0.02, abs(math.cos(math.pi * p))))


class Hud:
    """Bottom bar: Ian's cards, the board as it is dealt, Joe's hidden cards."""
    CW = 56

    def __init__(self, t0, t1, board):
        self.t0, self.t1, self.board = t0, t1, board
        self.bar = gfx.to_sprite(gfx.rounded_rect(790, 100, 24, (12, 12, 12, 215)))
        self.slot = gfx.to_sprite(gfx.rounded_rect(self.CW, int(self.CW * 1.4), 7, (255, 255, 255, 22),
                                                   outline=(255, 255, 255, 110), ow=2))
        self.ian = gfx.text_line("ian", 34, "Black", WHITE)
        self.joe = gfx.text_line("joe", 34, "Black", WHITE)
        self.mine = [gfx.card("K", "♦", self.CW), gfx.card("K", "♠", self.CW)]
        self.back = gfx.card("", "", self.CW, back=True)

    def draw(self, canvas, t, y=1420):
        if not self.t0 <= t < self.t1:
            return
        a = clamp01((t - self.t0) / 0.15)
        place(canvas, self.bar, 490, y, 1.0, a)
        place(canvas, self.ian, 143, y, 1.0, a)
        place(canvas, self.joe, 838, y, 1.0, a)
        for x, c in zip((208, 270), self.mine):
            place(canvas, c, x, y, 1.0, a)
        for x in (710, 772):
            place(canvas, self.back, x, y, 1.0, a)
        for x, (rank, suit, tb) in zip((366, 428, 490, 552, 614), self.board):
            if t < tb:
                place(canvas, self.slot, x, y, 1.0, a)
            else:
                place(canvas, gfx.card(rank, suit, self.CW), x, y, gfx.pop(t - tb))


class ChipRow:
    """Ian's six 30-second time chips filling in as he uses them."""

    def __init__(self, t0, t1, fills, cx=540, cy=480, gap=84):
        self.t0, self.t1, self.fills, self.cx, self.cy, self.gap = t0, t1, fills, cx, cy, gap

    def draw(self, canvas, t):
        if not self.t0 <= t < self.t1:
            return
        a = clamp01((t - self.t0) / 0.2)
        n = len(self.fills)
        for i, tf in enumerate(self.fills):
            x = self.cx + (i - (n - 1) / 2) * self.gap
            if t < tf:
                place(canvas, gfx.chip(70, False), x, self.cy, 1.0, a)
            else:
                place(canvas, gfx.chip(70, True), x, self.cy, gfx.pop(t - tf), a)


class Circle:
    def __init__(self, cx, cy, rx, ry, t0, dur, t1, rot=-3):
        self.cx, self.cy, self.rx, self.ry = cx, cy, rx, ry
        self.t0, self.dur, self.t1, self.rot = t0, dur, t1, rot

    def draw(self, canvas, t):
        if self.t0 <= t < self.t1:
            gfx.scribble_circle(canvas, self.cx, self.cy, self.rx, self.ry, (t - self.t0) / self.dur, rot=self.rot)


# ---------------------------------------------------------------- captions

FUNC = {"the", "a", "an", "to", "of", "at", "in", "my", "i"}
HOOK_EST = [("This", 0.03), ("is", 0.43), ("unreal", 1.12), ("that", 1.47), ("he's", 1.65),
            ("considering", 1.82), ("getting", 2.52), ("away", 2.87), ("from", 3.22), ("this.", 3.40)]
HOOK_END = 3.57


def hook_words() -> list[dict]:
    """Clip-relative word times for the spliced hook clip (Whisper file if present)."""
    p = tl.HERE / "plan" / "hook_words.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    ends = [s for _, s in HOOK_EST[1:]] + [HOOK_END]
    return [{"text": w, "start": s, "end": e} for (w, s), e in zip(HOOK_EST, ends)]


def voice_words() -> list[dict]:
    """Every captioned word in output time, grouped by clip."""
    lines = json.loads((tl.HERE / "plan" / "lines.json").read_text(encoding="utf-8"))
    ctc_path = tl.HERE / "plan" / "voice_words.json"
    ctc = json.loads(ctc_path.read_text(encoding="utf-8")) if ctc_path.exists() else {}
    out = []
    for name, parts, at, keys in tl.VOICE:
        end_src = parts[-1][1]
        if name in ctc:
            if ctc[name]["parts"] != [list(p) for p in parts]:
                raise RuntimeError(f"plan/voice_words.json is stale for {name}: run python align.py voice")
            out += [{"text": w["text"], "start": at + w["start"], "end": at + w["end"],
                     "speaker": lines[w["key"]]["speaker"], "clip": name} for w in ctc[name]["words"]]
            continue
        for key in keys:
            if key == "@hook":
                for w in hook_words():
                    out.append({"text": w["text"], "start": at + w["start"], "end": at + w["end"],
                                "speaker": None, "clip": name})
                continue
            line = lines[key]
            for w in line["words"]:
                off = tl.clip_offset(parts, w["start"])
                if off is None:
                    continue
                o0 = at + off
                o1 = o0 + max(0.05, min(w["end"], end_src) - w["start"])
                out.append({"text": w["text"], "start": o0, "end": o1, "speaker": line["speaker"], "clip": name})
    return sorted(out, key=lambda w: w["start"])


def build_pages(words: list[dict]) -> list[dict]:
    pages, cur = [], []

    def close():
        if cur:
            pages.append(list(cur))
            cur.clear()

    for w in words:
        if cur and cur[-1]["clip"] != w["clip"]:
            close()
        trial = cur + [w]
        if cur and (len(trial) > 3 or len(" ".join(x["text"] for x in trial)) > 18):
            if len(cur) >= 2 and cur[-1]["text"].lower().strip(".,?!") in FUNC:
                moved = cur.pop()
                close()
                cur.append(moved)
            else:
                close()
        cur.append(w)
        if w["text"][-1:] in ".?!,":
            close()
    close()
    out = []
    for i, pg in enumerate(pages):
        texts = [w["text"].lower() for w in pg]
        texts[-1] = texts[-1].rstrip(".,")
        start = pg[0]["start"]
        end = pg[-1]["end"] + 0.35
        if i + 1 < len(pages):
            end = min(end, pages[i + 1][0]["start"])
        y = 1430 if tl.T("reveal") <= start < tl.T("read") else 1285
        out.append({"start": start, "end": end, "y": y, "speaker": pg[0]["speaker"],
                    "words": [(tx, w["start"]) for tx, w in zip(texts, pg)]})
    return out


class Captions:
    def __init__(self, pages):
        self.pages = pages
        self.tag = gfx.pill("ian 🎙", 32, bg=DARK, fg=WHITE)

    def draw(self, canvas, t):
        for p in self.pages:
            if not p["start"] <= t < p["end"]:
                continue
            active = max(i for i, (_, s) in enumerate(p["words"]) if s <= t + 1e-9)
            markup = " ".join(f"*{w}*" if i == active else w for i, (w, _) in enumerate(p["words"]))
            spr = gfx.stroke_text(markup, 64, "Black", 820)
            sc = 0.86 + 0.14 * gfx.ease_out_back(clamp01((t - p["start"]) / 0.12))
            place(canvas, spr, 510, p["y"], sc)
            if p["speaker"] == "Ian":
                place(canvas, self.tag, 510, p["y"] - 72)
