"""Edit list for the TikTok cut: segments, timing helpers, voice clips, and per-frame video."""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

import gfx

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "examples" / "real" / "wpt_rolling_thunder_fold.webm"
SRC_SHA256 = "5d0d3f43e883affe89413562a60a5f11ec7aa69afc295defd211db25f51b4f79"
FPS = Fraction(24000, 1001)
W, H = 1080, 1920
BOX, BOX_Y = 1080, 400
SW, SH = 1280, 720
SR = 48000


@dataclass
class Seg:
    name: str
    src: float
    dur: float
    crop: tuple
    crop_end: tuple | None = None
    speed: float = 1.0
    freeze: bool = False
    mask: str = "default"
    ease_dur: float | None = None
    hold_after: float | None = None
    darken: float = 1.0
    blur: float = 0.0
    rgb_split: bool = False
    f0: int = 0
    nf: int = 0

    @property
    def start(self) -> float:
        return float(self.f0 / FPS)


SEGS = [
    Seg("hook", 338.10, 4.4, (520, 300, 600), (520, 290, 540)),
    Seg("rewind", 342.50, 0.5, (520, 290, 540), speed=-8.8, rgb_split=True),
    Seg("ian", 23.40, 3.3, (580, 300, 640), (580, 295, 600)),
    Seg("joe", 46.50, 2.6, (1000, 280, 560), (995, 280, 520)),
    Seg("flop", 58.20, 3.0, (686, 440, 560), (686, 450, 540), mask="table"),
    Seg("turn", 91.45, 2.35, (686, 440, 560), (690, 450, 540), mask="table"),
    Seg("river", 153.40, 3.2, (686, 440, 560), (740, 470, 470), mask="table"),
    Seg("bet_a", 196.70, 1.84, (600, 340, 640), (600, 335, 600)),
    Seg("bet_b", 198.54, 2.46, (690, 290, 580), (690, 285, 540)),
    Seg("shove", 213.30, 2.65, (525, 300, 600), (525, 290, 540)),
    Seg("react", 216.05, 1.2, (600, 330, 660), (600, 320, 560), ease_dur=0.15),
    Seg("wow", 232.55, 3.0, (680, 280, 600), (680, 280, 570)),
    Seg("stacks_a", 236.20, 1.75, (680, 270, 540)),
    Seg("stacks_b", 237.95, 1.0, (620, 330, 620)),
    Seg("poll", 290.0, 2.8, (700, 300, 600), (700, 290, 560), freeze=True, darken=0.62),
    Seg("tank_hf", 246.95, 3.15, (670, 290, 620)),
    Seg("tank_nw", 282.20, 2.10, (690, 300, 600)),
    Seg("tank_joe", 320.00, 1.5, (620, 300, 600), (620, 300, 560)),
    Seg("tank_30a", 370.80, 3.33, (710, 300, 600)),
    Seg("tank_30b", 374.13, 1.22, (640, 300, 600)),
    Seg("tank_kd", 399.75, 2.2, (720, 320, 600)),
    Seg("fold_a", 422.60, 1.37, (640, 330, 660)),
    Seg("fold_b", 423.97, 3.05, (760, 300, 600)),
    Seg("fold_c", 427.02, 2.28, (560, 360, 640)),
    Seg("regret", 433.75, 4.0, (860, 330, 640), (860, 320, 600)),
    Seg("build", 437.70, 1.4, (860, 320, 600), (860, 310, 540), freeze=True, darken=0.45, blur=5),
    Seg("reveal", 425.80, 4.5, (640, 360, 720), (260, 460, 520), freeze=True, mask="none", ease_dur=0.9),
    Seg("read", 345.0, 3.5, (735, 300, 600), (735, 290, 560), mask="none"),
    Seg("twist", 360.20, 5.0, (760, 340, 640), (740, 330, 600), mask="none", hold_after=2.1),
    Seg("outro", 353.0, 4.0, (735, 300, 600), (735, 295, 570), mask="none"),
]

_f = 0
for _s in SEGS:
    _s.nf = round(_s.dur * FPS)
    _s.f0 = _f
    _f += _s.nf
NFRAMES = _f
DURATION = float(NFRAMES / FPS)
BY_NAME = {s.name: s for s in SEGS}


def T(name: str, off: float = 0.0) -> float:
    """Output time of a segment start plus an offset."""
    return BY_NAME[name].start + off


def S(name: str, src_t: float) -> float:
    """Output time at which a segment shows a given source time."""
    s = BY_NAME[name]
    return T(name, (src_t - s.src) / s.speed)


def frame_time(k: int) -> float:
    return float(k / FPS)


def seg_of(k: int) -> Seg:
    for s in SEGS:
        if s.f0 <= k < s.f0 + s.nf:
            return s
    return SEGS[-1]


def src_time(seg: Seg, lt: float) -> float:
    if seg.freeze:
        return seg.src
    if seg.hold_after is not None:
        lt = min(lt, seg.hold_after)
    return seg.src + seg.speed * lt


# Voice clips: (name, [(src_a, src_b), ...], output start, caption line keys).
GAP = 0.12
VOICE = [
    ("hook", [(328.12, 329.08), (330.66, 333.20)], T("hook", 0.15), ["@hook"]),
    ("cowboys", [(23.72, 25.62)], S("ian", 23.72), ["cowboys"]),
    ("atleast", [(46.88, 48.81)], S("joe", 46.88), ["atleast"]),
    ("farbest", [(66.71, 68.88)], T("flop", 0.28), ["farbest"]),
    ("hate", [(125.90, 128.36)], T("turn", 0.10), ["hate"]),
    ("bet", [(197.02, 201.16)], S("bet_a", 197.02), ["bet800", "center"]),
    ("wow", [(232.62, 235.74)], S("wow", 232.62), ["wow"]),
    ("stacks", [(236.90, 239.07)], S("stacks_a", 236.90), ["stacks"]),
    ("tank_hf", [(247.06, 250.14)], S("tank_hf", 247.06), ["highfive"]),
    ("tank_nw", [(282.50, 284.32)], S("tank_nw", 282.50), ["noway"]),
    ("tank_30", [(370.86, 375.43)], S("tank_30a", 370.86), ["thirty"]),
    ("tank_kd", [(399.84, 401.95)], S("tank_kd", 399.84), ["killed"]),
    # Runs on through an uncaptioned "Oh my God!" so the cut lands in the quiet before "What".
    ("beating", [(423.70, 426.68)], S("fold_a", 423.70), ["beating"]),
    ("whatis", [(426.69, 430.15)], S("fold_b", 426.69), ["whatis"]),
    ("camo", [(440.52, 443.645)], T("reveal", 1.9), ["camo"]),
    ("figure", [(443.635, 444.86)], T("read", 0.62), ["figure"]),
    ("icant", [(446.16, 447.30)], T("outro", 1.9), ["icant"]),
]


def clip_offset(parts, st: float) -> float | None:
    """Clip-relative time of a source time, or None when it falls outside every part."""
    off = 0.0
    for a, b in parts:
        if a - 0.05 <= st < b:
            return off + max(0.0, st - a)
        off += (b - a) + GAP
    return None


# ---------------------------------------------------------------- camera moves

PUNCHES = [(T("fold_b", 0.78), 0.12, 0.25)]
SHAKES = [(T("shove", 2.45), 16.0), (T("fold_b", 0.78), 12.0)]
FLASHES = [(T("shove", 2.45), 0.25, 0.75), (T("reveal"), 0.35, 0.9)]


def crop_at(k: int, seg: Seg, lt: float) -> tuple[float, float, float]:
    t = frame_time(k)
    a = np.array(seg.crop, float)
    b = np.array(seg.crop_end or seg.crop, float)
    p = gfx.ease_in_out(lt / seg.ease_dur) if seg.ease_dur else gfx.clamp01(lt / seg.dur)
    cx, cy, size = a + (b - a) * p
    for tp, amount, tau in PUNCHES:
        if t >= tp:
            size *= 1 - amount * math.exp(-(t - tp) / tau)
    for ts, amp in SHAKES:
        dt = t - ts
        if 0 <= dt < 0.6:
            rng = np.random.default_rng(7919 + k)
            g = amp * math.exp(-dt / 0.18)
            cx += rng.uniform(-1, 1) * g
            cy += rng.uniform(-1, 1) * g
    size = min(size, float(SH))
    cx = min(max(cx, size / 2), SW - size / 2)
    cy = min(max(cy, size / 2), SH - size / 2)
    return float(cx), float(cy), float(size)


# ---------------------------------------------------------------- spoiler mask

MASKS = {"default": (0, 380, 425, 700), "table": (0, 370, 405, 700)}
FEATHER = 14


@functools.lru_cache(maxsize=None)
def _mask_alpha(kind: str):
    x0, y0, x1, y1 = MASKS[kind]
    fy0, fx1 = max(0, y0 - FEATHER), min(SW, x1 + FEATHER)
    ys = np.arange(fy0, y1, dtype=np.float32)[:, None]
    xs = np.arange(x0, fx1, dtype=np.float32)[None, :]
    dy = np.clip((y0 - ys) / FEATHER, 0, 1)
    dx = np.clip((xs - (x1 - 1)) / FEATHER, 0, 1)
    alpha = 1 - np.clip(np.sqrt(dy ** 2 + dx ** 2), 0, 1)
    return (x0, fy0, fx1, y1), alpha[..., None]


def apply_mask(rgb: np.ndarray, kind: str) -> np.ndarray:
    """Pixelate, grey and darken the hole-card graphics; fully opaque inside the rectangle."""
    if kind == "none":
        return rgb
    (x0, y0, x1, y1), alpha = _mask_alpha(kind)
    reg = rgb[y0:y1, x0:x1].astype(np.float32)
    h, w = reg.shape[:2]
    small = cv2.resize(reg, (max(1, w // 20), max(1, h // 20)), interpolation=cv2.INTER_AREA)
    big = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 6)
    proc = (0.35 * big + 0.65 * big.mean(axis=2, keepdims=True)) * 0.55
    out = rgb.copy()
    out[y0:y1, x0:x1] = np.clip(proc * alpha + reg * (1 - alpha), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- frame pieces

def background(rgb: np.ndarray, cx: float) -> np.ndarray:
    x0 = int(round(min(max(cx - 202.5, 0), SW - 405)))
    small = cv2.resize(rgb[:, x0:x0 + 405], (54, 96), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.2)
    mid = cv2.GaussianBlur(cv2.resize(small, (270, 480), interpolation=cv2.INTER_CUBIC), (0, 0), 2.5)
    return cv2.resize(mid, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 0.35


def warp(rgb: np.ndarray, cx: float, cy: float, size: float) -> np.ndarray:
    s = BOX / size
    m = np.float32([[s, 0, BOX / 2 - s * cx], [0, s, BOX / 2 - s * cy]])
    out = cv2.warpAffine(rgb, m, (BOX, BOX), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    soft = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.35, soft, -0.35, 0)


def box_effects(box: np.ndarray, seg: Seg, lt: float) -> np.ndarray:
    f = box.astype(np.float32)
    if seg.blur > 0:
        sigma = seg.blur * gfx.clamp01(lt / 0.3)
        if sigma > 0.3:
            f = cv2.GaussianBlur(f, (0, 0), sigma)
    if seg.darken < 1:
        f *= 1 - (1 - seg.darken) * gfx.clamp01(lt / 0.25)
    if seg.rgb_split:
        out = f.copy()
        out[:, 14:, 0] = f[:, :-14, 0]
        out[:, :-14, 2] = f[:, 14:, 2]
        out *= 0.9
        out[::4] *= 0.72
        f = out
    return f


def flash(canvas: np.ndarray, t: float) -> None:
    for tf, dur, strength in FLASHES:
        dt = t - tf
        if 0 <= dt < dur:
            a = strength * (1 - dt / dur) ** 2
            canvas *= 1 - a
            canvas += 255 * a


class Source:
    """Frame-accurate reader: returns the source frame on screen at time t."""
    TOL = 0.004

    def __init__(self, path: Path = SRC):
        self.c = av.open(str(path))
        self.vs = self.c.streams.video[0]
        self.vs.thread_type = "AUTO"
        self.tb = float(self.vs.time_base)
        self.it = None
        self.cur = None
        self.nxt = None
        self.seeks = 0
        self._key = None
        self._rgb = None

    def _restart(self, t: float) -> None:
        for back in (1, 3, 8, 20, 60, None):
            target = 0.0 if back is None else max(0.0, t - back)
            self.c.seek(int(target / self.tb), stream=self.vs, backward=True)
            self.it = self.c.decode(self.vs)
            first = next(self.it)
            self.seeks += 1
            if first.time <= t + self.TOL or back is None:
                self.cur, self.nxt = first, None
                return

    def frame_at(self, t: float):
        if self.cur is None or t + self.TOL < self.cur.time or t > self.cur.time + 2.0:
            self._restart(t)
        while True:
            if self.nxt is None:
                try:
                    self.nxt = next(self.it)
                except StopIteration:
                    break
            if self.nxt.time <= t + self.TOL:
                self.cur, self.nxt = self.nxt, None
            else:
                break
        return self.cur

    def rgb_at(self, t: float):
        fr = self.frame_at(t)
        if fr.pts != self._key:
            self._key, self._rgb = fr.pts, fr.to_ndarray(format="rgb24")
        return fr, self._rgb


def render_video(k: int, src: Source) -> tuple[np.ndarray, dict]:
    """Background plus the cropped video box for output frame k (float32 RGB)."""
    seg = seg_of(k)
    lt = float((k - seg.f0) / FPS)
    st = src_time(seg, lt)
    fr, rgb = src.rgb_at(st)
    rgb = apply_mask(rgb, seg.mask)
    cx, cy, size = crop_at(k, seg, lt)
    canvas = background(rgb, cx)
    canvas[BOX_Y:BOX_Y + BOX] = box_effects(warp(rgb, cx, cy, size), seg, lt)
    info = {"i": k, "t": round(frame_time(k), 4), "seg": seg.name, "src_t": round(st, 4),
            "pts_time": round(float(fr.time), 4), "mask": seg.mask,
            "crop": [round(cx, 1), round(cy, 1), round(size, 1)]}
    return canvas, info
