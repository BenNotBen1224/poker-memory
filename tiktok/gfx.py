"""Sprites (text, cards, pills, chips) and compositing for the TikTok edit.

Sprites are float32 premultiplied RGBA arrays so they can be resized and
blended without dark fringes.
"""
from __future__ import annotations

import functools
import math
import re
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
FONT_TT = str(HERE / "fonts" / "TikTokSans.ttf")
FONT_EMOJI = r"C:\Windows\Fonts\seguiemj.ttf"
FONT_SYM = r"C:\Windows\Fonts\seguisym.ttf"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 230, 0)
RED = (232, 19, 43)
GREEN = (52, 199, 89)
SUIT_ON_LIGHT = {"♠": (17, 17, 17), "♥": (215, 38, 61), "♦": (31, 111, 235), "♣": (30, 158, 74)}
SUIT_ON_DARK = {"♠": (255, 255, 255), "♥": (255, 72, 92), "♦": (77, 150, 255), "♣": (52, 199, 89)}

_TT_CMAP = set(TTFont(FONT_TT).getBestCmap())
_EMOJI_CMAP = set(TTFont(FONT_EMOJI).getBestCmap())


# ---------------------------------------------------------------- fonts

@functools.lru_cache(maxsize=None)
def tt(size: int, weight: str = "Black") -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(FONT_TT, size)
    font.set_variation_by_name(weight)
    return font


@functools.lru_cache(maxsize=None)
def emoji_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_EMOJI, size)


@functools.lru_cache(maxsize=None)
def sym_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_SYM, size)


def _kind(ch: str) -> str:
    if ch in SUIT_ON_LIGHT:
        return "suit"
    cp = ord(ch)
    if cp >= 0x1F000 or (cp not in _TT_CMAP and cp in _EMOJI_CMAP):
        return "emoji"
    if cp in _TT_CMAP:
        return "text"
    return "sym"


def parse_markup(text: str) -> list[tuple[str, bool]]:
    """'a *b* c' -> [('a ', False), ('b', True), (' c', False)]."""
    out = []
    for part in re.split(r"(\*[^*]+\*)", text):
        if not part:
            continue
        if len(part) > 1 and part[0] == part[-1] == "*":
            out.append((part[1:-1], True))
        else:
            out.append((part, False))
    return out


# ---------------------------------------------------------------- conversion

def to_sprite(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGBA"), dtype=np.float32)
    a[..., :3] *= a[..., 3:4] / 255.0
    return a


def _supersampled(w: int, h: int, s: int = 3):
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _down(img: Image.Image, s: int = 3) -> Image.Image:
    return img.resize((img.width // s, img.height // s), Image.LANCZOS)


# ---------------------------------------------------------------- text

def _items(segments, size, weight, fill, hl, suits, emoji_scale):
    font = tt(size, weight)
    efont = emoji_font(max(8, int(size * emoji_scale)))
    sfont = sym_font(size)
    items, x = [], 0.0
    for seg_text, is_hl in segments:
        color = hl if is_hl else fill
        buf, buf_kind = "", None

        def flush():
            nonlocal buf, buf_kind, x
            if buf:
                items.append(("text", buf, font, color, x))
                x += font.getlength(buf)
            buf, buf_kind = "", None

        for ch in seg_text.replace("\ufe0f", ""):
            k = _kind(ch)
            if k == "text":
                buf += ch
                continue
            flush()
            if k == "emoji":
                items.append(("emoji", ch, efont, None, x))
                x += efont.getlength(ch) + size * 0.04
            else:
                c = suits.get(ch, color) if k == "suit" else color
                items.append(("sym", ch, sfont, c, x))
                x += sfont.getlength(ch)
        flush()
    return items, x, font


@functools.lru_cache(maxsize=512)
def text_line(text: str, size: int, weight: str = "Black", fill=WHITE, hl=YELLOW,
              stroke: int = 0, stroke_fill=BLACK, dark_suits: bool = True,
              emoji_scale: float = 0.9, shadow: int = 0) -> np.ndarray:
    """One line of markup text as a premultiplied RGBA sprite."""
    suits = SUIT_ON_DARK if dark_suits else SUIT_ON_LIGHT
    items, width, font = _items(parse_markup(text), size, weight, fill, hl, suits, emoji_scale)
    asc, desc = font.getmetrics()
    pad = stroke + 8 + shadow
    w, h = int(math.ceil(width)) + 2 * pad, asc + desc + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    base = pad + asc
    if stroke:
        for kind, s, f, _, x in items:
            if kind != "emoji":
                d.text((pad + x, base), s, font=f, fill=stroke_fill, anchor="ls",
                       stroke_width=stroke, stroke_fill=stroke_fill)
    for kind, s, f, c, x in items:
        if kind == "emoji":
            d.text((pad + x, base), s, font=f, anchor="ls", embedded_color=True)
        else:
            d.text((pad + x, base), s, font=f, fill=c, anchor="ls")
    if shadow:
        alpha = img.split()[3].filter(ImageFilter.GaussianBlur(shadow))
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sh.putalpha(alpha.point(lambda v: int(v * 0.55)))
        sh = sh.transform(sh.size, Image.AFFINE, (1, 0, -shadow * 0.4, 0, 1, -shadow * 0.6))
        img = Image.alpha_composite(sh, img)
    return to_sprite(img)


def wrap(text: str, size: int, weight: str, max_width: int) -> list[str]:
    """Greedy wrap that keeps *highlight* markup balanced on every line."""
    lines = []
    for para in text.split("\n"):
        tokens, hl = [], False
        for word in para.split(" "):
            plain = word.replace("*", "")
            starts, ends = word.startswith("*"), word.endswith("*") and len(word) > 1
            tokens.append((plain, hl or starts))
            if starts and not ends:
                hl = True
            if ends and not starts:
                hl = False
            if starts and ends:
                hl = False
        cur: list[tuple[str, bool]] = []
        for tok in tokens:
            trial = cur + [tok]
            if cur and _measure(trial, size, weight) > max_width:
                lines.append(_join(cur))
                cur = [tok]
            else:
                cur = trial
        if cur:
            lines.append(_join(cur))
    return lines


def _join(tokens) -> str:
    out = []
    for word, hl in tokens:
        out.append(f"*{word}*" if hl and word else word)
    return " ".join(out).replace("* *", " ")


def _measure(tokens, size, weight) -> float:
    items, width, _ = _items(parse_markup(_join(tokens)), size, weight, WHITE, YELLOW, SUIT_ON_DARK, 0.9)
    return width


def stack(sprites: list[np.ndarray], gap: int = 0, align: str = "center") -> np.ndarray:
    w = max(s.shape[1] for s in sprites)
    h = sum(s.shape[0] for s in sprites) + gap * (len(sprites) - 1)
    out = np.zeros((max(h, 1), w, 4), np.float32)
    y = 0
    for s in sprites:
        x = (w - s.shape[1]) // 2 if align == "center" else 0
        region = out[y:y + s.shape[0], x:x + s.shape[1]]
        a = s[..., 3:4] / 255.0
        region[:] = s + region * (1 - a)
        y += s.shape[0] + gap
    return out


@functools.lru_cache(maxsize=256)
def stroke_text(text: str, size: int = 66, weight: str = "Black", max_width: int = 900,
                fill=WHITE, hl=YELLOW, stroke: int = 0, line_gap: float = -0.18) -> np.ndarray:
    """Centered multi-line white text with a heavy black outline."""
    stroke = stroke or max(4, size // 8)
    lines = wrap(text, size, weight, max_width)
    sprites = [text_line(l, size, weight, fill, hl, stroke, BLACK, True, 0.9, shadow=4) for l in lines]
    return stack(sprites, gap=int(size * line_gap))


def rounded_rect(w: int, h: int, r: int, color, s: int = 3, outline=None, ow: int = 0) -> Image.Image:
    img, d = _supersampled(w, h, s)
    d.rounded_rectangle([0, 0, w * s - 1, h * s - 1], radius=r * s, fill=color,
                        outline=outline, width=ow * s)
    return _down(img, s)


@functools.lru_cache(maxsize=256)
def box_text(text: str, size: int = 60, weight: str = "Bold", max_width: int = 900,
             bg=WHITE, fg=BLACK, hl=RED, rot: float = 0.0) -> np.ndarray:
    """TikTok-style text with a rounded box behind every line."""
    lines = wrap(text, size, weight, max_width - int(size * 0.7))
    pad_x, pad_y, radius = int(size * 0.32), int(size * 0.14), int(size * 0.30)
    dark = sum(bg[:3]) < 380
    line_imgs = []
    for l in lines:
        spr = text_line(l, size, weight, fg, hl, 0, BLACK, dark, 0.9)
        im = Image.fromarray(_unpremultiply(spr), "RGBA")
        bw, bh = im.width - 16 + 2 * pad_x, im.height - 16 + 2 * pad_y
        box = rounded_rect(bw, bh, radius, bg)
        box.alpha_composite(im, (pad_x - 8, pad_y - 8))
        line_imgs.append(box)
    overlap = int(size * 0.12)
    w = max(i.width for i in line_imgs)
    h = sum(i.height for i in line_imgs) - overlap * (len(line_imgs) - 1)
    canvas = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    y = 20
    for im in line_imgs:
        canvas.alpha_composite(im, ((canvas.width - im.width) // 2, y))
        y += im.height - overlap
    if rot:
        canvas = canvas.rotate(rot, resample=Image.BICUBIC, expand=True)
    return to_sprite(canvas)


def _unpremultiply(s: np.ndarray) -> np.ndarray:
    a = s[..., 3:4]
    rgb = np.where(a > 0, s[..., :3] * 255.0 / np.maximum(a, 1e-6), 0)
    return np.concatenate([np.clip(rgb, 0, 255), a], axis=2).astype(np.uint8)


@functools.lru_cache(maxsize=128)
def pill(text: str, size: int = 44, bg=WHITE, fg=BLACK, weight: str = "Bold",
         rot: float = 0.0, outline=None) -> np.ndarray:
    spr = text_line(text, size, weight, fg, RED, 0, BLACK, sum(bg[:3]) < 380, 0.9)
    im = Image.fromarray(_unpremultiply(spr), "RGBA")
    pad_x, pad_y = int(size * 0.55), int(size * 0.22)
    bw, bh = im.width - 16 + 2 * pad_x, im.height - 16 + 2 * pad_y
    box = rounded_rect(bw, bh, bh // 2, bg, outline=outline, ow=3 if outline else 0)
    box.alpha_composite(im, (pad_x - 8, pad_y - 8))
    canvas = Image.new("RGBA", (box.width + 24, box.height + 24), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 110), (14, 18, 14 + box.width, 18 + box.height), box.split()[3])
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(5)))
    canvas.alpha_composite(box, (12, 12))
    if rot:
        canvas = canvas.rotate(rot, resample=Image.BICUBIC, expand=True)
    return to_sprite(canvas)


# ---------------------------------------------------------------- cards and chips

@functools.lru_cache(maxsize=64)
def card(rank: str, suit: str = "", w: int = 150, back: bool = False, dim: bool = False) -> np.ndarray:
    s = 3
    h = int(w * 1.4)
    pad = int(w * 0.12)
    W, H = (w + 2 * pad) * s, (h + 2 * pad) * s
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).rounded_rectangle(
        [pad * s + 4 * s, pad * s + 8 * s, (pad + w) * s + 4 * s, (pad + h) * s + 8 * s],
        radius=int(w * 0.1 * s), fill=140)
    sh = sh.filter(ImageFilter.GaussianBlur(7 * s))
    img.paste((0, 0, 0, 255), (0, 0, W, H), sh)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = pad * s, pad * s, (pad + w) * s, (pad + h) * s
    r = int(w * 0.1 * s)
    if back:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=(182, 24, 40), outline=WHITE, width=5 * s)
        inset = int(w * 0.09 * s)
        d.rounded_rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], radius=r // 2,
                            outline=(255, 255, 255, 140), width=2 * s)
        f = tt(int(w * 0.62 * s), "Black")
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), "?", font=f, fill=WHITE, anchor="mm")
    else:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=WHITE, outline=(205, 205, 205), width=2 * s)
        color = SUIT_ON_LIGHT.get(suit, (17, 17, 17))
        if suit:
            f = tt(int(w * 0.50 * s), "Black")
            d.text((x0 + int(w * 0.09 * s), y0 + int(w * 0.06 * s)), rank, font=f, fill=color, anchor="lt")
            sf = sym_font(int(w * 0.66 * s))
            d.text((x1 - int(w * 0.10 * s), y1 - int(w * 0.06 * s)), suit, font=sf, fill=color, anchor="rs")
        else:
            f = tt(int(w * 0.72 * s), "Black")
            d.text(((x0 + x1) / 2, (y0 + y1) / 2), rank, font=f, fill=color, anchor="mm")
    out = _down(img, s)
    if dim:
        arr = np.asarray(out).copy()
        arr[..., :3] = (arr[..., :3] * 0.45).astype(np.uint8)
        out = Image.fromarray(arr, "RGBA")
    return to_sprite(out)


@functools.lru_cache(maxsize=16)
def chip(d: int = 70, used: bool = True) -> np.ndarray:
    s = 3
    img, dr = _supersampled(d + 10, d + 10, s)
    c = (d + 10) * s / 2
    R = d * s / 2
    if used:
        dr.ellipse([c - R, c - R, c + R, c + R], fill=(214, 38, 52), outline=(90, 0, 10), width=2 * s)
        for k in range(8):
            a = k * math.pi / 4
            pts = []
            for da, rr in ((-0.16, R * 0.98), (0.16, R * 0.98), (0.16, R * 0.74), (-0.16, R * 0.74)):
                pts.append((c + rr * math.cos(a + da), c + rr * math.sin(a + da)))
            dr.polygon(pts, fill=WHITE)
        dr.ellipse([c - R * 0.58, c - R * 0.58, c + R * 0.58, c + R * 0.58], fill=(240, 240, 240),
                   outline=(214, 38, 52), width=3 * s)
        f = tt(int(d * 0.34 * s), "Black")
        dr.text((c, c), "30s", font=f, fill=(150, 10, 25), anchor="mm")
    else:
        dr.ellipse([c - R, c - R, c + R, c + R], outline=(255, 255, 255, 150), width=4 * s)
    return to_sprite(_down(img, s))


@functools.lru_cache(maxsize=8)
def dots(n_on: int = 3, n: int = 10, d: int = 54, gap: int = 16) -> np.ndarray:
    s = 3
    w = n * d + (n - 1) * gap + 8
    img, dr = _supersampled(w, d + 8, s)
    for i in range(n):
        x = (4 + i * (d + gap)) * s
        box = [x, 4 * s, x + d * s, (4 + d) * s]
        if i < n_on:
            dr.ellipse(box, fill=GREEN, outline=BLACK, width=3 * s)
        else:
            dr.ellipse(box, fill=(255, 255, 255, 40), outline=WHITE, width=4 * s)
    return to_sprite(_down(img, s))


# ---------------------------------------------------------------- compositing

def blend(canvas: np.ndarray, spr: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    """Blend a premultiplied sprite onto a float32 RGB canvas at top-left (x, y)."""
    if alpha <= 0:
        return
    h, w = spr.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    dst = canvas[y0:y1, x0:x1]
    a = src[..., 3:4] * (alpha / 255.0)
    dst *= 1.0 - a
    dst += src[..., :3] * alpha


def place(canvas, spr, cx: float, cy: float, scale: float = 1.0, alpha: float = 1.0,
          sx: float | None = None) -> None:
    """Blend a sprite centered at (cx, cy) with optional uniform/horizontal scale."""
    if scale <= 0.01 or alpha <= 0:
        return
    sxv = scale if sx is None else sx
    if abs(sxv - 1) > 1e-3 or abs(scale - 1) > 1e-3:
        w = max(1, int(round(spr.shape[1] * sxv)))
        h = max(1, int(round(spr.shape[0] * scale)))
        spr = cv2.resize(spr, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    blend(canvas, spr, int(round(cx - spr.shape[1] / 2)), int(round(cy - spr.shape[0] / 2)), alpha)


def scribble_circle(canvas, cx, cy, rx, ry, progress, color=(255, 45, 45), width=9, seed=3, rot=-8):
    """A hand-drawn ellipse that draws itself as progress goes 0 -> 1."""
    if progress <= 0:
        return
    rng = np.random.default_rng(seed)
    n = 220
    turns = 1.12
    th = np.linspace(0, 2 * math.pi * turns, n)
    wob = 1 + 0.035 * np.sin(3 * th + rng.uniform(0, 6)) + 0.02 * np.sin(7 * th + rng.uniform(0, 6))
    grow = 1 + 0.06 * th / th[-1]
    a = math.radians(rot)
    px = rx * wob * grow * np.cos(th)
    py = ry * wob * grow * np.sin(th)
    x = cx + px * math.cos(a) - py * math.sin(a)
    y = cy + px * math.sin(a) + py * math.cos(a)
    k = max(2, int(n * min(1.0, progress)))
    pts = np.stack([x[:k], y[:k]], axis=1).round().astype(np.int32)
    layer = np.clip(canvas, 0, 255).astype(np.uint8)
    cv2.polylines(layer, [pts], False, (0, 0, 0), width + 6, lineType=cv2.LINE_AA)
    cv2.polylines(layer, [pts], False, color, width, lineType=cv2.LINE_AA)
    canvas[:] = layer.astype(np.float32)


# ---------------------------------------------------------------- easing

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_in_out(x: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * clamp01(x))


def ease_out_back(x: float, s: float = 1.9) -> float:
    x = clamp01(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def pop(dt: float, dur: float = 0.22) -> float:
    if dt >= dur:
        return 1.0
    return 0.55 + 0.45 * ease_out_back(dt / dur)
