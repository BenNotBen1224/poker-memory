"""Read the TV graphics off one picture: World Poker Tour, season 16 style, 1280x720.

Boxes are (left, top, right, bottom) in pixels. Player panels stack upward from slot 0, SLOT_PITCH apart, and a
panel slides SLIDE pixels right while its player is to act heads-up. The panels are see-through, so text is
compared as a mask of bright pixels, never as grey levels.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .video import Picture

Box = tuple[int, int, int, int]

SLOTS, SLOT_PITCH, SLIDE = 4, 96, 35
CARDS: tuple[Box, Box] = ((132, 512, 167, 568), (170, 512, 205, 568))
NAME: Box = (212, 520, 322, 542)  # the win % to its right is left out
STATUS: Box = (216, 550, 364, 571)  # inside the coloured border
HOLE_RANK: Box = (2, 1, 34, 29)  # relative to the card's top-left
BOARD: tuple[Box, ...] = tuple((189 + 37 * i, 616, 224 + 37 * i, 677) for i in range(5))
BOARD_RANK: Box = (2, 2, 33, 34)
STRIP: Box = (189, 616, 373, 677)
BLINDS: Box = (190, 622, 370, 644)
ANTE: Box = (190, 652, 370, 672)
# Every box above lies inside AREA: from slot 3's cards at the top left to the board's bottom and a slid status
# line's right edge.
AREA: Box = (CARDS[0][0], CARDS[0][1] - SLOT_PITCH * (SLOTS - 1), STATUS[2] + SLIDE, STRIP[3])

TEXT_CUT = 150  # graphics text is near white; the panel and the video behind it are darker
DIM_CUT = 70  # some status lines are drawn dim, peaking near 100; an empty one stays under about 55
SAME = 0.13  # see `same`
RANKS = ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2")


@dataclass(frozen=True)
class Card:
    glyph: np.ndarray  # ink of the rank, as a mask
    rank: np.ndarray  # brightness crop of the rank, for OCR
    suit: str


@dataclass(frozen=True)
class Panel:
    slot: int
    slid: bool
    name: np.ndarray  # mask
    name_crop: np.ndarray
    status: np.ndarray | None  # mask; None when the status line is empty
    status_crop: np.ndarray
    cards: tuple[Card | None, Card | None]  # None while the card is face down


@dataclass(frozen=True)
class Look:
    time: float
    panels: tuple[Panel, ...]
    board: tuple[Card | None, ...]  # face-up cards only
    strip: np.ndarray  # mask of the whole board strip, to time its changes
    blinds: np.ndarray
    ante: np.ndarray


def _crop(plane: np.ndarray, box: Box, dx: int = 0, dy: int = 0) -> np.ndarray:
    left, top, right, bottom = box
    return plane[top + dy : bottom + dy, left + dx : right + dx]


def _shift(box: Box, dx: int, dy: int) -> Box:
    left, top, right, bottom = box
    return left + dx, top + dy, right + dx, bottom + dy


def _count(mask: np.ndarray, axis: int) -> np.ndarray:
    """How many pixels are set along an axis at most 255 long, counted in bytes: far quicker than numpy's 64-bit
    default."""
    return mask.view(np.uint8).sum(axis=axis, dtype=np.uint8)


def _span_sums(columns: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Sums of the last axis over each [start, stop) span, from one running total."""
    run = np.zeros((*columns.shape[:-1], columns.shape[-1] + 1), np.int64)
    np.cumsum(columns, axis=-1, dtype=np.int64, out=run[..., 1:])
    starts, stops = zip(*spans)
    return run[..., list(stops)] - run[..., list(starts)]


def _percentiles(crops: np.ndarray, q: int) -> np.ndarray:
    """Exactly `np.percentile(crop, q)` for each crop in a stack of 8-bit crops, from one sort: numpy sorts 8-bit
    values stably by counting them, faster than it partitions."""
    flat = crops.reshape(len(crops), -1)
    at = (flat.shape[1] - 1) * (q / 100)
    below = math.floor(at)
    gamma = at - below
    a, b = np.sort(flat, axis=1, kind="stable")[:, below : below + 2].T.astype(np.int64)
    return a + (b - a) * gamma if gamma < 0.5 else b - (b - a) * (1 - gamma)


def _cards(pictures: list[Picture], where: list[tuple[int, Box]], rank_box: Box) -> list[Card]:
    """Face-up cards of one size, given as (picture, box): the rank as a mask and a crop, the suit from its ink."""
    if not where:
        return []
    crops = np.stack([_crop(pictures[i].y, box) for i, box in where])
    cuts = _percentiles(crops, 90) - 40
    left, top, right, bottom = rank_box
    ranks = np.ascontiguousarray(crops[:, top:bottom, left:right])
    glyphs = ranks < cuts[:, None, None]
    # The suit is the mean colour of the ink, on the half-size colour planes. There, boxes of one size can still
    # differ by a pixel, so read the largest and leave out whatever lies past each card's own edge.
    halves = [(i, *(c // 2 for c in box)) for i, box in where]
    tall = max(b - t for _, _, t, _, b in halves)
    wide = max(r - l for _, l, _, r, _ in halves)
    ys = np.stack([pictures[i].y[2 * t : 2 * (t + tall) : 2, 2 * l : 2 * (l + wide) : 2] for i, l, t, _, _ in halves])
    us = np.stack([pictures[i].u[t : t + tall, l : l + wide] for i, l, t, _, _ in halves])
    vs = np.stack([pictures[i].v[t : t + tall, l : l + wide] for i, l, t, _, _ in halves])
    rows = np.arange(tall) < np.array([b - t for _, _, t, _, b in halves])[:, None]
    cols = np.arange(wide) < np.array([r - l for _, l, _, r, _ in halves])[:, None]
    ink = (ys < cuts[:, None, None]) & rows[:, :, None] & cols[:, None, :]
    count = ink.sum(axis=(1, 2)).tolist()
    blue = (us * ink).sum(axis=(1, 2), dtype=np.uint32).tolist()
    red = (vs * ink).sum(axis=(1, 2), dtype=np.uint32).tolist()
    cards = []
    for k in range(len(where)):
        suit = "?"
        if count[k] >= 10:
            cb, cr = blue[k] / count[k] - 128, red[k] / count[k] - 128
            suit = "h" if cr > 20 else "d" if cb > 20 else "c" if cb < -10 and cr < -10 else "s"
        cards.append(Card(glyphs[k], ranks[k], suit))
    return cards


def look_all(pictures: list[Picture]) -> list[Look]:
    """Read the graphics off many pictures at once.

    Each step below is one numpy call for every picture together. numpy lets go of the interpreter lock inside
    each call and must win it back from the decoding threads, so the number of calls costs more than their size.
    """
    if not pictures:
        return []
    n = len(pictures)
    x0, y0, x1, y1 = AREA
    area = np.stack([p.y[y0:y1, x0:x1] for p in pictures])

    def cut(box: Box) -> np.ndarray:
        left, top, right, bottom = box
        return area[:, top - y0 : bottom - y0, left - x0 : right - x0]

    # Panels. A panel has two cards and a dark name bar. A face-up card is mostly white and a face-down one flat
    # grey. Video seen through the panel is neither, even a bright hat right behind it; a card still fading in or
    # sliding is neither too. [:, s] below starts at slot s's cards.
    slots = area[:, : SLOTS * SLOT_PITCH].reshape(n, SLOTS, SLOT_PITCH, x1 - x0)[:, ::-1]
    left, top, right, bottom = CARDS[0]
    spans = [(c_left - x0 + dx, c_right - x0 + dx) for dx in (0, SLIDE) for c_left, _, c_right, _ in CARDS]
    cards = slots[:, :, : bottom - top, : spans[-1][1]]
    white = _span_sums(_count(cards > 200, axis=2), spans)  # card 1, card 2, then both again slid
    face_up = white > 0.25 * (right - left) * (bottom - top)  # a white face; the grey back stays well under 200
    inset = cards[:, :, 2:-2]  # 2 pixels in from each edge
    inner = [(start + 2, stop - 2) for start, stop in spans]
    size = (right - left - 4) * (bottom - top - 4)
    total = _span_sums(inset.sum(axis=2, dtype=np.uint16), inner)
    # grey: mean 140 to 170, standard deviation under 20. The spread costs the most, and decides only where the
    # mean is in range on a card that is not face up: about 3 in 100.
    grey = (140 * size <= total) & (total <= 170 * size) & ~face_up
    i, s = np.nonzero(grey.any(axis=2))
    if len(i):
        squares = _span_sums(np.square(inset[i, s], dtype=np.uint16).sum(axis=1, dtype=np.uint32), inner)
        grey[i, s] &= size * squares - total[i, s] * total[i, s] < 400 * size * size
    is_card = face_up | grey

    left, top, right, bottom = NAME
    width = right - left
    bar = slots[:, :, top - CARDS[0][1] : bottom - CARDS[0][1], left - x0 : right - x0 + SLIDE]
    size = width * (bottom - top)
    dark = _span_sums(_count(bar < 110, axis=2), [(0, width), (SLIDE, SLIDE + width)])
    named = dark > size // 2  # the median is under 110
    for i, s, o in zip(*np.nonzero(dark == size // 2)):  # split down the middle: the median needs the values
        named[i, s, o] = np.median(bar[i, s, :, o * SLIDE : o * SLIDE + width]) < 110

    at_rest = (is_card[..., 0] & is_card[..., 1] & named[..., 0]).tolist()
    slid = (is_card[..., 2] & is_card[..., 3] & named[..., 1]).tolist()
    face_up = face_up.tolist()
    found = [(i, s, 0 if at_rest[i][s] else SLIDE) for i in range(n) for s in range(SLOTS)
             if at_rest[i][s] or slid[i][s]]

    def stack(box: Box) -> np.ndarray:
        return np.stack([_crop(pictures[i].y, box, dx, -SLOT_PITCH * s) for i, s, dx in found])

    holes = [(k, j) for k, (i, s, dx) in enumerate(found) for j in range(2) if face_up[i][s][j + 2 * (dx > 0)]]
    hole_cards = _cards(pictures, [(found[k][0], _shift(CARDS[j], found[k][2], -SLOT_PITCH * found[k][1]))
                                   for k, j in holes], HOLE_RANK)
    hole = dict(zip(holes, hole_cards))
    panels: list[list[Panel]] = [[] for _ in pictures]
    if found:
        names, statuses = stack(NAME), stack(STATUS)
        # status text can be dim: cut at three quarters of its brightest, and call it empty under 1% lit
        status_cuts = np.minimum(TEXT_CUT, np.maximum(DIM_CUT, 0.75 * _percentiles(statuses, 99)))
        status_masks = statuses > status_cuts[:, None, None]
        lit = (status_masks.sum(axis=(1, 2)) >= 0.01 * statuses[0].size).tolist()
        name_masks = names > TEXT_CUT
        for k, (i, s, dx) in enumerate(found):
            panels[i].append(Panel(s, dx > 0, name_masks[k], names[k], status_masks[k] if lit[k] else None,
                                   statuses[k], (hole.get((k, 0)), hole.get((k, 1)))))

    # The board cards fill the strip's height.
    strip = cut(STRIP)
    board_spans = [(b_left - STRIP[0], b_right - STRIP[0]) for b_left, _, b_right, _ in BOARD]
    left, top, right, bottom = BOARD[0]
    board_up = (_span_sums(_count(strip > 200, axis=1), board_spans) > 0.25 * (right - left) * (bottom - top)).tolist()
    shown = [(i, b) for i in range(n) for b in range(len(BOARD)) if board_up[i][b]]
    board = dict(zip(shown, _cards(pictures, [(i, BOARD[b]) for i, b in shown], BOARD_RANK)))
    strips = strip > TEXT_CUT
    blinds, antes = np.ascontiguousarray(cut(BLINDS)), np.ascontiguousarray(cut(ANTE))
    return [Look(p.time, tuple(panels[i]), tuple(board.get((i, b)) for b in range(len(BOARD))), strips[i],
                 blinds[i], antes[i]) for i, p in enumerate(pictures)]


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two masks show the same text: under SAME of the pixels differ in the worst 6-pixel-wide column band.
    One changed digit scores about 0.2 there; the same text over a moving background at most about 0.08. When even
    the total of differing pixels is under the limit, so is every band, so most calls stop at the total."""
    diff = a != b
    if np.count_nonzero(diff) / (6 * len(a)) < SAME:
        return True
    columns = np.add.reduce(diff.view(np.uint8), axis=0, dtype=np.uint8)  # no mask is over 255 high
    return int(_band(columns[None])[0]) / (6 * len(a)) < SAME


def _band(columns: np.ndarray) -> np.ndarray:
    """For each row of differing-pixel counts per column: the most in any 6-pixel-wide column band."""
    run = np.zeros((len(columns), columns.shape[1] + 1), np.int64)
    np.cumsum(columns, axis=1, dtype=np.int64, out=run[:, 1:])
    return (run[:, 6:] - run[:, :-6]).max(axis=1)


def _columns(mask: np.ndarray) -> np.ndarray:
    """Each pixel column of a mask (at most 64 high) as the bits of one number, so the differing pixels in a
    column are the set bits of an exclusive or."""
    packed = np.zeros((mask.shape[1], 8), np.uint8)
    packed[:, : (len(mask) + 7) // 8] = np.packbits(mask, axis=0).T
    return packed.view(np.uint64)[:, 0]


def _differ(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is not b
    return not same(a, b)


def _differ_card(a: Card | None, b: Card | None) -> bool:
    if a is None or b is None:
        return a is not b
    return a.suit != b.suit or _differ(a.glyph, b.glyph)


def same_look(a: Look, b: Look) -> bool:
    if [(q.slot, q.slid) for q in a.panels] != [(q.slot, q.slid) for q in b.panels]:
        return False
    for qa, qb in zip(a.panels, b.panels):
        if _differ(qa.name, qb.name) or _differ(qa.status, qb.status):
            return False
        if any(_differ_card(ca, cb) for ca, cb in zip(qa.cards, qb.cards)):
            return False
    if [c is None for c in a.board] != [c is None for c in b.board]:
        return False
    return not _differ(a.strip, b.strip)


class Catalog:
    """Numbers each distinct crop of one kind, so OCR reads each one once.

    Keeps the crop with the most ink for each number, which skips frames caught mid-fade.
    """

    def __init__(self) -> None:
        self.masks: list[np.ndarray] = []  # what new crops are matched against, one per number
        self.columns = np.empty((0, 0), np.uint64)  # the same masks as `_columns`, plus room
        self.crops: list[np.ndarray] = []
        self.crop_masks: list[np.ndarray] = []
        self.ink: list[int] = []  # pixels set in each crop mask
        self.count: list[int] = []
        self.last: list[bytes] = []  # the pixels of the latest mask given each number

    def add(self, mask: np.ndarray, crop: np.ndarray, hint: int | None = None) -> int:
        pixels = mask.tobytes()
        if hint is not None and pixels == self.last[hint]:
            # A mask given a number always passes `same` against that number's mask, and leaves at most as much ink.
            self.count[hint] += 1
            return hint
        if hint is not None and same(self.masks[hint], mask):
            i = hint
        else:
            columns = _columns(mask)
            i = self._find(columns, len(mask))
            if i is None:
                return self._new(mask, columns, crop, pixels)
        ink = int(np.count_nonzero(mask))
        if ink > self.ink[i]:
            self.crops[i], self.crop_masks[i], self.ink[i] = crop, mask, ink
        self.count[i] += 1
        self.last[i] = pixels
        return i

    def _new(self, mask: np.ndarray, columns: np.ndarray, crop: np.ndarray, pixels: bytes) -> int:
        i = len(self.count)
        if i == len(self.columns):  # full: double it
            self.columns = (np.concatenate((self.columns, np.empty_like(self.columns))) if i
                            else np.empty((16, len(columns)), np.uint64))
        self.columns[i] = columns
        self.masks.append(mask)
        self.crops.append(crop)
        self.crop_masks.append(mask)
        self.ink.append(int(np.count_nonzero(mask)))
        self.count.append(1)
        self.last.append(pixels)
        return i

    def _find(self, columns: np.ndarray, height: int) -> int | None:
        if not self.count:
            return None
        scores = _band(np.bitwise_count(self.columns[: len(self.count)] ^ columns))
        best = int(scores.argmin())
        return best if scores[best] < SAME * 6 * height else None

    def settled(self) -> list[int]:
        return [i for i, n in enumerate(self.count) if n >= 2]

    def text(self, i: int, pad: int = 4) -> np.ndarray:
        """The best crop trimmed to its text, so the see-through background beside it cannot be misread."""
        cols = np.flatnonzero(self.crop_masks[i].any(axis=0))
        return self.crops[i][:, max(0, cols[0] - pad) : cols[-1] + pad + 1] if len(cols) else self.crops[i]


class Ocr:
    """RapidOCR's text recogniser alone (no text finder, no rotation check), running PaddleOCR's small
    English-and-digits model (models/NOTICE): about 5x less work than RapidOCR's default Chinese model."""

    MODEL = Path(__file__).with_name("models") / "en_number_mobile_v2.0_rec_infer.onnx"

    def __init__(self) -> None:
        from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer
        from rapidocr_onnxruntime.main import DEFAULT_CFG_PATH
        from rapidocr_onnxruntime.utils import read_yaml, update_model_path

        config = update_model_path(read_yaml(DEFAULT_CFG_PATH))["Rec"]
        config["model_path"] = str(self.MODEL)
        self._rec = TextRecognizer(config)
        self._rec.rec_image_shape = [3, 32, 64]
        self.calls = 0
        self._counting = threading.Lock()  # calls may come from many threads at once

    def __call__(self, crops: list[np.ndarray]) -> list[str]:
        if not crops:
            return []
        with self._counting:
            self.calls += len(crops)
        out, _ = self._rec([np.repeat(c[:, :, None], 3, axis=2) for c in crops])
        return [text for text, _ in out]


def parse_rank(text: str) -> str | None:
    t = "".join(ch for ch in text.upper() if ch.isalnum())
    t = {"0": "Q", "O": "Q", "D": "Q", "IO": "T", "1O": "T", "LO": "T", "10": "T"}.get(t, t)
    return t if t in RANKS else None


_WORDS = ("TOCALL", "RAISETO", "RAISE", "ALLIN", "CALL", "BET", "CHECK", "FOLD", "WIN")


def parse_status(text: str) -> tuple[str, int | None] | None:
    """'TO CALL 30,000' -> ('TOCALL', 30000). OCR drops spaces and mixes up O/0, so both are allowed for."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper().replace("T0", "TO"))
    for word in _WORDS:
        if t.startswith(word):
            digits = t[len(word) :].replace("O", "0")
            return word, int(digits) if digits.isdigit() else None
    return None


def parse_chips(text: str) -> int | None:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KM]?)", text)
    if not m:
        return None
    return round(float(m.group(1)) * {"": 1, "K": 1_000, "M": 1_000_000}[m.group(2)])


def parse_blinds(blinds: str, ante: str) -> tuple[int, int, int] | None:
    """'SB/BB 30K/60K' and 'ANTE 10K' -> (30000, 60000, 10000)."""
    b = re.sub(r"[^A-Z0-9./]", "", blinds.upper()).replace("O", "0").removeprefix("SB/BB")
    a = re.sub(r"[^A-Z0-9.]", "", ante.upper()).removeprefix("ANTE").replace("O", "0")
    sb, _, bb = b.partition("/")
    nums = parse_chips(sb), parse_chips(bb), parse_chips(a) if a else 0
    return None if None in nums else nums
