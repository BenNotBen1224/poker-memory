"""Development-calibrated red/black suit glyphs, separate from WPT's four-colour path.

Calibration: candidate63 source frames 245491 and 245251 (club raster variant), native 854x480. Inputs are RGB
interior suit crops, excluding card borders and rank text, from two nearby frames.
This is not a general font/layout recognizer or held-out accuracy result.
"""

from __future__ import annotations

import numpy as np

_SHAPES = {
    "d": (".....##.....", "....####....", "...#####....", "...######...", "..########..",
          ".##########.", "############", "############", "############", ".##########.",
          "..########..", "...######...", "....#####...", "....####....", ".....##....."),
    "s": (".....###.....", "....#####....", "...#######...", "..#########..", ".###########.",
          "############.", "#############", "#############", "#############", "#############",
          "############.", "..#########..", ".....##......", "....#####....", "...######...."),
    "c": (".....####.....", "....######....", "....######....", "....######....", ".....#####....",
          "..###.######..", ".############.", ".#############", "##############", ".############.",
          "..###.##.###..", "......##......", "......##......", ".....####.....", "....######...."),
    "h": (".#####.#####.", "############.", "#############", "#############", "#############",
          "#############", "#############", "############.", ".###########.", ".##########..",
          "..#########..", "...#######...", "....#####....", "....####.....", "......#......"),
}


def _center(mask: np.ndarray) -> np.ndarray:
    centered = np.zeros((17, 18), bool)
    left = (18 - mask.shape[1]) // 2
    centered[1:1 + mask.shape[0], left:left + mask.shape[1]] = mask
    return centered


_TEMPLATES = {suit: _center(np.array([[ch == "#" for ch in row] for row in shape]))
              for suit, shape in _SHAPES.items()}
_CLUB_RASTER = _center(np.array([[ch == "#" for ch in row] for row in (
    "....####.....", "...######....", "...######....", "...######....", "....#####....",
    ".#######.##..", "############.", "#############", "#############", "#############",
    ".####.#.####.", "......#......", ".....##......", "....####.....", "...######....",
)]))


def _glyph(rgb: np.ndarray) -> tuple[np.ndarray, bool] | None:
    if rgb.dtype != np.uint8 or rgb.shape not in ((16, 16, 3), (17, 16, 3)):
        raise ValueError("suit crop must be native uint8 RGB, 16 pixels wide and 16 or 17 high")
    brightness = rgb.mean(axis=2)
    ink = brightness < 150
    count = int(ink.sum())
    if not 70 <= count <= 175 or np.median(brightness[~ink]) < 180:
        return None
    if np.mean(ink != (brightness < 120)) > 0.065:
        return None  # Reject ghosted/fading shapes whose ink disappears at a nearby threshold.
    if np.mean((brightness >= 120) & (brightness < 180)) > 0.13:
        return None
    ys, xs = np.nonzero(ink)
    height, width = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    if height != 15 or not 12 <= width <= 14:
        return None
    color = rgb[ink].astype(np.int16).mean(axis=0)
    red = color[0] - max(color[1], color[2])
    if 15 < red < 45:
        return None
    return _center(ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]), red >= 45


def read_suit(rgb: np.ndarray, previous_rgb: np.ndarray) -> str | None:
    """Read a stable native PokerGO suit pair; None means unsupported/uncertain.

    Both frames must independently match a calibrated shape and agree. Colour
    rules out opposite-colour shapes; black cap/lobe structure rules out
    incompatible shapes. Neither replaces full-template matching.
    """
    current, previous = _glyph(rgb), _glyph(previous_rgb)
    if current is None or previous is None or current[1] != previous[1]:
        return None
    if np.mean(current[0] != previous[0]) > 0.025:
        return None
    answers = []
    for mask, red in (current, previous):
        widths = np.count_nonzero(mask, axis=1)[1:16]
        club_lobes = widths[4] < min(widths[2], widths[3]) and widths[6] >= widths[4] + 5
        spade_cap = (widths[0] <= 4 and widths[4] >= 2 * widths[0]
                     and np.all(np.diff(widths[:7]) >= 0))
        eligible = ("h", "d") if red else tuple(
            suit for suit, visible_shape in (("c", club_lobes), ("s", spade_cap)) if visible_shape)
        if not eligible:
            return None
        scores = sorted((min(float(np.mean(mask != np.roll(variant, shift, axis=1)))
                             for variant in ((template, _CLUB_RASTER) if suit == "c" and club_lobes else (template,))
                             for shift in (-1, 0, 1)), suit)
                        for suit, template in _TEMPLATES.items() if suit in eligible)
        if scores[0][0] > 0.07 or (len(scores) > 1 and scores[1][0] - scores[0][0] < 0.08):
            return None
        suit = scores[0][1]
        answers.append(suit)
    return answers[0] if answers[0] == answers[1] else None
