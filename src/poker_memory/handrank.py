"""Fast hand strength for whole ranges at once. Bigger is better; equal means a tie.

Cards are numbers 0..51 (rank * 4 + suit). A combo is two cards; COMBOS lists all 1326.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache

import numpy as np

RANKS, SUITS = "23456789TJQKA", "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]
_INDEX = {c: i for i, c in enumerate(DECK)}
COMBOS = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)])
_COMBO_INDEX = {(a, b): i for i, (a, b) in enumerate(COMBOS.tolist())}
_HOLDING = np.array([np.flatnonzero((COMBOS == c).any(axis=1)) for c in range(52)])  # (52, 51) combos per card
_CATEGORY = 13**5
_SHIFT = np.arange(52)[:, None] * 10 * _CATEGORY  # keeps each card's sorted scores in its own band


def card(text: str) -> int:
    return _INDEX[text[0].upper() + text[1].lower()]


def combo(cards: Iterable[str]) -> int:
    a, b = sorted(card(c) for c in cards)
    return _COMBO_INDEX[(a, b)]


def combo_text(i: int) -> str:
    a, b = COMBOS[i]
    return DECK[b] + DECK[a]


def touching(cards: Iterable[int]) -> np.ndarray:
    """True for every combo that uses one of these cards."""
    used = np.zeros(52, dtype=bool)
    used[list(cards)] = True
    return used[COMBOS].any(axis=1)


@cache
def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lookups over 13-bit rank sets: highest rank, top-k ranks as base-13 digits, straight top, bit count."""
    high = np.full(1 << 13, -1)
    top = np.zeros((6, 1 << 13), dtype=np.int64)
    straight = np.full(1 << 13, -1)
    bits = np.zeros(1 << 13, dtype=np.int64)
    for m in range(1, 1 << 13):
        ranks = [r for r in range(12, -1, -1) if m >> r & 1]
        high[m], bits[m] = ranks[0], len(ranks)
        padded = ranks + [0] * 5
        for k in range(1, 6):
            value = 0
            for r in padded[:k]:
                value = value * 13 + r
            top[k, m] = value
        for t in range(12, 3, -1):
            if (m >> (t - 4)) & 0b11111 == 0b11111:
                straight[m] = t
                break
        else:
            if m & 0b1000000001111 == 0b1000000001111:  # A-2-3-4-5
                straight[m] = 3
    return high, top, straight, bits


_RANK_ONE = np.eye(13, dtype=np.int64)[np.arange(52) >> 2]  # card -> one-hot rank
_SUIT_BIT = np.zeros((52, 4), dtype=np.int64)  # card -> its rank bit, in its suit's column
_SUIT_BIT[np.arange(52), np.arange(52) & 3] = 1 << (np.arange(52) >> 2)


def _bit(rank: np.ndarray) -> np.ndarray:
    return np.where(rank >= 0, 1 << np.maximum(rank, 0), 0)


def _score(counts: np.ndarray, suit_ranks: np.ndarray) -> np.ndarray:
    """counts: (n, 13) cards per rank. suit_ranks: (n, 4) rank bits held in each suit."""
    high, top, straight, bits = _tables()
    weights = 1 << np.arange(13)
    present, pairs = (counts > 0) @ weights, (counts >= 2) @ weights
    trips, quads = (counts >= 3) @ weights, (counts == 4) @ weights
    in_suit = bits[suit_ranks]
    flush = in_suit.max(axis=1) >= 5
    fmask = np.where(flush, suit_ranks[np.arange(len(suit_ranks)), in_suit.argmax(axis=1)], 0)

    q, t = high[quads], high[trips]
    full_pair = high[pairs & ~_bit(t)]
    p1 = high[pairs]
    p2 = high[pairs & ~_bit(p1)]
    sf, st = straight[fmask], straight[present]
    b = _CATEGORY
    return np.select(
        [sf >= 0, q >= 0, (t >= 0) & (full_pair >= 0), flush, st >= 0, t >= 0, p2 >= 0, p1 >= 0],
        [8 * b + sf,
         7 * b + q * 13 + high[present & ~_bit(q)],
         6 * b + t * 13 + full_pair,
         5 * b + top[5][fmask],
         4 * b + st,
         3 * b + t * 169 + top[2][present & ~_bit(t)],
         2 * b + p1 * 169 + p2 * 13 + high[present & ~_bit(p1) & ~_bit(p2)],
         1 * b + p1 * 2197 + top[3][present & ~_bit(p1)]],
        default=top[5][present],
    )


def strength(cards: np.ndarray) -> np.ndarray:
    """Score hands of 5 to 7 different cards. cards has shape (..., k); the result has shape (...)."""
    c = np.asarray(cards, dtype=np.int64)
    lead = c.shape[:-1]
    c = c.reshape(-1, c.shape[-1])
    return _score(_RANK_ONE[c].sum(axis=1), np.bitwise_or.reduce(_SUIT_BIT[c], axis=1)).reshape(lead)


def board_strength(board: Iterable[int]) -> np.ndarray:
    """Strength of every combo on a full five-card board (meaningless for combos touching the board)."""
    board = list(board)
    a, b = COMBOS[:, 0], COMBOS[:, 1]
    counts = _RANK_ONE[board].sum(axis=0) + _RANK_ONE[a] + _RANK_ONE[b]
    return _score(counts, np.bitwise_or.reduce(_SUIT_BIT[board], axis=0) | _SUIT_BIT[a] | _SUIT_BIT[b])


class Showdown:
    """For every combo on one board: the opponent weight it beats, ties and faces in total, leaving
    out opponent combos that share a card with it. Opponent combos touching the board must have zero
    weight. The sorting depends only on the board, so it is done once; `counts` takes weights of
    shape (1326,) or (1326, k) and answers for all k columns at once."""

    def __init__(self, score: np.ndarray):
        self.order = np.argsort(score, kind="stable")
        ranked = score[self.order]
        self.below = np.searchsorted(ranked, score, side="left")
        self.upto = np.searchsorted(ranked, score, side="right")
        # The same counts inside the 51 combos holding each card, to take out the ones that share a card.
        by_card = score[_HOLDING]
        sort = np.argsort(by_card, axis=1, kind="stable")
        self.card_sort = np.take_along_axis(_HOLDING, sort, axis=1)
        shifted = (np.take_along_axis(by_card, sort, axis=1) + _SHIFT).ravel()
        # +card skips one leading zero per earlier row of the (52, 52) running totals
        self.left = [np.searchsorted(shifted, score + _SHIFT[c, 0], side="left") + c for c in COMBOS.T]
        self.right = [np.searchsorted(shifted, score + _SHIFT[c, 0], side="right") + c for c in COMBOS.T]
        self.last = [c * 52 + 51 for c in COMBOS.T]

    def counts(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = weights.shape[1:]
        total = np.concatenate([np.zeros((1, *cols), weights.dtype), np.cumsum(weights[self.order], axis=0)])
        card_total = np.concatenate([np.zeros((52, 1, *cols), weights.dtype),
                                     np.cumsum(weights[self.card_sort], axis=1)], axis=1).reshape(52 * 52, *cols)

        def shared(at: list[np.ndarray]) -> np.ndarray:
            return card_total[at[0]] + card_total[at[1]]

        win = total[self.below] - shared(self.left)
        not_lose = total[self.upto] - shared(self.right) + weights  # the combo itself sits in its own tie group
        faced = total[-1] - shared(self.last) + weights
        return win, not_lose - win, faced

    @staticmethod
    def faced(weights: np.ndarray) -> np.ndarray:
        """Only the third result of `counts`, without ranking anything: all weight minus the combos
        sharing a card (the combo itself is taken out twice, so it is added back once)."""
        by_card = weights[_HOLDING].sum(axis=1)
        return weights.sum(axis=0) - by_card[COMBOS[:, 0]] - by_card[COMBOS[:, 1]] + weights


def showdown_counts(score: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return Showdown(score).counts(weights)
