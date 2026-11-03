"""Pure, CPU-only poker math. Never runs without the inputs it needs: no range, no equity."""

from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Literal

import pokerkit
from pokerkit import Card, Deck, StandardHighHand
from pydantic import BaseModel

TOOL_VERSION = "odds-1"
Combo = tuple[str, str]


class PotOdds(BaseModel):
    to_call: int
    winnable_if_call: int
    required_equity: float  # to_call / winnable_if_call


def winnable_if_call(committed: dict[int, int], hero: int, to_call: int, hero_stack: int) -> tuple[int, int]:
    """(amount Hero actually calls, pot Hero can win if calling). Each player's
    contribution counts only up to Hero's total; anything above is returned or
    contested in a side pot Hero is not eligible for."""
    call = min(to_call, hero_stack)
    hero_total = committed[hero] + call
    winnable = sum(min(c, hero_total) for seat, c in committed.items() if seat != hero) + hero_total
    return call, winnable


def pot_odds(to_call: int, winnable: int) -> PotOdds:
    if to_call <= 0 or winnable <= to_call:
        raise ValueError("pot odds need a positive call into a larger pot")
    return PotOdds(to_call=to_call, winnable_if_call=winnable, required_equity=to_call / winnable)


@lru_cache(maxsize=None)
def _card(text: str) -> Card:
    return next(iter(Card.parse(text)))


def _all_cards() -> list[str]:
    return [repr(c) for c in Deck.STANDARD]


def parse_range(notation: str) -> list[Combo]:
    """'QQ+, AKs, 77' -> sorted explicit combos. Raises ValueError for unreadable ranges."""
    text = notation.replace(" ", "")
    if not text:
        raise ValueError("empty range")
    try:
        combos = pokerkit.parse_range(*text.split(","))
    except (ValueError, KeyError) as e:
        raise ValueError(f"cannot read range {notation!r}") from e
    if not combos:
        raise ValueError(f"range {notation!r} has no hands")
    return sorted(tuple(sorted(repr(c) for c in combo)) for combo in combos)


def _rank(hole: Combo, board: list[str]) -> StandardHighHand:
    return StandardHighHand.from_game([_card(c) for c in hole], [_card(c) for c in board])


class Partition(BaseModel):
    """Every live villain combo on a complete board, split by result against Hero. Exact; no assumption."""

    beats_hero: list[Combo]
    ties: list[Combo]
    loses: list[Combo]


def river_partition(hero: Combo, board: list[str]) -> Partition:
    if len(board) != 5:
        raise ValueError("a partition needs all five board cards")
    dead = set(hero) | set(board)
    live = [c for c in _all_cards() if c not in dead]
    mine = _rank(hero, board)
    beats, ties, loses = [], [], []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            combo = tuple(sorted((a, b)))
            theirs = _rank(combo, board)
            (beats if theirs > mine else ties if theirs == mine else loses).append(combo)
    return Partition(beats_hero=beats, ties=ties, loses=loses)


class EquityResult(BaseModel):
    equity: float  # win + tie / 2
    win: float
    tie: float
    method: Literal["exact", "monte_carlo", "what_if"]
    trials: int
    std_error: float
    seed: int | None
    combos_used: int
    combos_removed: int
    assumption: str
    tool_version: str = TOOL_VERSION


def equity(hero: Combo, villain_range: str, board: list[str], *, samples: int = 3000, seed: int = 7) -> EquityResult:
    """Hero's equity against an explicit villain range. Exact on the river; seeded
    Monte Carlo earlier. Combos that collide with known cards are removed."""
    combos = parse_range(villain_range)
    dead = set(hero) | set(board)
    live = [c for c in combos if not (set(c) & dead)]
    if not live:
        raise ValueError(f"every combo in {villain_range!r} collides with known cards")
    removed = len(combos) - len(live)
    if len(board) == 5:
        mine = _rank(hero, board)
        wins = ties = 0
        for combo in live:
            theirs = _rank(combo, board)
            wins += mine > theirs
            ties += mine == theirs
        n = len(live)
        return EquityResult(equity=(wins + ties / 2) / n, win=wins / n, tie=ties / n, method="exact", trials=n,
                            std_error=0.0, seed=None, combos_used=n, combos_removed=removed,
                            assumption=f"villain range {villain_range}")
    rng = random.Random(seed)
    deck = [c for c in _all_cards() if c not in dead]
    wins = ties = 0
    for _ in range(samples):
        combo = rng.choice(live)
        rest = [c for c in deck if c not in combo]
        runout = board + rng.sample(rest, 5 - len(board))
        mine, theirs = _rank(hero, runout), _rank(combo, runout)
        wins += mine > theirs
        ties += mine == theirs
    eq = (wins + ties / 2) / samples
    return EquityResult(equity=eq, win=wins / samples, tie=ties / samples, method="monte_carlo", trials=samples,
                        std_error=math.sqrt(eq * (1 - eq) / samples), seed=seed, combos_used=len(live),
                        combos_removed=removed, assumption=f"villain range {villain_range}")


def bluff_share_what_if(share: float) -> EquityResult:
    """River sensitivity scenario: 'he bluffs X%' means X% of his range is hands
    Hero beats and the rest beats Hero. Ties are excluded. Not an estimate of
    the opponent's real range."""
    if not 0 <= share <= 1:
        raise ValueError("bluff share must be between 0% and 100%")
    return EquityResult(equity=share, win=share, tie=0.0, method="what_if", trials=0, std_error=0.0, seed=None,
                        combos_used=0, combos_removed=0,
                        assumption=f"what-if: {share:.0%} of his range is hands you beat, the rest beat you; ties excluded")
