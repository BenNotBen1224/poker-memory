"""GTO estimate for a heads-up river decision in a recorded hand.

Ranges come from one simple rule applied to the hand's own betting, never from anyone's real
cards: both players start with every hand, and a call keeps only the hands with enough equity,
against the other player's range at that moment, to pay the price. Bets, raises and checks keep all.
"""

from __future__ import annotations

import time

import numpy as np
from pydantic import BaseModel

from .engine import Step
from .handrank import COMBOS, board_strength, card, combo, combo_text, showdown_counts, touching
from .solver import IP, OOP, River, Sizes

TOOL_VERSION = "gto-river-1"
SAMPLES = 50  # random runouts per call before the turn; a turn call uses every river card
SEED = 7


class NotSolved(Exception):
    pass


class RiverGTO(BaseModel):
    hand: str
    moves: list[str]
    mix: list[float]  # how often perfect play picks each move with this exact hand
    ev: list[float]  # average chips back, counted from the start of the river, after each move
    taken: int
    best: int
    ev_lost: float
    beats_you: float | None  # facing a bet: share of his betting hands that beat yours
    exploitability: float  # share of the pot a perfect opponent could still win against the solution
    iterations: int
    seconds: float
    range_hands: dict[int, int]  # seat -> hands in his river range
    hero_in_range: bool
    bet_sizes: list[int]
    pot: int  # in the middle at the start of the river
    tool_version: str = TOOL_VERSION


def range_equity(theirs: np.ndarray, board: list[int], rng: np.random.Generator) -> np.ndarray:
    """Equity of every combo against the weights `theirs`, over the rest of the board."""
    rest = [c for c in range(52) if c not in board]
    if len(board) == 5:
        runouts = [board]
    elif len(board) == 4:
        runouts = [board + [c] for c in rest]
    else:
        runouts = [board + rng.choice(rest, 5 - len(board), replace=False).tolist() for _ in range(SAMPLES)]
    won, faced = np.zeros(len(COMBOS)), np.zeros(len(COMBOS))
    for full in runouts:
        live = ~touching(full)
        win, tie, total = showdown_counts(board_strength(full), theirs * live)
        won += (win + tie / 2) * live
        faced += total * live
    return won / np.maximum(faced, 1e-12)


def default_ranges(steps: list[Step], seats: list[int]) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(SEED)
    weights = {s: np.ones(len(COMBOS)) for s in seats}
    for st in steps:
        if st.kind != "action" or st.verb != "call" or st.seat not in seats:
            continue
        b = st.before
        paid = min(b.to_call, b.stacks[st.seat])
        other = seats[1 - seats.index(st.seat)]
        eq = range_equity(weights[other], [card(c) for c in b.board], rng)
        weights[st.seat] = weights[st.seat] * (eq >= paid / (b.pot + paid))
    return weights


def river_gto(steps: list[Step], i: int, hero: int, hero_cards: list[str], seats: list[int]) -> RiverGTO:
    """Solve the river for the decision at steps[i]. Raises NotSolved with a plain reason."""
    start = time.perf_counter()
    step = steps[i]
    if step.before.street != "river" or step.kind != "action" or step.seat != hero:
        raise NotSolved("only your own river decisions are solved so far")
    folded = {s.seat for s in steps[:i] if s.verb == "fold"}
    live_seats = [s for s in sorted(seats) if s not in folded]
    if len(live_seats) != 2:
        raise NotSolved("only heads-up rivers are solved so far, and more than two players were still in")
    # The river card's own step still shows the turn board, so start at the first river action.
    first = next(n for n, s in enumerate(steps) if s.kind == "action" and s.before.street == "river")
    river = steps[first].before
    stack = min(river.stacks[s] for s in live_seats)
    if stack <= 0:
        raise NotSolved("someone was already all-in before the river")
    line = [s for s in steps[first:i + 1] if s.kind == "action"]
    oop = line[0].seat
    ip = live_seats[1 - live_seats.index(oop)]
    role = {oop: OOP, ip: IP}

    def put(st: Step) -> int:
        return st.after.committed[st.seat] - river.committed[st.seat]

    board = [card(c) for c in river.board]
    ranges = default_ranges(steps[:first], live_seats)
    range_hands = {s: int(((ranges[s] > 0) & ~touching(board)).sum()) for s in live_seats}
    mine = combo(hero_cards)
    hero_in_range = bool(ranges[hero][mine] > 0)
    ranges[hero][mine] = max(ranges[hero][mine], 1e-3)  # so the solver still has a plan for your hand
    sizes = tuple(sorted({put(s) for s in line if s.verb in ("bet", "raise", "allin")}))
    game = River(board, river.pot, stack, (ranges[oop], ranges[ip]), Sizes(extra=sizes))

    node, taken = 0, None
    for st in line:
        n = game.nodes[node]
        taken = _move(n.moves, st.verb, put(st)) if n.actor == role[st.seat] else None
        if taken is None:
            raise NotSolved(f"the river line ({st.label}) does not fit the solver's betting tree")
        if st is not step:
            node = n.kids[taken]

    sol = game.solve()
    p, n = role[hero], game.nodes[node]
    j = int(np.flatnonzero(game.hands[p] == mine)[0])
    ev = game.move_values(p, mine, node, sol.strategy)
    best = int(np.argmax(ev))
    beats = None
    if n.put[1 - p] > n.put[p]:
        theirs = game.hands[1 - p]
        weight = game.reach(sol.strategy)[node][1 - p] * ~touching([card(c) for c in hero_cards])[theirs]
        score = board_strength(board)
        beats = float(weight @ (score[theirs] > score[mine]) / weight.sum())
    return RiverGTO(hand=combo_text(mine), moves=[_label(v, to, stack) for v, to in n.moves],
                    mix=[float(x) for x in sol.strategy[node][j]], ev=ev, taken=taken, best=best,
                    ev_lost=ev[best] - ev[taken], beats_you=beats, exploitability=sol.exploitability,
                    iterations=sol.iterations, seconds=time.perf_counter() - start, range_hands=range_hands,
                    hero_in_range=hero_in_range, bet_sizes=[to for v, to in game.nodes[0].moves if v == "bet"],
                    pot=river.pot)


def _move(moves: list[tuple[str, int]], verb: str, to: int) -> int | None:
    """Which tree move a recorded move is. Calls, bets, raises and all-ins match on the amount."""
    for k, (v, x) in enumerate(moves):
        if verb in ("check", "fold") and v == verb:
            return k
        if verb not in ("check", "fold") and v not in ("check", "fold") and x == to:
            return k
    return None


def _label(verb: str, to: int, stack: int) -> str:
    text = {"check": "check", "fold": "fold", "call": "call", "bet": f"bet {to:,}", "raise": f"raise to {to:,}"}[verb]
    return text + (" (all-in)" if to == stack and verb in ("call", "bet", "raise") else "")
