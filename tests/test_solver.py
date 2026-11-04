import numpy as np
import pytest

from poker_memory.handrank import COMBOS, card, combo, touching
from poker_memory.solver import IP, OOP, River, Sizes, build_tree


def _range(*hands: tuple[str, str]) -> np.ndarray:
    weights = np.zeros(len(COMBOS))
    for a, b in hands:
        weights[combo([a, b])] = 1.0
    return weights


def _all(ranks: str, suits_a: str, suits_b: str) -> list[tuple[str, str]]:
    return [(ranks[0] + s, ranks[1] + t) for s in suits_a for t in suits_b if ranks[0] + s != ranks[1] + t]


BOARD = [card(c) for c in ("Kc", "Td", "7h", "4s", "2c")]  # no straight or flush is possible


def test_textbook_river_bluff_and_call_frequencies():
    # The first player holds only a bluff-catcher (AK) and must check. The second has a set of tens
    # (beats AK) or 65 high (loses), one pot-sized bet. Theory: he bets every set, bluffs so that one
    # bet in three is a bluff, and AK calls half the time. With 3 sets and 16 x 9/16 air, sets are
    # a quarter of his range, so he bluffs 1/6 of his air and the first player wins 62.5 of 100.
    oop = _range(*_all("AK", "cdhs", "dhs"))
    tens = _range(("Ts", "Th"), ("Ts", "Tc"), ("Th", "Tc"))
    air = _range(*_all("65", "cdhs", "cdhs"))
    river = River(BOARD, pot=100, stack=100, ranges=(oop, tens + air * 9 / 16),
                  sizes=Sizes(bets=(1.0,), allin=False, max_raises=0, first_can_bet=False))
    sol = river.solve()
    assert sol.exploitability < 0.003

    root = river.nodes[0]
    assert root.moves == [("check", 0)]
    second = root.kids[0]
    assert river.nodes[second].moves == [("check", 0), ("bet", 100)]
    facing = river.nodes[second].kids[1]
    assert river.nodes[facing].moves == [("fold", 0), ("call", 100)]

    tens_combos, air_combos = np.flatnonzero(tens), np.flatnonzero(air)
    assert river.mix(second, sol.strategy, tens_combos)[1] == pytest.approx(1.0, abs=0.01)
    assert river.mix(second, sol.strategy, air_combos)[1] == pytest.approx(1 / 6, abs=0.02)
    assert river.mix(facing, sol.strategy)[1] == pytest.approx(0.5, abs=0.03)
    assert river.move_values(OOP, combo(["As", "Kd"]), 0, sol.strategy) == [pytest.approx(62.5, abs=0.5)]
    assert river.move_values(IP, combo(["Ts", "Th"]), second, sol.strategy) == [
        pytest.approx(100, abs=0.5), pytest.approx(150, abs=1.5)]


def test_tree_keeps_the_real_sizes_and_caps_at_the_stack():
    nodes = build_tree(1_410_000, 2_940_000, Sizes(extra=(800_000, 2_940_000)))
    assert nodes[0].moves == [("check", 0), ("bet", 465_300), ("bet", 800_000), ("bet", 1_057_500),
                              ("bet", 2_940_000)]
    after_bet = nodes[nodes[0].kids[2]]
    assert after_bet.moves == [("fold", 0), ("call", 800_000), ("raise", 2_940_000)]
    facing_allin = nodes[after_bet.kids[2]]
    assert facing_allin.moves == [("fold", 800_000), ("call", 2_940_000)]
    assert all(n.put[0] <= 2_940_000 and n.put[1] <= 2_940_000 for n in nodes)
