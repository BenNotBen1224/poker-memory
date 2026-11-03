import pytest

from poker_memory.odds import (
    bluff_share_what_if,
    equity,
    parse_range,
    pot_odds,
    river_partition,
    winnable_if_call,
)

BOARD = ["Qs", "7c", "2d", "9h", "3c"]
HERO = ("Ah", "Qd")


def test_pot_odds_for_the_river_raise():
    call, winnable = winnable_if_call({1: 1, 2: 34, 3: 89, 4: 0}, hero=2, to_call=55, hero_stack=216)
    assert (call, winnable) == (55, 179)
    assert round(pot_odds(call, winnable).required_equity, 4) == 0.3073


def test_short_stack_can_only_win_what_it_matches():
    # Hero has 40 behind and faces 100 more; only 40 of each opponent's chips are contestable.
    call, winnable = winnable_if_call({1: 10, 2: 10, 3: 110}, hero=2, to_call=100, hero_stack=40)
    assert (call, winnable) == (40, 10 + 50 + 50)


def test_exact_river_equity_against_explicit_range():
    r = equity(HERO, "77,AK", BOARD)
    assert (r.method, r.win, r.tie, r.equity) == ("exact", 0.8, 0.0, 0.8)
    assert (r.combos_used, r.combos_removed) == (15, 7)  # 3 sets beat us, 12 AK lose


def test_ties_count_half():
    r = equity(HERO, "AQ", BOARD)
    assert (r.win, r.tie, r.equity, r.combos_used) == (0.0, 1.0, 0.5, 6)


def test_turn_equity_is_seeded_monte_carlo_and_reproducible():
    a = equity(HERO, "QQ+,AK,77", BOARD[:4], samples=500, seed=3)
    b = equity(HERO, "QQ+,AK,77", BOARD[:4], samples=500, seed=3)
    assert a == b
    assert (a.method, a.trials, a.seed) == ("monte_carlo", 500, 3)
    assert 0 < a.std_error < 0.03


def test_river_partition_is_exhaustive():
    p = river_partition(HERO, BOARD)
    assert (len(p.beats_hero), len(p.ties), len(p.loses)) == (100, 6, 884)
    assert ("7d", "7h") in p.beats_hero
    assert ("As", "Qh") in p.ties


def test_bluff_share_what_if_is_labeled_and_excludes_ties():
    r = bluff_share_what_if(0.10)
    assert (r.method, r.equity, r.tie) == ("what_if", 0.10, 0.0)
    assert "ties excluded" in r.assumption


@pytest.mark.parametrize("bad", ["", "ZZ", "AK,  QQ++"])
def test_unreadable_ranges_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_range(bad)


def test_range_parses_to_explicit_combos():
    assert len(parse_range("QQ+, AKs, 77")) == 6 + 6 + 6 + 4 + 6
