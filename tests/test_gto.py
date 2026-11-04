"""River GTO on the real WPT hand, with ranges from the betting only (never the hole-camera cards)."""

import pytest

from poker_memory.engine import replay
from poker_memory import gto
from poker_memory.gto import NotSolved, RiverGTO, _move, river_gto
from poker_memory.memory import _gto_assumptions, _gto_text
from poker_memory.model import parse_cards

IAN = 1


@pytest.fixture()
def spot(real_doc):
    h = real_doc.hands[0]
    steps = replay(h, real_doc.session.stakes).steps
    return steps, parse_cards(h.hero.cards), [s.seat for s in h.seats]


def test_the_river_fold_to_the_all_in_should_have_been_a_call(spot):
    steps, cards, seats = spot
    assert steps[15].label == "Ian folds"
    g = river_gto(steps, 15, IAN, cards, seats)
    assert g.moves == ["fold", "call (all-in)"]
    assert (g.taken, g.best) == (0, 1)
    assert g.ev[0] == pytest.approx(-800000, abs=5)  # folding gives up the 800,000 bet; 32-bit floats
    assert 3_400_000 < g.ev[1] < 3_600_000
    assert 4_200_000 < g.ev_lost < 4_400_000
    assert 0.08 < g.beats_you < 0.15
    assert g.exploitability <= 0.005
    # Counted on the five-card river board: the river card's own step still shows the turn board.
    assert g.range_hands == {1: 826, 2: 1015}
    assert g.seconds < 3


def test_the_river_bet_was_bigger_than_the_best_size(spot):
    steps, cards, seats = spot
    assert steps[13].label == "Ian bets 800000"
    g = river_gto(steps, 13, IAN, cards, seats)
    assert g.moves == ["check", "bet 465,300", "bet 800,000", "bet 1,057,500", "bet 2,940,000 (all-in)"]
    assert (g.taken, g.best) == (2, 1)
    assert 0 < g.ev_lost < 100_000
    assert g.beats_you is None


def test_spots_the_solver_does_not_cover_say_why(spot):
    steps, cards, seats = spot
    with pytest.raises(NotSolved, match="river"):
        river_gto(steps, 11, IAN, cards, seats)  # Ian calls on the turn
    with pytest.raises(NotSolved, match="river"):
        river_gto(steps, 14, IAN, cards, seats)  # Joe's move, not Ian's
    with pytest.raises(NotSolved, match="heads-up"):
        river_gto(steps[3:], 12, IAN, cards, seats)  # as if the three early folders were still in


def test_a_recorded_all_in_matches_the_tree_move_with_the_same_amount():
    assert _move([("fold", 0), ("call", 2940000)], "allin", 2940000) == 1
    assert _move([("check", 0), ("bet", 465300), ("bet", 2940000)], "allin", 2940000) == 2
    assert _move([("check", 0), ("bet", 465300)], "bet", 800000) is None


def test_a_river_that_began_with_three_players_is_not_solved(spot):
    steps, cards, seats = spot
    # A third player who saw the river folds before Ian acts: the river was not heads-up.
    gone = steps[13].model_copy(update={"seat": 9, "verb": "fold", "label": "Ann folds"})
    three = steps[:13] + [gone] + steps[13:]
    with pytest.raises(NotSolved, match="heads-up"):
        river_gto(three, 14, IAN, cards, seats + [9])


def test_a_warmed_up_server_gives_the_same_answer_as_a_cold_one(spot):
    steps, cards, seats = spot
    gto._EQUITY.clear()
    cold = river_gto(steps, 15, IAN, cards, seats)
    gto._EQUITY.clear()
    gto.warm()
    warm = river_gto(steps, 15, IAN, cards, seats)
    again = river_gto(steps, 13, IAN, cards, seats)  # same hand, so the same ranges come from the cache
    assert warm.range_hands == cold.range_hands == again.range_hands == {1: 826, 2: 1015}
    assert (warm.mix, warm.ev, warm.iterations) == (cold.mix, cold.ev, cold.iterations)


def _estimate(taken: int, best: int, ev: list[float], ev_lost: float) -> RiverGTO:
    return RiverGTO(
        hand="As Qd",
        moves=["bet 800,000", "bet 465,300"],
        mix=[0.5, 0.5],
        ev=ev,
        taken=taken,
        best=best,
        ev_lost=ev_lost,
        beats_you=None,
        exploitability=0.005,
        iterations=1000,
        seconds=0.5,
        range_hands={1: 826, 2: 1015},
        hero_in_range=True,
        bet_sizes=[465300, 800000],
        pot=1_000_000,
    )


def test_small_positive_ev_gap_is_not_collapsed_using_range_averaged_exploitability():
    # This is below the old exploitability * pot threshold, but is still a nonzero
    # model-estimated gap for this exact hand and must not be called equivalent.
    text = _gto_text(_estimate(taken=0, best=1, ev=[1000, 2000], ev_lost=1000), "chips")
    assert "under the default ranges inferred from the betting" in text
    assert "highest estimated-value move was bet 465,300" in text
    assert "your bet 800,000 was estimated at" in text
    assert "as good as" not in text
    assumptions = " ".join(_gto_assumptions(
        _estimate(taken=0, best=1, ev=[1000, 2000], ev_lost=1000), {1: "Opponent", 2: "you"}
    ))
    assert "range-averaged" in assumptions
    assert "not a per-hand error bound" in assumptions
    assert "few percent" not in assumptions


def test_move_with_highest_estimated_value_is_described_without_claiming_optimality():
    text = _gto_text(_estimate(taken=1, best=1, ev=[1000, 2000], ev_lost=0), "chips")
    assert "under the default ranges inferred from the betting" in text
    assert "your bet 465,300 has the highest estimated value" in text
    assert "optimal" not in text
    assert "as good as" not in text