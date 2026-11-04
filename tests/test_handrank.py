import numpy as np
from pokerkit import Card, StandardHighHand

from poker_memory.handrank import (COMBOS, DECK, Showdown, board_strength, card, combo, combo_text, showdown_counts,
                                   strength, touching)


def _pokerkit(hand: list[int]) -> StandardHighHand:
    cards = [next(iter(Card.parse(DECK[c]))) for c in hand]
    return StandardHighHand.from_game(cards[:2], cards[2:])


def _hands(rng: np.random.Generator, deck: list[int], n: int) -> list[list[int]]:
    return [rng.choice(deck, 7, replace=False).tolist() for _ in range(n)]


def test_ranks_seven_card_hands_in_the_same_order_as_pokerkit():
    rng = np.random.default_rng(3)
    full = list(range(52))
    few_ranks = [r * 4 + s for r in (0, 1, 7, 12) for s in range(4)]  # quads and full houses
    two_suits = [r * 4 + s for r in range(13) for s in (0, 1)]  # flushes and straight flushes
    low = [r * 4 + s for r in (0, 1, 2, 3, 4, 12) for s in range(4)]  # wheels and small straights
    hands = (_hands(rng, full, 800) + _hands(rng, few_ranks, 300) + _hands(rng, two_suits, 300)
             + _hands(rng, low, 300))
    ours = strength(np.array(hands))
    theirs = [_pokerkit(h) for h in hands]
    order = np.argsort(ours, kind="stable")
    for i, j in zip(order[:-1], order[1:]):
        assert (ours[i] == ours[j]) == (theirs[i] == theirs[j]), (hands[i], hands[j])
        assert theirs[i] <= theirs[j], (hands[i], hands[j])
    categories = {int(s) // 13**5 for s in ours}
    assert categories == set(range(9))


def test_named_hands():
    def score(text: str) -> int:
        return int(strength(np.array([card(text[i:i + 2]) for i in range(0, 14, 2)])))

    wheel, six_high = score("Ah2c3d4s5h9cKd"), score("2c3d4s5h6hJcKd")
    assert six_high > wheel > score("AhAcKdQsJh9c2d")
    assert score("KdKsAh5h7sJcKc") < score("QcTdAh5h7sJcKc")  # the real river: set of kings loses to the straight
    assert combo_text(combo(["Ks", "Kd"])) == "KsKd"


def test_board_shortcut_matches_full_hands():
    board = [card(c) for c in ("9h", "Th", "Jh", "2h", "2c")]
    live = np.flatnonzero(~touching(board))
    full = strength(np.concatenate([COMBOS[live], np.tile(board, (len(live), 1))], axis=1))
    assert (board_strength(board)[live] == full).all()


def test_showdown_counts_skip_hands_that_share_a_card():
    rng = np.random.default_rng(5)
    board = [card(c) for c in ("Ah", "5h", "7s", "Jc", "Kc")]
    weights = rng.random(len(COMBOS)) * ~touching(board)
    score = board_strength(board)
    win, tie, faced = showdown_counts(score, weights)
    for i in rng.choice(np.flatnonzero(~touching(board)), 40, replace=False):
        clear = ~touching(COMBOS[i]) & (weights > 0)
        assert np.isclose(win[i], weights[clear & (score < score[i])].sum())
        assert np.isclose(tie[i], weights[clear & (score == score[i])].sum())
        assert np.isclose(faced[i], weights[clear].sum())


def test_showdown_counts_many_columns_at_once():
    rng = np.random.default_rng(6)
    board = [card(c) for c in ("Ah", "5h", "7s", "Jc", "Kc")]
    weights = rng.random((len(COMBOS), 3)) * ~touching(board)[:, None]
    many = Showdown(board_strength(board)).counts(weights)
    for k in range(3):
        for got, want in zip(many, showdown_counts(board_strength(board), weights[:, k])):
            assert np.allclose(got[:, k], want)
    assert np.allclose(Showdown.faced(weights), many[2])
