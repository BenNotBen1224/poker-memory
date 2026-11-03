import warnings

import pytest

from conftest import hand
from poker_memory.engine import replay
from poker_memory.model import HandRecord, Stakes, VideoTime

STAKES = Stakes(sb=1, bb=2)


def with_events(record: HandRecord, events: list[str], **extra) -> HandRecord:
    return record.model_copy(update={"events": events, **extra})


def test_river_raise_spot_matches_hand_checked_numbers(doc):
    rp = replay(hand(doc, 7), doc.session.stakes)
    assert rp.status == "verified" and rp.issues == []
    fold = next(s for s in rp.steps if s.seq == 14)
    assert fold.label == "Hero folds"
    assert fold.before.street == "river"
    assert fold.before.board == ["Qs", "7c", "2d", "9h", "3c"]
    assert fold.before.actor == 2
    assert fold.before.pot == 124
    assert fold.before.to_call == 55
    assert fold.before.committed == {1: 1, 2: 34, 3: 89, 4: 0}
    assert fold.before.stacks == {1: 199, 2: 216, 3: 221, 4: 180}


def test_uncalled_raise_is_returned_to_the_raiser(doc):
    rp = replay(hand(doc, 7), doc.session.stakes)
    # Black shirt wins 69 (124 minus the uncalled 55) after putting in 34 that was called.
    assert rp.payoffs == {1: -1, 2: -34, 3: 35, 4: 0}


def test_three_way_all_in_builds_main_and_side_pot(doc):
    rp = replay(hand(doc, 8), doc.session.stakes)
    assert rp.status == "verified", rp.issues
    call = next(s for s in rp.steps if s.seq == 6)
    assert call.label == "Black shirt calls 66"
    assert call.after.pots == [180, 312]
    assert call.after.pot == 492
    assert rp.payoffs == {1: 0, 2: 96, 3: -216, 4: 120}
    assert rp.revealed == {4: ["Ac", "Jd"], 3: ["9s", "9d"], 2: ["Kh", "Ks"]}
    assert rp.board == ["Ah", "8c", "4d", "2s", "Tc"]


def test_short_all_in_raise_label(doc):
    rp = replay(hand(doc, 8), doc.session.stakes)
    assert [s.label for s in rp.steps[:4]] == [
        "Bob raises to 60 (all-in)", "Alice folds", "Hero raises to 150", "Black shirt calls 148",
    ]


def test_showdown_hero_is_shown_automatically(doc):
    rp = replay(hand(doc, 9), doc.session.stakes)
    assert rp.status == "verified"
    assert rp.revealed == {3: ["Qd", "Js"], 2: ["As", "Qh"]}
    assert rp.payoffs == {1: -2, 2: 47, 3: -44, 4: -1}


def test_replay_emits_no_pokerkit_warnings(doc):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for h in doc.hands:
            replay(h, doc.session.stakes)


@pytest.mark.parametrize(
    ("index", "events", "code", "seq"),
    [
        (7, ["s4 fold @00:49"], "out_of_turn", 0),
        (7, ["s3 raise 3"], "illegal_amount", 0),
        (7, ["s3 bet 6"], "bet_when_facing", 0),
        (7, ["s3 check"], "illegal_check", 0),
        (7, ["s3 raise 6", "s4 fold", "s1 fold", "s2 call", "turn 9h"], "wrong_street", 4),
        (7, ["s3 raise 6", "s4 fold", "flop Qs7c2d"], "board_out_of_order", 2),
        (7, ["s3 raise 6", "s4 fold", "s1 fold", "s2 call", "flop AhQs2d"], "card_reused", 4),
        (7, ["s3 raise six"], "unreadable_event", 0),
        (7, ["s3 raise 6", "s9 fold"], "unknown_seat", 1),
        (7, ["s3 raise 6", "s4 fold"], "hand_incomplete", None),
        (9, None, "showdown_missing", None),
    ],
)
def test_invalid_hands_are_reported_not_accepted(doc, index, events, code, seq):
    base = hand(doc, index)
    if events is None:  # drop the villain's showdown reveal
        events = base.events[:-1]
    rp = replay(with_events(base, events), doc.session.stakes)
    assert rp.status == "invalid"
    assert rp.payoffs is None
    assert (rp.issues[0].code, rp.issues[0].seq) == (code, seq)


def test_replay_stops_at_first_error_and_says_so(doc):
    rp = replay(with_events(hand(doc, 7), ["s3 raise 6", "s1 fold", "s4 fold"]), doc.session.stakes)
    assert [i.code for i in rp.issues] == ["out_of_turn", "replay_stopped"]
    assert [s.seq for s in rp.steps] == [0]


def test_recorded_result_must_match_replay(doc):
    rp = replay(hand(doc, 7).model_copy(update={"result": {3: 40}}), doc.session.stakes)
    assert rp.status == "invalid"
    assert rp.issues[0].code == "result_mismatch"
    assert rp.issues[0].message == "recorded result for seat 3 is +40, replay gives +35"


def test_heads_up_button_posts_small_blind_and_acts_first():
    record = HandRecord(
        index=1, button=5,
        seats=[{"seat": 2, "alias": "Hero", "stack": 100}, {"seat": 5, "alias": "Dana", "stack": 100}],
        hero={"seat": 2, "cards": "AsKs"},
        events=["s5 call", "s2 check", "flop 2c3d4h", "s2 check", "s5 bet 4", "s2 fold"],
    )
    rp = replay(record, STAKES)
    assert rp.status == "verified", rp.issues
    assert rp.steps[0].before.committed == {2: 2, 5: 1}
    assert rp.payoffs == {2: -2, 5: 2}


@pytest.mark.parametrize(
    ("text", "earliest", "latest"),
    [("00:47", 47_000, 47_999), ("01:55.4", 115_400, 115_499), ("01:55.45", 115_450, 115_459), ("1:02:03", 3_723_000, 3_723_999)],
)
def test_written_time_precision_sets_uncertainty(text, earliest, latest):
    t = VideoTime.parse(text)
    assert (t.earliest_ms, t.latest_ms) == (earliest, latest)
