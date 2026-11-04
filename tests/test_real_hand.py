"""The real hand: a WPT final-table clip transcribed from the broadcast overlays.

The 35 MB video is not in git, so a stand-in file is used wherever a video must exist.
"""

import pytest
from fastapi.testclient import TestClient

from poker_memory.api import create_app
from poker_memory.engine import replay
from poker_memory.memory import PokerMemory

KEY = "wpt-rolling-thunder-final-table"
VIDEO = "wpt_rolling_thunder_fold.webm"
FOLD_Q = "Was my river fold to the raise reasonable?"


def fact(answer, name):
    return next(f for f in answer.facts if f.name == name)


@pytest.fixture()
def source_dir(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    (folder / VIDEO).write_bytes(b"stand-in for the real video")
    return folder


@pytest.fixture()
def memory(tmp_path, real_doc, source_dir):
    m = PokerMemory.open(tmp_path / "data")
    m.import_session(real_doc, source_dir)
    m.end_session(KEY)
    return m


def test_pots_stacks_and_result_match_the_broadcast_overlay(real_doc):
    rp = replay(real_doc.hands[0], real_doc.session.stakes)
    assert rp.status == "verified" and rp.issues == []
    # Every pot below is a number the broadcast printed on screen.
    assert [(s.label, s.after.pot) for s in rp.steps] == [
        ("David folds", 140000), ("Ping folds", 140000), ("Andreas folds", 140000),
        ("Ian raises to 160000", 270000), ("Joe calls 100000", 370000), ("Flop Ah 5h 7s", 370000),
        ("Ian bets 150000", 520000), ("Joe calls 150000", 670000), ("Turn Jc", 670000), ("Ian checks", 670000),
        ("Joe bets 370000", 1040000), ("Ian calls 370000", 1410000), ("River Kc", 1410000),
        ("Ian bets 800000", 2210000), ("Joe raises to 2940000 (all-in)", 5150000), ("Ian folds", 0),
    ]
    assert rp.payoffs == {1: -1490000, 2: 1520000, 3: -10000, 4: -10000, 5: -10000}
    fold = rp.steps[-1]
    assert fold.before.stacks == {1: 4175000, 2: 0, 3: 1025000, 4: 740000, 5: 2035000}


def test_everyone_pays_the_ante_before_the_first_action(real_doc):
    first = replay(real_doc.hands[0], real_doc.session.stakes).steps[0]
    assert first.before.pot == 140000
    assert first.before.committed == {1: 40000, 2: 70000, 3: 10000, 4: 10000, 5: 10000}


def test_hole_camera_cards_never_change_the_replay(real_doc):
    record, stakes = real_doc.hands[0], real_doc.session.stakes
    plain = record.model_copy(update={"seats": [s.model_copy(update={"hole_cam": None}) for s in record.seats]})
    assert replay(plain, stakes) == replay(record, stakes)


def test_board_card_that_a_hole_camera_showed_is_refused(real_doc):
    record = real_doc.hands[0]
    events = [e.replace("river Kc", "river Qc") for e in record.events]
    rp = replay(record.model_copy(update={"events": events, "result": None}), real_doc.session.stakes)
    assert rp.status == "invalid" and rp.issues[0].code == "card_reused"


def test_the_river_fold_answer(memory):
    a = memory.ask(FOLD_Q, KEY)
    assert (a.kind, a.hand, a.decision.street, a.decision.label) == ("analysis", 1, "river", "Ian folds")
    assert (fact(a, "hero_cards").value, fact(a, "board").value) == (
        ["Kd", "Ks"], ["Ah", "5h", "7s", "Jc", "Kc"])
    pot_odds = fact(a, "pot_odds")
    assert pot_odds.value == 0.2936
    assert "Calling cost 2140000 to win a pot of 7290000, so a call needed 29.4% equity" in pot_odds.text
    assert fact(a, "equity").kind == "unknown"
    assert fact(a, "villain_time").value == [25001, 26999]
    assert fact(a, "your_time").value == [208001, 209999]
    assert "about 3 min 29 s" in fact(a, "your_time").text
    assert [e.start_ms for e in a.evidence] == [189000, 215000, 424000]
    assert fact(a, "result").value == -1490000
    gto = fact(a, "gto")
    assert gto.kind == "estimated"
    assert "the best move was call (all-in)" in gto.text and "your fold got -800,000 chips" in gto.text
    assert "About 12% of the hands he plays this way beat yours." in gto.text
    assert any("Cards shown later are not used. River hands: you 826, Joe 1015." in s for s in a.assumptions)
    assert ("gto_river", 1) in {(t.tool, t.revision) for t in a.tools}


def test_hole_camera_cards_are_hindsight_only(memory):
    a = memory.ask(FOLD_Q, KEY)
    assert [f.value for f in a.facts if f.name == "hole_cam"] == [["Qc", "Td"]]  # only the player still in the hand
    cam = fact(a, "hole_cam")
    assert cam.kind == "observed" and "Joe holding Qc Td" in cam.text and "not used in any number" in cam.text
    sd = fact(a, "hindsight_showdown")
    assert sd.value == {"hero_hand": "Three of a kind", "villain_hand": "Straight", "winner": "villain"}
    assert "Joe has a straight and you have three of a kind: Joe's hand wins." in sd.text
    assert "Calling here would have lost the pot." in sd.text
    assert {t.tool for t in a.tools} >= {"pot_odds", "showdown"}


def test_bluff_what_if_on_the_real_river(memory):
    a = memory.ask(FOLD_Q, KEY)
    b = memory.ask("What if he bluffs 10%?", conversation_id=a.conversation_id)
    assert (fact(b, "equity").kind, fact(b, "equity").value) == ("estimated", 0.1)
    assert "the fold was reasonable: you needed 29.4% and had 10.0%" in fact(b, "verdict").text
    assert any("19 two-card hands beat you, 0 tie" in s for s in b.assumptions)


def test_session_list_carries_the_source_and_webm_is_served_as_webm(tmp_path, real_doc, source_dir):
    app = create_app(tmp_path / "data")
    app.state.memory.import_session(real_doc, source_dir)
    client = TestClient(app)
    [s] = client.get("/api/sessions").json()
    assert (s["consent"], s["stakes"]["ante"], s["source"]["license"], s["source"]["author"]) == (
        "open_license", 10000, "CC BY 3.0", "World Poker Tour")
    r = client.get(f"/api/sessions/{KEY}/video", headers={"Range": "bytes=0-3"})
    assert (r.status_code, r.headers["content-type"], r.content) == (206, "video/webm", b"stan")
