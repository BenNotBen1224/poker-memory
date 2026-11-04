import re

import pytest

from conftest import EXAMPLES, hand
from poker_memory.memory import PokerMemory
from poker_memory.model import SessionDoc

KEY = "2026-10-31-home-game"
FOLD_Q = "Was my river fold to the raise reasonable?"


def fact(answer, name):
    return next(f for f in answer.facts if f.name == name)


@pytest.fixture()
def memory(tmp_path, doc):
    m = PokerMemory.open(tmp_path / "data")
    m.import_session(doc, EXAMPLES)
    return m


@pytest.fixture()
def ended(memory):
    memory.end_session(KEY)
    return memory


def test_live_session_is_refused(memory):
    a = memory.ask(FOLD_Q, KEY)
    assert a.kind == "refused" and "after a session has ended" in a.text
    assert a.conversation_id is None and not a.facts


def test_import_copies_the_consented_video(memory):
    assert memory.video_path(KEY).read_bytes() == (EXAMPLES / "sample_session.mp4").read_bytes()


def test_end_to_end_river_fold(ended, doc, tmp_path):
    a = ended.ask(FOLD_Q, KEY)
    assert (a.kind, a.hand, a.revision, a.decision.seq, a.decision.street) == ("analysis", 7, 1, 14, "river")
    assert (fact(a, "pot_odds").kind, fact(a, "pot_odds").value) == ("derived", 0.3073)
    assert fact(a, "equity").kind == "unknown"
    assert fact(a, "gto").kind == "estimated"  # heads-up river decision, so the solver answers
    assert ("gto_river", 1) in {(t.tool, t.revision) for t in a.tools}
    assert (fact(a, "villain_time").kind, fact(a, "villain_time").value) == ("derived", [16401, 17499])
    assert "16.4-17.5 s" in fact(a, "villain_time").text
    assert [e.start_ms for e in a.evidence] == [98000, 115400, 141000]
    assert a.video_url == f"/api/sessions/{KEY}/video"
    assert "verdict" not in [f.name for f in a.facts]  # no verdict without an equity assumption

    b = ended.ask("What if he bluffs 10%?", conversation_id=a.conversation_id)
    assert (b.hand, fact(b, "equity").kind, fact(b, "equity").value) == (7, "estimated", 0.1)
    assert fact(b, "verdict").value == "fold"
    assert "the fold was reasonable: you needed 30.7% and had 10.0%" in fact(b, "verdict").text
    assert any("ties excluded" in s for s in b.assumptions)
    assert {(t.tool, t.revision) for t in b.tools} >= {("bluff_what_if", 1), ("pot_odds", 1)}

    # Correct the raise from 75 to 95; the next answer uses revision 2 and the earlier assumption.
    events = [e.replace("s3 raise 75", "s3 raise 95") for e in hand(doc, 7).events]
    fixed = ended.correct_hand(KEY, 7, 1, hand(doc, 7).model_copy(update={"events": events, "result": None}),
                               "raise was 95")
    assert (fixed.revision, fixed.active_revision) == (2, 2)
    c = ended.ask("And now?", conversation_id=a.conversation_id)
    assert (c.revision, fact(c, "pot_odds").value) == (2, 0.3425)
    assert "Calling cost 75 to win a pot of 219" in fact(c, "pot_odds").text
    assert c.assumptions[0] == "Using the assumption from your earlier question."
    assert {t.revision for t in c.tools} == {2}

    # Everything survives a restart.
    reopened = PokerMemory.open(tmp_path / "data")
    turns = reopened.store.turns(a.conversation_id)
    assert [t["answer"]["revision"] for t in turns] == [1, 1, 2]
    assert [r.tool for r in reopened.store.tool_runs(turns[2]["id"])] == [
        "replay", "pot_odds", "bluff_what_if", "gto_river", "timing"]


def test_ambiguous_search_returns_candidates_then_follow_up_picks(ended):
    a = ended.ask("the AQ hand", KEY)
    assert a.kind == "candidates" and [c.index for c in a.candidates] == [7, 9]
    b = ended.ask("the second one", conversation_id=a.conversation_id)
    assert (b.kind, b.hand, b.decision.label, b.understood) == ("analysis", 9, "Hero calls 30", "choice 2")
    c = ended.ask("#1", conversation_id=a.conversation_id)
    assert (c.hand, c.understood) == (7, "#1")
    assert round(fact(b, "pot_odds").value, 3) == 0.330
    hindsight = fact(b, "hindsight")
    assert hindsight.value == ["Qd", "Js"] and "not used" in hindsight.text
    assert fact(b, "equity").kind == "unknown"  # the shown cards are never used as his range


def test_explicit_river_range_is_exact(ended):
    a = ended.ask("hand 7 if he has 77, AK", KEY)
    eq = fact(a, "equity")
    assert (eq.kind, eq.value) == ("estimated", 0.8)
    assert "exact count over 15 hands" in eq.text
    assert fact(a, "verdict").value == "call"
    assert "a call would have been better" in fact(a, "verdict").text


def test_bluff_what_if_needs_the_river(ended):
    a = ended.ask("hand 8 on the flop if he bluffs 30%", KEY)
    assert a.decision.street == "flop"
    assert fact(a, "equity").kind == "unknown" and "only works on the river" in fact(a, "equity").text


def test_multiway_equity_is_flagged(ended):
    a = ended.ask("hand 8 if he has AK, 99", KEY)
    assert any("2 opponents were still in the hand" in s for s in a.assumptions)
    assert fact(a, "gto").kind == "unknown"


def test_no_match(ended):
    a = ended.ask("my turn fold with pocket kings", KEY)
    assert a.kind == "not_found"


def test_bad_range_is_unknown_not_guessed(ended):
    a = ended.ask("hand 7 if he has 77, AK, AQ", KEY)
    assert fact(a, "equity").kind == "estimated"
    b = ended.ask("hand 7 vs ZZ", KEY)
    assert fact(b, "equity").kind == "unknown"


def test_untimed_hand_has_unknown_timing_and_no_evidence(tmp_path, doc):
    untimed = hand(doc, 7).model_copy(update={
        "events": [re.sub(r"\s*@.*$", "", e) for e in hand(doc, 7).events], "dealt_at": None})
    session = doc.session.model_copy(update={"key": "untimed", "video": None, "consent": "none"})
    m = PokerMemory.open(tmp_path / "d2")
    m.import_session(SessionDoc(session=session, hands=[untimed]))
    m.end_session("untimed")
    a = m.ask(FOLD_Q, "untimed")
    assert fact(a, "villain_time").kind == "unknown"
    assert a.evidence == [] and a.video_url is None


def test_invalid_hand_needs_fix(tmp_path, doc):
    broken = hand(doc, 7).model_copy(update={"events": hand(doc, 7).events[:5] + ["s1 check"]})
    session = doc.session.model_copy(update={"key": "broken", "video": None})
    m = PokerMemory.open(tmp_path / "d3")
    m.import_session(SessionDoc(session=session, hands=[broken]))
    m.end_session("broken")
    a = m.ask("hand 7", "broken")
    assert a.kind == "needs_fix" and "Fix it first" in a.text
