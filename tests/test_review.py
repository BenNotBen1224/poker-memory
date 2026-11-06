import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from poker_memory.api import create_app
from poker_memory.model import SessionDoc
from poker_memory.review import (
    CASE_SHA256, KEY, build_review, call_advantage, case_digest, decision_from_record,
)
from poker_memory.store import Conflict

REAL = Path(__file__).resolve().parents[1] / "examples" / "real"


@pytest.fixture
def wpt():
    return SessionDoc.load(REAL / "wpt_rolling_thunder.yaml")


@pytest.fixture
def demo(tmp_path, wpt):
    app = create_app(tmp_path / "demo")
    memory = app.state.memory
    memory.import_session(wpt, REAL)
    memory.end_session(KEY)
    return TestClient(app), memory


def test_exact_replay_price_stacks_and_scenario(wpt):
    assert case_digest(wpt) == CASE_SHA256
    d = decision_from_record(wpt.hands[0], wpt.session.stakes)
    assert (d.pot_before_call, d.extra_call, d.called_final_pot) == (5150000, 2140000, 7290000)
    assert d.break_even_share == pytest.approx(214 / 729)
    assert (d.stack_if_fold, d.stack_if_losing_call) == (4175000, 2035000)
    assert d.hero_cards == ["Kd", "Ks"]
    assert d.board == ["Ah", "5h", "7s", "Jc", "Kc"]
    assert d.fold_future_ev == 0
    assert d.scenarios[30].call_advantage == 47000


@pytest.mark.parametrize(("p", "expected"), [
    (0, -2140000), (.1, -1411000), (.2, -682000), (.25, -317500),
    (214 / 729, 0), (.3, 47000), (.4, 776000), (.5, 1505000),
    (.75, 3327500), (1, 5150000),
])
def test_literal_scenarios(p, expected):
    assert call_advantage(p, 2140000, 7290000) == pytest.approx(expected)


@pytest.mark.parametrize("p", [-.01, 1.01, float("nan"), float("inf")])
def test_invalid_scenarios(p):
    with pytest.raises(ValueError):
        call_advantage(p, 2140000, 7290000)


def test_decision_independent_of_camera_and_recorded_outcome(wpt):
    record = wpt.hands[0]
    changed = record.model_copy(update={
        "seats": [s.model_copy(update={"hole_cam": None}) for s in record.seats],
    })
    assert decision_from_record(record, wpt.session.stakes) == decision_from_record(changed, wpt.session.stakes)


def test_read_only_default_response_has_no_hindsight_or_video(demo):
    client, memory = demo
    with patch.object(memory, "ask", side_effect=AssertionError("writeful ask must not be used")):
        response = client.get("/api/demo/wpt/review")
    assert response.status_code == 200
    body = response.json()
    assert body["lens"] == "decision" and "hindsight" not in body
    assert "Qc" not in response.text and "Td" not in response.text and "video_url" not in response.text
    assert len(body["timeline"]) == 16
    assert body["timeline"][14]["label"] == "Joe raises to 2940000 (all-in)"
    assert body["players"][0]["hand_count"] == 1
    assert body["players"][1]["long_run_profile"] == "unknown"
    assert body["reported_context"][0]["kind"] == "attributed_report"
    assert "displayed stacks were wrong" in body["reported_context"][0]["text"]
    assert "not a measured bluff frequency" in body["reported_context"][1]["text"]
    assert any("actual survival" in limit for limit in body["limitations"])
    with sqlite3.connect(memory.store.path) as db:
        for table in ("conversations", "turns", "analysis_runs"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_explicit_broadcast_reveals_exact_outcome_and_seekable_source(demo):
    client, _ = demo
    response = client.get("/api/demo/wpt/review?lens=broadcast")
    assert response.status_code == 200
    h = response.json()["hindsight"]
    assert h["opponent_cards"] == ["Qc", "Td"]
    assert (h["actual_hero_net"], h["known_card_call_net"], h["additional_loss_avoided"]) == (
        -1490000, -3630000, 2140000)
    assert h["opponent_hand"] == "Straight" and h["hero_hand"] == "Three of a kind"
    video = client.get(h["video_url"], headers={"Range": "bytes=0-99"})
    assert video.status_code == 206 and len(video.content) == 100
    assert response.json()["source"]["license"] == "CC BY 3.0"


@pytest.mark.parametrize("change", ["live", "permission", "invalid", "pending", "record", "video"])
def test_refuse_ineligible_changed_or_unavailable_evidence(demo, change):
    client, memory = demo
    session = memory.store.session(KEY)
    hand = memory.store.hand(KEY, 1)
    path = memory.video_path(KEY)
    if change == "live":
        session = session.model_copy(update={"status": "live"})
    elif change == "permission":
        session = session.model_copy(update={"info": session.info.model_copy(update={"consent": "none"})})
    elif change == "invalid":
        hand = hand.model_copy(update={"status": "invalid"})
    elif change == "pending":
        hand = hand.model_copy(update={"latest_revision": 2})
    elif change == "record":
        hand = hand.model_copy(update={"record": hand.record.model_copy(update={"button": 4})})
    else:
        path.write_bytes(b"changed")
    with pytest.raises(Conflict):
        build_review(session, hand, path, "decision")
    if change == "video":
        assert client.get("/api/demo/wpt/review").status_code == 409
        assert client.get("/api/demo/wpt/video").status_code == 409


def test_missing_video_and_missing_session_are_explicit(demo, tmp_path):
    client, memory = demo
    memory.video_path(KEY).unlink()
    assert client.get("/api/demo/wpt/review").status_code == 404
    assert client.get("/api/demo/wpt/video").status_code == 404
    empty = TestClient(create_app(tmp_path / "empty"))
    assert empty.get("/api/demo/wpt/review").status_code == 404
    assert client.get("/api/demo/wpt/review?lens=secret").status_code == 422


def test_unsupported_incomplete_hand_rejected(wpt):
    bad = wpt.hands[0].model_copy(update={"events": wpt.hands[0].events[:-1]})
    with pytest.raises(Conflict, match="complete verified replay"):
        decision_from_record(bad, wpt.session.stakes)
