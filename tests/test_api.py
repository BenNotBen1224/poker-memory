import pytest
from fastapi.testclient import TestClient

from conftest import EXAMPLES, hand
from poker_memory.api import create_app

KEY = "2026-10-31-home-game"


@pytest.fixture()
def client(tmp_path, doc):
    app = create_app(tmp_path / "data")
    app.state.memory.import_session(doc, EXAMPLES)
    return TestClient(app)


def test_sessions_and_hands(client):
    [s] = client.get("/api/sessions").json()
    assert (s["key"], s["status"], s["hand_count"], s["verified"], s["consent"]) == (
        KEY, "live", 3, 3, "all_participants")
    rows = client.get(f"/api/sessions/{KEY}/hands").json()
    assert [(r["index"], r["combo"], r["status"]) for r in rows] == [(7, "AQo", "verified"), (8, "KK", "verified"),
                                                                      (9, "AQo", "verified")]
    h = client.get(f"/api/sessions/{KEY}/hands/7").json()
    fold = next(s for s in h["replay"]["steps"] if s["seq"] == 14)
    assert (fold["label"], fold["before"]["to_call"]) == ("Hero folds", 55)


def test_ask_is_refused_until_the_session_ends(client):
    assert client.post("/api/ask", json={"question": "hand 7", "session_key": KEY}).json()["kind"] == "refused"
    assert client.post(f"/api/sessions/{KEY}/end").json()["status"] == "ended"
    a = client.post("/api/ask", json={"question": "hand 7", "session_key": KEY}).json()
    assert (a["kind"], a["hand"]) == ("analysis", 7)
    turns = client.get(f"/api/conversations/{a['conversation_id']}").json()["turns"]
    assert len(turns) == 1


def test_correction_conflicts_and_review(client, doc):
    record = hand(doc, 7).model_copy(update={"result": None})
    bad = record.model_copy(update={"events": [e.replace("raise 75", "raise 21") for e in record.events]})
    r = client.put(f"/api/sessions/{KEY}/hands/7",
                   json={"base_revision": 1, "note": "typo", "record": bad.model_dump(mode="json")})
    assert r.status_code == 200 and (r.json()["status"], r.json()["active_revision"]) == ("invalid", 1)
    stale = client.put(f"/api/sessions/{KEY}/hands/7",
                       json={"base_revision": 1, "note": "x", "record": record.model_dump(mode="json")})
    assert stale.status_code == 409
    revs = client.get(f"/api/sessions/{KEY}/hands/7/revisions").json()
    assert [(v["revision"], v["status"]) for v in revs] == [(1, "verified"), (2, "invalid")]
    assert revs[1]["issues"][0]["code"] == "illegal_amount"


def test_unknown_things_are_404(client):
    assert client.get("/api/sessions/nope/hands").status_code == 404
    assert client.get(f"/api/sessions/{KEY}/hands/99").status_code == 404


def test_video_supports_range_requests_for_seeking(client):
    full = (EXAMPLES / "sample_session.mp4").read_bytes()
    r = client.get(f"/api/sessions/{KEY}/video", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 100-199/{len(full)}"
    assert r.content == full[100:200]
