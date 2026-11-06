import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from poker_memory.api import create_app
from poker_memory.engine import replay
from poker_memory.model import HandRecord, Stakes


def five_players():
    return HandRecord(
        index=1, button=5, hero={"seat": 2, "cards": "AsKs"},
        seats=[{"seat": s, "alias": f"Player {s}", "stack": 1_000_000} for s in range(1, 6)],
        events=["s3 fold", "s4 fold", "s5 fold", "s1 fold"],
    )


def test_big_blind_ante_initial_pot_and_debits():
    result = replay(five_players(), Stakes(sb=50_000, bb=100_000, ante=100_000, ante_kind="big_blind"))
    assert result.status == "verified", result.issues
    initial = result.steps[0].before
    assert initial.pot == 250_000
    assert initial.stacks == {1: 950_000, 2: 800_000, 3: 1_000_000, 4: 1_000_000, 5: 1_000_000}
    assert initial.committed == {1: 50_000, 2: 200_000, 3: 0, 4: 0, 5: 0}
    assert initial.to_call == 100_000
    assert result.payoffs == {1: -50_000, 2: 50_000, 3: 0, 4: 0, 5: 0}


@pytest.mark.parametrize("button,bb", [(2, 5), (5, 2)])
def test_heads_up_big_blind_ante_is_not_charged_to_button(button, bb):
    record = HandRecord(
        index=1, button=button, hero={"seat": bb, "cards": "AsKs"},
        seats=[{"seat": s, "alias": f"Player {s}", "stack": 1_000_000} for s in (2, 5)],
        events=[f"s{button} fold"],
    )
    result = replay(record, Stakes(sb=50_000, bb=100_000, ante=100_000, ante_kind="big_blind"))
    assert result.status == "verified", result.issues
    assert result.steps[0].before.pot == 250_000
    assert result.steps[0].before.committed == {button: 50_000, bb: 200_000}
    assert result.steps[0].before.actor == button
    assert result.steps[0].before.to_call == 50_000
    assert result.payoffs == {button: -50_000, bb: 50_000}


@pytest.mark.parametrize("bb_stack,initial_pot,bb_commit,main_pot,side_pot", [
    (75_000, 125_000, 75_000, 75_000, 400_000),
    (150_000, 200_000, 150_000, 250_000, 300_000),
])
def test_short_big_blind_posts_ante_first_and_can_win_dead_ante(
    bb_stack, initial_pot, bb_commit, main_pot, side_pot,
):
    record = HandRecord(
        index=1, button=3, hero={"seat": 2, "cards": "AsAh"},
        seats=[{"seat": s, "alias": f"Player {s}", "stack": bb_stack if s == 2 else 1_000_000}
               for s in (1, 2, 3)],
        events=["s3 raise 200000", "s1 call", "flop 2c3d7h", "s1 check", "s3 check",
                "turn 9c", "s1 check", "s3 check", "river Jd", "s1 check", "s3 check",
                "s1 show KsKh", "s3 show QsQh"],
    )
    result = replay(record, Stakes(sb=50_000, bb=100_000, ante=100_000, ante_kind="big_blind"))
    assert result.status == "verified", result.issues
    assert result.steps[0].before.pot == initial_pot
    assert result.steps[0].before.stacks[2] == 0
    assert result.steps[0].before.committed == {1: 50_000, 2: bb_commit, 3: 0}
    assert result.steps[1].after.pots == [main_pot, side_pot]
    assert result.payoffs == {1: side_pot - 200_000, 2: main_pot - bb_stack, 3: -200_000}


@pytest.mark.parametrize("ante,expected_pot,committed", [
    (0, 150_000, {1: 50_000, 2: 100_000, 3: 0, 4: 0, 5: 0}),
    (10_000, 200_000, {1: 60_000, 2: 110_000, 3: 10_000, 4: 10_000, 5: 10_000}),
])
def test_legacy_no_ante_and_uniform_ante(ante, expected_pot, committed):
    stakes = Stakes.model_validate_json(f'{{"sb":50000,"bb":100000,"ante":{ante}}}')
    assert stakes.ante_kind == "uniform"
    result = replay(five_players(), stakes)
    assert result.status == "verified", result.issues
    assert result.steps[0].before.pot == expected_pot
    assert result.steps[0].before.committed == committed


def test_invalid_ante_kind_is_rejected():
    with pytest.raises(ValidationError, match="ante_kind"):
        Stakes(sb=50_000, bb=100_000, ante=100_000, ante_kind="guess")


@pytest.mark.parametrize("ante_fields,expected_kind,pot,committed", [
    ({"ante": 100_000, "ante_kind": "big_blind"}, "big_blind", 250_000,
     {"1": 50_000, "2": 200_000, "3": 0, "4": 0, "5": 0}),
    ({"ante": 10_000}, "uniform", 200_000,
     {"1": 60_000, "2": 110_000, "3": 10_000, "4": 10_000, "5": 10_000}),
    ({}, "uniform", 150_000, {"1": 50_000, "2": 100_000, "3": 0, "4": 0, "5": 0}),
])
def test_api_and_sqlite_round_trip_preserve_ante_kind_and_replay(tmp_path, ante_fields, expected_kind, pot, committed):
    path = tmp_path / "data"
    payload = {
        "session": {"key": "bba", "title": "Synthetic BBA regression", "consent": "none",
                    "synthetic": True, "stakes": {"sb": 50_000, "bb": 100_000, **ante_fields}},
        "hands": [five_players().model_dump(mode="json")],
    }
    with TestClient(create_app(path)) as client:
        response = client.post("/api/sessions", json=payload)
        assert response.status_code == 201, response.text
        assert client.get("/api/sessions/bba/hands/1").json()["status"] == "verified"
    with TestClient(create_app(path)) as reopened:
        session = reopened.get("/api/sessions").json()[0]
        assert session["stakes"] == {"sb": 50_000, "bb": 100_000, "ante": ante_fields.get("ante", 0),
                                     "ante_kind": expected_kind, "currency": "chips"}
        stored = reopened.get("/api/sessions/bba/hands/1").json()
        initial = stored["replay"]["steps"][0]["before"]
        assert initial["pot"] == pot
        assert initial["committed"] == committed
        response = reopened.put("/api/sessions/bba/hands/1",
                                json={"base_revision": 1, "record": payload["hands"][0], "note": "replay"})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "verified"
        assert response.json()["replay"]["steps"][0]["before"]["pot"] == pot


def test_api_rejects_invalid_ante_kind(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as client:
        response = client.post("/api/sessions", json={
            "session": {"key": "invalid", "title": "Synthetic regression", "consent": "none",
                        "synthetic": True, "stakes": {"sb": 50_000, "bb": 100_000, "ante": 100_000,
                                                    "ante_kind": "guess"}},
            "hands": [five_players().model_dump(mode="json")],
        })
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "session", "stakes", "ante_kind"]
        assert client.get("/api/sessions").json() == []
