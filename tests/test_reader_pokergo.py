import av
import numpy as np
import pytest

from poker_memory.reader import read_hand
from poker_memory.engine import replay
from poker_memory.model import Stakes
from poker_memory.reader.pokergo import (
    Observation, PokerGoObservations, UncertainHand, build_preflop, parse_action, parse_amount, recognizes,
)


@pytest.mark.parametrize("text,value", [
    ("1.575M", 1575000), ("100K", 100000), ("50,000", 50000), ("2.45M", 2450000),
    ("4,650,000", 4650000), ("375000", 375000), ("0.000001M", 1),
    ("4,650000", 4650000), ("1,57,500", None), ("1.575", None), ("1.5.75M", None),
    ("1O0K", None), ("-100", None), ("", None), ("100,,000", None), ("100,", None),
])
def test_decimal_chip_amounts_are_not_repaired(text, value):
    assert parse_amount(text) == value


@pytest.mark.parametrize("text,maximum,expected", [
    ("RAISE TO 1.575M", 2500000, ("raise", 1575000)),
    ("RAISETO1575M", 2500000, None), ("RAISE TO 1.575M", None, None),
    ("RAISE TO 3M", 2500000, None), ("CALL 100K", 2500000, ("call", 100000)),
    ("FOLD", None, ("fold", None)), ("+200K", 2500000, None), ("PING", 2500000, None),
])
def test_action_amounts_respect_observed_stack_bound(text, maximum, expected):
    assert parse_action(text, maximum) == expected


def test_layout_requires_both_labels():
    rgb = np.zeros((480, 854, 3), np.uint8)
    assert recognizes(rgb, lambda crops: ["BLINDS", "BB ANTE"])
    assert not recognizes(rgb, lambda crops: ["BLINDS", "ANTE"])


def test_native_rank_crop_excludes_lower_card_border_and_requires_suit_pixels(monkeypatch):
    from poker_memory.reader.pokergo import observe

    club = np.full((17, 16, 3), 225, np.uint8)
    rows = (".....####.....", "....######....", "....######....", "....######....", ".....#####....",
            "..###.######..", ".############.", ".#############", "##############", ".############.",
            "..###.##.###..", "......##......", "......##......", ".....####.....", "....######....")
    mask = np.array([[char == "#" for char in row] for row in rows])
    club[:15, 1:15][mask] = 35

    rgb = np.zeros((480, 854, 3), np.uint8)
    for top in (298, 370):
        for left in (73, 108):
            rgb[top:top + 18, left:left + 16] = 225
            rgb[top + 3:top + 14, left + 4:left + 9] = 30
            rgb[top + 18:top + 21, left:left + 16] = 30
            suit_top = top + (2 if top == 298 else 1)
            height = 17 if top == 298 else 16
            rgb[suit_top:suit_top + height, left + 16:left + 32] = club[:height]
    frames = [(i, i * 400, "1/11988", i / 30, rgb, rgb) for i in (0, 6, 12)]
    monkeypatch.setattr("poker_memory.reader.pokergo._native_frames", lambda *_: iter(frames))

    def ocr(crops):
        if crops and crops[0].shape == (18, 16):
            assert all(crop.shape == (18, 16) for crop in crops)
            assert all(np.all(crop[-1] > 180) for crop in crops)
            return ["7"] * len(crops)
        # Header/name, stack, position, owner, action calls follow the native catalog order.
        nonlocal calls
        calls += 1
        return ["ALICE"] * len(crops) if calls in {1, 4, 7, 10, 13, 20} else ["1000"] * len(crops)

    calls = 0
    result = observe(None, ocr)
    pairs = [item.value for item in result.observations if item.field == "camera_cards" and item.value]
    assert pairs
    assert all(pair == ["7c", "7c"] for pair in pairs)


def test_rank_padding_retries_only_unsupported_text_without_neighbouring_border():
    from poker_memory.reader.pokergo import _rank_texts

    crops = [np.full((18, 16), 225, np.uint8) for _ in range(3)]
    for crop in crops:
        crop[3:15, 4:9] = 30
    calls = []

    def ocr(images):
        calls.append([image.copy() for image in images])
        if len(calls) == 1:
            return ["9", "U", "?"]
        assert len(images) == 2
        assert all(image.shape == (22, 16) and np.all(image[18:] == 225) for image in images)
        return ["J", "?"]

    assert _rank_texts(crops, ocr) == ["9", "J", "?"]
    assert len(calls) == 2
    assert all(image.shape == (18, 16) for image in calls[0])


@pytest.fixture
def native_video(tmp_path):
    path = tmp_path / "unknown.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 854, 480, "yuv420p"
        for _ in range(7):
            frame = av.VideoFrame.from_ndarray(np.zeros((480, 854, 3), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def test_normal_reader_rejects_unknown_native_layout(native_video, monkeypatch):
    monkeypatch.setattr("poker_memory.reader._ocr", lambda: lambda crops: ["", ""])
    with pytest.raises(ValueError, match="unsupported native"):
        read_hand(native_video)


def test_normal_reader_exposes_unknown_panels_without_a_hand(native_video, monkeypatch):
    calls = 0

    def ocr(crops):
        nonlocal calls
        calls += 1
        return ["BLINDS", "BB ANTE"] if calls == 1 else [""] * len(crops)

    ocr.calls = 0
    monkeypatch.setattr("poker_memory.reader._ocr", lambda: ocr)
    with pytest.raises(UncertainHand) as caught:
        read_hand(native_video)
    evidence = caught.value.evidence.model_dump(mode="json")
    assert evidence["decoded_frames"] == 7
    assert evidence["sampled_frames"] == 2
    assert evidence["review_status"] == "NOT REVIEWED"
    unknown = {o["field"]: o["kind"] for o in evidence["observations"]}
    assert unknown == {"panel_owner_1": "unknown", "panel_owner_2": "unknown", "board": "unknown"}
    assert evidence["stakes"] is None


def test_cli_uncertainty_is_nonzero_without_success_shaped_hand(monkeypatch, capsys):
    import yaml

    from poker_memory.cli import main
    from poker_memory.reader.pokergo import Observation, PokerGoObservations, UncertainHand

    evidence = PokerGoObservations(issues=["complete action sequence missing"], observations=[
        Observation(field="action_label", owner="Player", value="raise 1575000", raw="RAISE TO 1.575M",
                    frame_index=10, pts=4000, time_base="1/11988", earliest_seconds=0.3, latest_seconds=0.4),
    ])

    def uncertain(path):
        raise UncertainHand(evidence)

    monkeypatch.setattr("poker_memory.reader.read_hand", uncertain)
    with pytest.raises(SystemExit) as result:
        main(["read", "development.mp4"])
    assert result.value.code == 2
    output = capsys.readouterr()
    body = yaml.safe_load(output.out)
    assert body["status"] == "uncertain"
    assert body["stage"] == "pokergo_sequence"
    assert body["observations"]["observations"][0]["value"] == "raise 1575000"
    assert "hands" not in body
    assert "complete action sequence missing" in output.err


def preflop_evidence():
    evidence = PokerGoObservations(stakes=Stakes(sb=50, bb=100, ante=100, ante_kind="big_blind"))

    def add(field, owner, value, index):
        evidence.observations.append(Observation(
            field=field, owner=owner, value=value, raw=str(value), frame_index=index, pts=index * 400,
            time_base="1/11988", earliest_seconds=index / 30, latest_seconds=index / 30,
        ))

    for slot, name in enumerate(("Alice", "Bob", "Carol", "Dana", "Evan")):
        add("graphic_slot", name, slot, 0)
        add("displayed_stack", name, 10000, 0)
    for role, name in (("D", "Carol"), ("SB", "Dana"), ("BB", "Evan")):
        add("position", name, role, 0)
    add("camera_cards", "Dana", ["Kd", "6s"], 2)
    add("displayed_pot", None, 250, 3)
    for index, name, action in ((5, "Alice", "fold"), (6, "Bob", "fold"), (7, "Carol", "fold"),
                                (8, "Dana", "raise 1000"), (9, "Evan", "fold")):
        add("action_label", name, action, index)
    add("displayed_pot", None, 1200, 8)
    add("win_label", "Dana", 200, 9)
    return evidence


def test_preflop_builder_uses_explicit_actions_and_shared_replay():
    evidence = preflop_evidence()
    draft = build_preflop(evidence)
    assert draft.record.button == 3
    assert draft.record.hero.model_dump() == {"seat": 4, "cards": "Kd6s"}
    assert draft.record.events == ["s1 fold @00:00.2", "s2 fold @00:00.2", "s3 fold @00:00.2",
                                   "s4 raise 1000 @00:00.3", "s5 fold @00:00.3"]
    checked = replay(draft.record, evidence.stakes)
    assert checked.status == "verified"
    assert checked.steps[0].before.pot == 250
    assert checked.steps[3].after.pot == 1200
    assert checked.payoffs == {1: 0, 2: 0, 3: 0, 4: 200, 5: -200}
    assert checked.board == []
    assert checked.revealed == {}
    assert evidence.observations[-1].kind == "inferred"
    assert evidence.review_status == "NOT REVIEWED"


@pytest.mark.parametrize("change,message", [
    ("missing_fold", "missing/repeated actions"),
    ("call", "calls, checks"),
    ("pot", "displayed pots disagree"),
    ("win", "winner net-gain"),
    ("role", "position markers changed"),
    ("board", "postflop input"),
    ("wrong_raise", "replay rejected"),
    ("wrong_order", "action order"),
])
def test_preflop_builder_rejects_unsupported_or_contradictory_sequences(change, message):
    evidence = preflop_evidence()
    if change == "missing_fold":
        evidence.observations = [o for o in evidence.observations if not (o.field == "action_label" and o.owner == "Bob")]
    elif change == "call":
        next(o for o in evidence.observations if o.field == "action_label" and o.owner == "Bob").value = "call"
    elif change == "pot":
        next(o for o in evidence.observations if o.field == "displayed_pot").value = 650
    elif change == "win":
        next(o for o in evidence.observations if o.field == "win_label").value = 250
    elif change == "role":
        role = next(o for o in evidence.observations if o.field == "position")
        evidence.observations.append(role.model_copy(update={"owner": "Bob", "frame_index": 10}))
    elif change == "board":
        evidence.observations.append(evidence.observations[0].model_copy(update={"field": "board", "value": ["Ah", "2d", "3c"]}))
    elif change == "wrong_raise":
        next(o for o in evidence.observations if o.field == "action_label" and o.owner == "Dana").value = "raise 150"
    elif change == "wrong_order":
        next(o for o in evidence.observations if o.field == "action_label" and o.owner == "Alice").owner = "Bob"
        next(o for o in evidence.observations if o.field == "action_label" and o.owner == "Bob" and o.frame_index == 6).owner = "Alice"
    with pytest.raises(UncertainHand) as caught:
        build_preflop(evidence)
    assert message in caught.value.evidence.issues[-1]
    assert caught.value.evidence.review_status == "NOT REVIEWED"


def test_normal_reader_can_return_justified_native_reading(native_video, monkeypatch):
    class Labels:
        calls = 0

        def __call__(self, crops):
            self.calls += 1
            return ["BLINDS", "BB ANTE"]

    monkeypatch.setattr("poker_memory.reader._ocr", Labels)
    monkeypatch.setattr("poker_memory.reader.pokergo.observe", lambda path, ocr: preflop_evidence())
    reading = read_hand(native_video)
    assert reading.record.hero.cards == "Kd6s"
    assert reading.wins == {4: 200}
    assert replay(reading.record, reading.stakes).payoffs == {1: 0, 2: 0, 3: 0, 4: 200, 5: -200}
    assert reading.observations.status == "preflop_replay_verified"
