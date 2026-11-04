"""The fast reader: its text parsers, its exact shortcuts, the hand builder, and the real clip end to end.

The real video is not in git (scripts/fetch_real_video.py gets it), so the end-to-end test skips without it.
"""

import io
from contextlib import redirect_stdout

import pytest
import yaml

np = pytest.importorskip("numpy")
pytest.importorskip("av")

from conftest import EXAMPLES  # noqa: E402
from poker_memory.cli import main  # noqa: E402
from poker_memory.engine import replay  # noqa: E402
from poker_memory.model import HandRecord, Stakes, VideoTime  # noqa: E402
from poker_memory.reader import read_hand  # noqa: E402
from poker_memory.reader.hand import BLANK, Moment, Seen, build  # noqa: E402
from poker_memory.reader.overlay import (  # noqa: E402
    SAME,
    Catalog,
    _percentiles,
    parse_blinds,
    parse_rank,
    parse_status,
)
from poker_memory.reader.video import Clip  # noqa: E402

VIDEO = EXAMPLES / "real" / "wpt_rolling_thunder_fold.webm"


@pytest.mark.parametrize("text, expected", [
    ("TO CALL 30,000", ("TOCALL", 30000)),
    ("T0 CALL 3O,000", ("TOCALL", 30000)),
    ("RAISE TO 160,000", ("RAISETO", 160000)),
    ("ALL IN", ("ALLIN", None)),
    ("CHECK", ("CHECK", None)),
    ("WIN +1,520,000", ("WIN", 1520000)),
    ("PING", None),
])
def test_status_lines(text, expected):
    assert parse_status(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("K", "K"), ("q", "Q"), ("10", "T"), ("1O", "T"), ("0", "Q"), ("A.", "A"), ("X", None), ("", None),
])
def test_card_ranks(text, expected):
    assert parse_rank(text) == expected


@pytest.mark.parametrize("blinds, ante, expected", [
    ("SB/BB 30K/60K", "ANTE 10K", (30000, 60000, 10000)),
    ("SB/BB 1.5M/3M", "ANTE 500K", (1500000, 3000000, 500000)),
    ("SB/BB 30K/6OK", "", (30000, 60000, 0)),
    ("SB/BB ?", "ANTE 10K", None),
])
def test_blinds(blinds, ante, expected):
    assert parse_blinds(blinds, ante) == expected


def test_percentiles_are_exactly_numpys():
    rng = np.random.default_rng(7)
    for shape in [(5, 9, 13), (3, 20, 31), (4, 1, 2), (2, 24, 64)]:
        crops = rng.integers(0, 256, shape, dtype=np.uint8)
        for q in (10, 50, 75, 90, 95, 99):
            assert _percentiles(crops, q).tolist() == [np.percentile(c, q) for c in crops]


def _band(a, b) -> int:
    """The most differing pixels in any 6-pixel-wide column band, counted plainly."""
    columns = (a != b).sum(axis=0)
    return max(int(columns[i : i + 6].sum()) for i in range(len(columns) - 5))


def test_catalog_numbers_crops_as_a_plain_search_would():
    """The catalog's shortcuts (a repeat of the last bytes, the hint, bit-packed columns) give the same numbers and
    keep the same best crops as a plain search that tries the hint, then every numbered crop."""
    rng = np.random.default_rng(3)
    texts = rng.random((6, 14, 40)) < 0.3
    showing = {place: int(rng.integers(6)) for place in range(4)}
    catalog, last = Catalog(), {}
    firsts, best, ink, count = [], [], [], []  # the plain search's own record of each number
    for _ in range(600):
        place = int(rng.integers(4))
        if rng.random() < 0.1:
            showing[place] = int(rng.integers(6))
        mask = texts[showing[place]].copy()
        flips = int(rng.choice([0, 0, 0, 3, 30, 200]))
        mask.flat[rng.choice(mask.size, flips, replace=False)] ^= True
        crop = mask.astype(np.uint8) * 255
        hint = last.get(place)

        limit = SAME * 6 * len(mask)
        scores = [_band(first, mask) for first in firsts]
        if hint is not None and scores[hint] < limit:
            expected = hint
        elif scores and min(scores) < limit:
            expected = int(np.argmin(scores))
        else:
            expected = len(firsts)
            firsts.append(mask)
            best.append(crop)
            ink.append(-1)
            count.append(0)
        if int(mask.sum()) > ink[expected]:
            best[expected], ink[expected] = crop, int(mask.sum())
        count[expected] += 1

        last[place] = catalog.add(mask, crop, hint)
        assert last[place] == expected
    assert catalog.count == count
    assert all(np.array_equal(a, b) for a, b in zip(catalog.crops, best, strict=True))


def _moment(time, ann=None, bob=None, cat=None, board=()):
    """One picture: each player's status line (None while their panel is off screen) and the board shown."""
    cards = {"ANN": ("Ah", "Ad"), "BOB": ("Kc", "Kd"), "CAT": ("7s", "2h")}
    seen = tuple(Seen(name, status, cards[name]) for name, status in (("ANN", ann), ("BOB", bob), ("CAT", cat))
                 if status is not None)
    return Moment(float(time), seen, tuple(board) + (None,) * (5 - len(board)), np.zeros((8, 40), bool))


def test_build_turns_a_timeline_into_a_hand():
    """Three players. Panels first show up in the order players act before the flop: button, small blind, big
    blind. CAT's panel goes away for good before the flop, and BOB's on the flop, so both folded."""
    flop = ("Qs", "Jd", "2c")
    moments = [
        _moment(0, ann=BLANK), _moment(1, ann=BLANK),
        _moment(2, ann="RAISETO 6"), _moment(3, ann="RAISETO 6"),
        _moment(4, bob=BLANK), _moment(5, bob=BLANK),
        _moment(6, bob="CALL"), _moment(7, bob="CALL"),
        _moment(8, cat=BLANK), _moment(9, cat=BLANK),
        _moment(10, ann=BLANK, bob=BLANK), _moment(11, ann=BLANK, bob=BLANK),
        _moment(12, ann=BLANK, bob=BLANK, board=flop), _moment(13, ann=BLANK, bob=BLANK, board=flop),
        _moment(14, ann=BLANK, bob="CHECK", board=flop), _moment(15, ann=BLANK, bob="CHECK", board=flop),
        _moment(16, ann="BET 8", bob="CHECK", board=flop), _moment(17, ann="BET 8", bob="CHECK", board=flop),
        _moment(18, ann="BET 8", board=flop), _moment(19, ann="BET 8", board=flop),
        _moment(20, ann="WIN 14", board=flop), _moment(21, ann="WIN 14", board=flop),
    ]
    draft = build(moments, Stakes(sb=1, bb=2))
    record = draft.record
    assert (record.button, record.hero.seat, record.hero.cards) == (3, 1, "KcKd")
    assert [(s.seat, s.alias, s.hole_cam) for s in record.seats] == [
        (1, "Bob", None), (2, "Cat", "7s2h"), (3, "Ann", "AhAd")]
    assert record.events == [
        "s3 raise 6 @00:02.0",
        "s1 call @00:06.0",
        "s2 fold @00:10.0",
        "flop QsJd2c @00:12.0",
        "s1 check @00:14.0",
        "s3 bet 8 @00:16.0",
        "s1 fold @00:18.0",
    ]
    assert (draft.wins, draft.issues) == ({3: 14}, [])


def _event(line: str) -> tuple[str, float]:
    body, _, when = line.rpartition("@")
    return body.strip(), VideoTime.parse(when).earliest_ms / 1000


@pytest.mark.skipif(not VIDEO.exists(), reason="the video is not in git: run scripts/fetch_real_video.py")
def test_read_command_gives_the_hand_read_by_eye(real_doc):
    pytest.importorskip("rapidocr_onnxruntime")
    out = io.StringIO()
    with redirect_stdout(out):
        main(["read", str(VIDEO), str(VIDEO)])
    first, second = yaml.safe_load_all(out.getvalue())
    assert first == second  # the second video reads with the OCR model already loaded

    truth = real_doc.hands[0]
    stakes = Stakes(**first["stakes"])
    record = HandRecord.model_validate(first["hands"][0])
    assert stakes == real_doc.session.stakes
    assert (record.button, record.hero) == (truth.button, truth.hero)
    assert [(s.seat, s.alias, s.hole_cam) for s in record.seats] == [
        (s.seat, s.alias, s.hole_cam) for s in truth.seats]
    got, want = [_event(e) for e in record.events], [_event(e) for e in truth.events]
    assert [body for body, _ in got] == [body for body, _ in want]
    # Times by eye are good to 1-3 s. The fold misses the 3 s line: the graphics clear Ian's panel about 3 s after
    # his cards hit the felt.
    late = [round(g - w, 1) for (_, g), (_, w) in zip(got, want)]
    assert all(abs(x) <= 3.0 for x in late[:-1]) and 3.0 < late[-1] <= 3.5, late
    assert replay(record, stakes).payoffs == truth.result


@pytest.mark.skipif(not VIDEO.exists(), reason="the video is not in git: run scripts/fetch_real_video.py")
def test_reads_on_past_the_last_keyframe(real_doc, monkeypatch):
    """Cut the clip just after Joe goes all in, so the all-in shows only after the last keyframe left."""
    pytest.importorskip("rapidocr_onnxruntime")
    full = Clip.read(VIDEO)
    end = next(i for i in full.keys if full.times[i] > 216)
    cut = Clip(full.codec, full.extradata, full.packets[:end], full.times[:end], [i for i in full.keys if i < end])
    monkeypatch.setattr(Clip, "read", classmethod(lambda cls, path: cut))

    got = [_event(e) for e in read_hand(VIDEO).record.events]
    want = [(body, t) for body, t in map(_event, real_doc.hands[0].events) if t < cut.times[-1]]
    assert want[-1] == ("s2 allin", 215.0) and cut.times[cut.keys[-1]] < 215.0
    assert [body for body, _ in got] == [body for body, _ in want]
    assert all(abs(g - w) <= 3.0 for (_, g), (_, w) in zip(got, want)), got
