import pytest
from pydantic import ValidationError

from poker_memory.model import HandRecord, SessionInfo

BASE = {"key": "k", "title": "t", "stakes": {"sb": 1, "bb": 2}}
SOURCE = {"title": "clip", "author": "someone", "url": "https://example.org/clip",
          "license": "CC BY 3.0", "license_url": "https://creativecommons.org/licenses/by/3.0/"}


def hand(seat2_cam: str | None = None, hero_cards: str = "AhKd", seat1_cam: str | None = None) -> HandRecord:
    return HandRecord.model_validate({
        "index": 1, "button": 2, "hero": {"seat": 1, "cards": hero_cards}, "events": [],
        "seats": [{"seat": 1, "alias": "A", "stack": 100, "hole_cam": seat1_cam},
                  {"seat": 2, "alias": "B", "stack": 100, "hole_cam": seat2_cam}],
    })


def test_open_license_video_needs_its_source():
    with pytest.raises(ValidationError, match="needs its source"):
        SessionInfo(**BASE, consent="open_license", video="clip.webm")
    assert SessionInfo(**BASE, consent="open_license", video="clip.webm", source=SOURCE).source.license == "CC BY 3.0"


def test_video_is_refused_without_consent_or_license():
    with pytest.raises(ValidationError, match="consented or it has an open license"):
        SessionInfo(**BASE, consent="none", video="clip.mp4")
    assert SessionInfo(**BASE, consent="all_participants", video="clip.mp4").video == "clip.mp4"


def test_ante_defaults_to_zero():
    assert SessionInfo(**BASE, consent="none").stakes.ante == 0


def test_hole_cam_cards_are_normalised():
    assert hand(seat2_cam="Qc Td").seats[1].hole_cam == "QcTd"


@pytest.mark.parametrize("cam, message", [
    ("Qc", "exactly two cards"),
    ("QcQc", "duplicate card"),
    ("AhQc", "appears twice"),
])
def test_bad_hole_cam_is_refused(cam, message):
    with pytest.raises(ValidationError, match=message):
        hand(seat2_cam=cam)


def test_heros_own_cards_do_not_go_in_hole_cam():
    with pytest.raises(ValidationError, match="not 'hole_cam'"):
        hand(seat1_cam="AhKd")
