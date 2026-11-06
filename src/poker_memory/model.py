"""Input records and small value types. Chips are always integers."""

from __future__ import annotations

import re
from typing import Literal

from pokerkit import Card
from pydantic import BaseModel, Field, field_validator, model_validator

Street = Literal["preflop", "flop", "turn", "river"]
STREETS: tuple[Street, ...] = ("preflop", "flop", "turn", "river")
Verb = Literal["fold", "check", "call", "bet", "raise", "allin"]

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$")


class VideoTime(BaseModel):
    """A position in the session video. The written precision sets the interval:
    "01:55.4" means somewhere in [115400, 115499] ms."""

    earliest_ms: int
    latest_ms: int
    text: str

    @staticmethod
    def parse(text: str) -> "VideoTime":
        m = _TIME_RE.match(text.strip())
        if not m:
            raise ValueError(f"bad time {text!r}; use mm:ss or mm:ss.d")
        hours, minutes, seconds, frac = m.groups()
        if int(seconds) >= 60:
            raise ValueError(f"bad time {text!r}; seconds must be < 60")
        base = ((int(hours or 0) * 60 + int(minutes)) * 60 + int(seconds)) * 1000
        if frac:
            unit = 10 ** (3 - len(frac))
            base += int(frac) * unit
        else:
            unit = 1000
        return VideoTime(earliest_ms=base, latest_ms=base + unit - 1, text=text.strip())


def parse_cards(text: str) -> list[str]:
    """'AhQd' -> ['Ah', 'Qd']; raises ValueError on anything that is not a real card."""
    compact = text.replace(" ", "")
    if len(compact) % 2:
        raise ValueError(f"bad cards {text!r}")
    out = []
    for i in range(0, len(compact), 2):
        token = compact[i : i + 2]
        try:
            cards = list(Card.parse(token))
        except ValueError:
            raise ValueError(f"bad card {token!r}") from None
        if len(cards) != 1 or cards[0].unknown_status:
            raise ValueError(f"bad card {token!r}")
        out.append(repr(cards[0]))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate card in {text!r}")
    return out


class Event(BaseModel):
    """One parsed line of a hand's event list."""

    seq: int
    raw: str
    kind: Literal["action", "board", "show", "muck"]
    seat: int | None = None
    verb: Verb | None = None
    amount: int | None = None
    street: Street | None = None
    cards: list[str] = []
    time: VideoTime | None = None


_ACTION_RE = re.compile(r"^s(\d+)\s+(fold|check|call|bet|raise|allin|all-in)(?:\s+(\d+))?$", re.I)
_BOARD_RE = re.compile(r"^(flop|turn|river)\s+(\S+)$", re.I)
_SHOW_RE = re.compile(r"^s(\d+)\s+show\s+(\S+)$", re.I)
_MUCK_RE = re.compile(r"^s(\d+)\s+muck$", re.I)


def parse_event(seq: int, raw: str) -> Event:
    body, _, when = raw.partition("@")
    body = " ".join(body.split())
    time = VideoTime.parse(when) if when.strip() else None
    if m := _ACTION_RE.match(body):
        verb = m.group(2).lower().replace("-", "")
        amount = int(m.group(3)) if m.group(3) else None
        if verb in ("bet", "raise") and amount is None:
            raise ValueError(f"{verb} needs a 'to' amount, e.g. 's3 raise 75'")
        if verb in ("fold", "check", "allin") and amount is not None:
            raise ValueError(f"{verb} takes no amount")
        return Event(seq=seq, raw=raw, kind="action", seat=int(m.group(1)), verb=verb, amount=amount, time=time)
    if m := _BOARD_RE.match(body):
        street = m.group(1).lower()
        cards = parse_cards(m.group(2))
        if len(cards) != (3 if street == "flop" else 1):
            raise ValueError(f"{street} needs {3 if street == 'flop' else 1} card(s)")
        return Event(seq=seq, raw=raw, kind="board", street=street, cards=cards, time=time)
    if m := _SHOW_RE.match(body):
        cards = parse_cards(m.group(2))
        if len(cards) != 2:
            raise ValueError("show needs exactly two cards")
        return Event(seq=seq, raw=raw, kind="show", seat=int(m.group(1)), cards=cards, time=time)
    if m := _MUCK_RE.match(body):
        return Event(seq=seq, raw=raw, kind="muck", seat=int(m.group(1)), time=time)
    raise ValueError(f"cannot read event {raw!r}")


class Seat(BaseModel):
    seat: int = Field(ge=1, le=10)
    alias: str = Field(min_length=1, max_length=40)
    stack: int = Field(gt=0)
    note: str = ""
    hole_cam: str | None = None  # cards a broadcast hole-card camera showed; hindsight only, never used in the math

    @field_validator("hole_cam")
    @classmethod
    def _two_real_cards(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cards = parse_cards(value)
        if len(cards) != 2:
            raise ValueError("hole_cam needs exactly two cards")
        return "".join(cards)


class Hero(BaseModel):
    seat: int
    cards: str


class HandRecord(BaseModel):
    """What a person writes down about one hand. Event amounts for bet/raise are
    'to' totals for the street. Opponent hole cards are unknown unless shown."""

    index: int = Field(ge=1)
    button: int
    seats: list[Seat] = Field(min_length=2, max_length=10)
    hero: Hero
    dealt_at: str | None = None
    events: list[str]
    result: dict[int, int] | None = None

    @model_validator(mode="after")
    def _check(self) -> "HandRecord":
        numbers = [s.seat for s in self.seats]
        if len(set(numbers)) != len(numbers):
            raise ValueError("duplicate seat numbers")
        if self.button not in numbers:
            raise ValueError("button must be an occupied seat")
        if self.hero.seat not in numbers:
            raise ValueError("hero seat must be an occupied seat")
        if len(parse_cards(self.hero.cards)) != 2:
            raise ValueError("hero needs exactly two hole cards")
        if any(s.hole_cam for s in self.seats if s.seat == self.hero.seat):
            raise ValueError("hero's cards go in 'hero', not 'hole_cam'")
        seen = parse_cards(self.hero.cards) + [c for s in self.seats if s.hole_cam for c in parse_cards(s.hole_cam)]
        if len(set(seen)) != len(seen):
            raise ValueError("a card appears twice among hero and hole_cam cards")
        if self.dealt_at:
            VideoTime.parse(self.dealt_at)
        return self

    def alias(self, seat: int) -> str:
        return next((s.alias for s in self.seats if s.seat == seat), f"seat {seat}")


class Stakes(BaseModel):
    sb: int = Field(gt=0)
    bb: int = Field(gt=0)
    ante: int = Field(default=0, ge=0)
    ante_kind: Literal["uniform", "big_blind"] = "uniform"
    currency: str = "chips"

    def antes_for(self, seats: list[int], big_blind: int) -> dict[int, int]:
        return {seat: self.ante if self.ante_kind == "uniform" or seat == big_blind else 0 for seat in seats}


class Source(BaseModel):
    """Where an openly licensed video came from, so it can be credited next to the player."""

    title: str
    author: str
    url: str
    license: str
    license_url: str


class SessionInfo(BaseModel):
    key: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    title: str
    consent: Literal["all_participants", "open_license", "none"]
    synthetic: bool = False
    stakes: Stakes
    video: str | None = None
    source: Source | None = None

    @model_validator(mode="after")
    def _video_needs_permission(self) -> "SessionInfo":
        if self.consent == "open_license" and not self.source:
            raise ValueError("an open-license video needs its source: title, author, url, license, license_url")
        if self.video and self.consent == "none":
            raise ValueError("a video can only be attached when every participant consented or it has an open license")
        return self


class SessionDoc(BaseModel):
    session: SessionInfo
    hands: list[HandRecord]

    @staticmethod
    def load(path) -> "SessionDoc":
        from pathlib import Path

        import yaml

        return SessionDoc.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
