"""Turn a timeline of read pictures into a hand record: who sat where, their cards, the board and every action.

Rules of the broadcast that this relies on:
- Before the flop the graphics show one panel at a time, in the order players act, so the order panels first
  appear is under-the-gun ... button, small blind, big blind. Seats are numbered from the small blind (seat 1).
- A status line ("RAISE TO 160,000", "CALL") stays up until its betting round ends, then clears.
- A player whose panel goes away for good while others are still shown has folded.
- Each event is timed from the first picture that differs from the settled one before it, because the graphics
  lag the table by a second or more.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..model import HandRecord, Hero, Seat, Stakes
from .overlay import same

VERBS = {"RAISETO": "raise", "RAISE": "raise", "BET": "bet", "CALL": "call", "CHECK": "check",
         "ALLIN": "allin", "FOLD": "fold"}
ABSENT, UNREAD, BLANK = "-", "?", ""


@dataclass(frozen=True)
class Seen:
    """One player's panel in one picture, as read."""

    name: str
    status: str | None  # BLANK, None while it kept changing, else "WORD" or "WORD AMOUNT" ("TOCALL 30000")
    cards: tuple[str | None, str | None]


@dataclass(frozen=True)
class Moment:
    time: float
    seen: tuple[Seen, ...]
    board: tuple[str | None, ...]  # per strip slot: a card, "" if face up but unread, None if not face up
    strip: np.ndarray


@dataclass
class Action:
    time: float
    seat: int
    verb: str
    amount: int | None = None


@dataclass
class Draft:
    record: HandRecord
    issues: list[str]
    wins: dict[int, int]  # seat -> the WIN amount on screen


def clock(seconds: float) -> str:
    tenths = round(seconds * 10)
    minutes, rest = divmod(tenths, 600)
    return f"{minutes:02d}:{rest // 10:02d}.{rest % 10}"


def _runs(keys: list) -> list[tuple[object, int, int]]:
    runs: list[tuple[object, int, int]] = []
    for i, key in enumerate(keys):
        if runs and runs[-1][0] == key:
            runs[-1] = (key, runs[-1][1], i)
        else:
            runs.append((key, i, i))
    return runs


def _split(state: str) -> tuple[str, int | None]:
    word, _, amount = state.partition(" ")
    return word, int(amount) if amount else None


def _states(moments: list[Moment], name: str) -> list[str]:
    out = []
    for m in moments:
        seen = next((s for s in m.seen if s.name == name), None)
        out.append(ABSENT if seen is None else UNREAD if seen.status is None else seen.status)
    return out


def _cards(moments: list[Moment], name: str) -> list[str | None]:
    votes: list[Counter] = [Counter(), Counter()]
    for m in moments:
        for s in m.seen:
            if s.name == name:
                for j, card in enumerate(s.cards):
                    if card:
                        votes[j][card] += 1
    return [v.most_common(1)[0][0] if v else None for v in votes]


def _actions(moments: list[Moment], name: str, seat: int) -> tuple[list[Action], list[int], int | None, int | None]:
    """Actions from the player's status line, plus every TO CALL amount, the WIN amount, and the index of the
    first picture after their last settled panel (where a fold would be timed)."""
    states = _states(moments, name)
    actions: list[Action] = []
    to_call: list[int] = []
    win = None
    end = None  # index of the last picture of the previous settled run
    open_verb = None  # verb of the action still being typed in, until a blank or another word breaks it
    last_present = None
    for state, first, last in _runs(states):
        if last == first or state == UNREAD:
            continue  # a passing picture, or text caught mid-fade: neither ends the last state nor starts one
        start = end + 1 if end is not None else first
        end = last
        if state == ABSENT:
            continue
        last_present = last
        if state == BLANK or state.startswith("#"):
            open_verb = None
            continue
        word, amount = _split(state)
        verb = VERBS.get(word)
        if verb and verb == open_verb:
            if amount is not None:
                actions[-1].amount = amount
            continue
        open_verb = verb
        if verb:
            actions.append(Action(moments[start].time, seat, verb, amount))
        elif word == "TOCALL" and amount:
            to_call.append(amount)
        elif word == "WIN":
            win = amount
    fold_at = last_present + 1 if last_present is not None and last_present + 1 < len(moments) else None
    return actions, to_call, win, fold_at


def _board(moments: list[Moment], issues: list[str]) -> list[tuple[float, str, list[str]]]:
    counts = [sum(c is not None for c in m.board) for m in moments]
    strips: list[tuple[int, int]] = []  # settled runs of a steady strip
    anchor = 0
    for i in range(1, len(moments) + 1):
        if i == len(moments) or not same(moments[anchor].strip, moments[i].strip):
            if i - 1 > anchor:
                strips.append((anchor, i - 1))
            anchor = i
    out = []
    shown = 0
    for count, first, last in _runs(counts):
        if last == first or count <= shown:
            continue
        steady = [(a, b) for a, b in strips if b < first and all(c == shown for c in counts[a : b + 1])]
        long = [(a, b) for a, b in steady if moments[b].time - moments[a].time >= 2.0]
        base = (long or steady or [(first - 1, first - 1)])[-1]
        when = moments[base[1] + 1].time
        cards = []
        for slot in range(count):
            votes = Counter(m.board[slot] for m in moments[first : last + 1] if m.board[slot])
            cards.append(votes.most_common(1)[0][0] if votes else None)
        for street, lo, hi in (("flop", 0, 3), ("turn", 3, 4), ("river", 4, 5)):
            if shown < hi <= count:
                if None in cards[lo:hi]:
                    issues.append(f"{street} card unread")
                else:
                    out.append((when, street, cards[lo:hi]))
        shown = count
    return out


def _stacks(actions: list[Action], board: list[tuple[float, str, list[str]]], seats: list[int], stakes: Stakes,
            sb: int, bb: int, to_call: dict[int, list[int]]) -> dict[int, tuple[int, str]]:
    """The broadcast never shows stacks. An all-in gives one exactly; every other seat gets the least it could
    hold: what it put in plus the last amount it was asked to call."""
    total = stakes.antes_for(seats, bb)
    street = {s: 0 for s in seats}
    street[sb], street[bb] = stakes.sb, stakes.bb
    high = stakes.bb
    exact: dict[int, int] = {}
    timeline = sorted([(a.time, 0, a) for a in actions] + [(t, 1, None) for t, _, _ in board], key=lambda e: e[:2])
    for _, _, a in timeline:
        if a is None:
            for s in seats:
                total[s] += street[s]
                street[s] = 0
            high = 0
        elif a.verb in ("bet", "raise", "allin") and a.amount:
            if a.verb == "allin":
                exact[a.seat] = total[a.seat] + a.amount
            street[a.seat] = a.amount
            high = max(high, a.amount)
        elif a.verb == "call":
            street[a.seat] = high
    out = {}
    for s in seats:
        if s in exact:
            out[s] = (exact[s], "stack worked out from the all-in amount")
        else:
            last_ask = to_call[s][-1] if to_call.get(s) else stakes.bb
            out[s] = (total[s] + street[s] + last_ask, "stack not shown on the broadcast; this is the least it could be")
    return out


def _roles(n: int) -> list[str]:
    roles = [""] * n
    roles[0] = "under the gun"
    if n >= 4:
        roles[n - 4] = "cutoff"
    if n >= 3:
        roles[n - 3] = "button"
    roles[n - 2] = "small blind" if n >= 3 else "button, small blind"
    roles[n - 1] = "big blind"
    return roles


def build(moments: list[Moment], stakes: Stakes) -> Draft:
    issues: list[str] = []
    first: dict[str, float] = {}
    count: Counter = Counter()
    for m in moments:
        for s in m.seen:
            first.setdefault(s.name, m.time)
            count[s.name] += 1
    cards = {name: _cards(moments, name) for name in first}
    names = []
    for name in sorted(first, key=first.get):
        if count[name] >= 2 and None not in cards[name]:
            names.append(name)
        else:
            issues.append(f"ignored panel {name!r}: seen {count[name]} times, cards {cards[name]}")
    n = len(names)
    if n < 2:
        raise ValueError(f"read {n} player(s); need at least two")
    seat_of = {name: (i - (n - 2)) % n + 1 for i, name in enumerate(names)}
    button = seat_of[names[n - 3]] if n >= 3 else seat_of[names[0]]
    sb, bb = seat_of[names[n - 2]], seat_of[names[n - 1]]

    actions: list[Action] = []
    to_call: dict[int, list[int]] = {}
    wins: dict[int, int] = {}
    present_after: dict[str, int | None] = {}
    for name in names:
        acts, asks, win, fold_at = _actions(moments, name, seat_of[name])
        actions += acts
        to_call[seat_of[name]] = asks
        if win is not None:
            wins[seat_of[name]] = win
        present_after[name] = fold_at
    for name in names:
        seat, at = seat_of[name], present_after[name]
        if at is None or seat in wins or any(a.seat == seat and a.verb == "fold" for a in actions):
            continue
        if any(s.name != name for m in moments[at:] for s in m.seen):
            actions.append(Action(moments[at].time, seat, "fold"))
    for verb in ("bet", "raise"):
        for a in actions:
            if a.verb == verb and a.amount is None:
                issues.append(f"seat {a.seat} {verb} at {clock(a.time)}: amount unread")
    if to_call.get(sb) and to_call[sb][0] != stakes.bb - stakes.sb:
        issues.append(f"small blind was first asked to call {to_call[sb][0]}, expected {stakes.bb - stakes.sb}")

    board = _board(moments, issues)
    lines = [(a.time, f"s{a.seat} {a.verb}" + (f" {a.amount}" if a.verb in ("bet", "raise") and a.amount else ""))
             for a in actions]
    lines += [(t, f"{street} {''.join(c)}") for t, street, c in board]
    lines.sort(key=lambda line: line[0])

    seats = sorted(seat_of.values())
    stacks = _stacks(actions, board, seats, stakes, sb, bb, to_call)
    roles = _roles(n)
    hero_seat = max(actions, key=lambda a: a.time).seat if actions else seat_of[names[0]]
    hero_name = next(name for name in names if seat_of[name] == hero_seat)
    record = HandRecord(
        index=1,
        button=button,
        seats=[
            Seat(seat=seat_of[name], alias=name.title(), stack=stacks[seat_of[name]][0],
                 note="; ".join(x for x in (roles[i], stacks[seat_of[name]][1]) if x),
                 hole_cam=None if name == hero_name else "".join(cards[name]))
            for i, name in sorted(enumerate(names), key=lambda pair: seat_of[pair[1]])
        ],
        hero=Hero(seat=hero_seat, cards="".join(cards[hero_name])),
        dealt_at=clock(min(m.time for m in moments if m.seen)),
        events=[f"{body} @{clock(t)}" for t, body in lines],
    )
    return Draft(record, issues, wins)
