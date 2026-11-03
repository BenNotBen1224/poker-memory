"""Structured hand search over replayed hands and decision-point lookup."""

from __future__ import annotations

from .engine import Step
from .interpret import Query
from .model import parse_cards
from .store import StoredHand

_RANK = "AKQJT98765432"


def hero_cards(h: StoredHand) -> list[str]:
    return parse_cards(h.record.hero.cards)


def combo_name(cards: list[str]) -> str:
    a, b = sorted(cards, key=lambda c: _RANK.index(c[0]))
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def _opponent_raised_last(steps: list[Step], i: int, hero: int) -> bool:
    for prev in reversed(steps[:i]):
        if prev.kind != "action" or prev.before.street != steps[i].before.street:
            return False
        if prev.seat != hero:
            return "raises to" in prev.label
    return False


def find_decision(h: StoredHand, q: Query) -> Step | None:
    """Hero's decision the question is about: the last Hero action matching the
    street/verb/raise filters, preferring ones where Hero faced a bet."""
    hero = h.record.hero.seat
    steps = h.replay.steps
    picks = [
        s for i, s in enumerate(steps)
        if s.kind == "action" and s.seat == hero
        and (q.street is None or s.before.street == q.street)
        and (q.hero_verb is None or s.verb == q.hero_verb or (q.hero_verb == "allin" and "(all-in)" in s.label))
        and (not q.faced_raise or _opponent_raised_last(steps, i, hero))
    ]
    facing = [s for s in picks if s.before.to_call]
    return (facing or picks or [None])[-1]


def matches(h: StoredHand, q: Query) -> bool:
    cards = hero_cards(h)
    if q.hand_index is not None and h.index != q.hand_index:
        return False
    if q.cards and sorted(q.cards) != sorted(cards):
        return False
    if q.ranks:
        name = combo_name(cards)
        if name[:2] != q.ranks or (q.suited is not None and name[2:] != ("s" if q.suited else "o")):
            return False
    if q.player and q.player.lower() not in {s.alias.lower() for s in h.record.seats if s.seat != h.record.hero.seat}:
        return False
    if q.street or q.hero_verb or q.faced_raise:
        return find_decision(h, q) is not None
    return True


def search(hands: list[StoredHand], q: Query) -> list[StoredHand]:
    found = [h for h in hands if matches(h, q)]
    if q.latest and found:
        return [max(found, key=lambda h: h.index)]
    return found


def you_label(label: str, alias: str) -> str:
    """'Hero raises to 75' -> 'you raise to 75'."""
    if not label.startswith(alias + " "):
        return label
    verb, _, tail = label[len(alias) + 1:].partition(" ")
    verb = verb[:-1] if verb.endswith("s") else verb
    return f"you {verb} {tail}".strip()


def summary(h: StoredHand) -> str:
    hero = h.record.hero.seat
    cards = " ".join(hero_cards(h))
    mine = [s for s in h.replay.steps if s.kind == "action" and s.seat == hero]
    last = f"{you_label(mine[-1].label, h.record.alias(hero))} on the {mine[-1].before.street}" if mine else "no action"
    net = h.replay.payoffs.get(hero) if h.replay.payoffs else None
    result = f"{net:+}" if net is not None else "not verified"
    return f"Hand {h.index}: {cards}, {last}, result {result}"
