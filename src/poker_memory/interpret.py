"""Deterministic English question reader. No model, no guessing: anything it does
not recognise is simply not used, and the answer says what was understood."""

from __future__ import annotations

import re

from pydantic import BaseModel

from .model import Street, parse_cards

_RANK = "AKQJT98765432"
_RANK_WORDS = {"ace": "A", "king": "K", "queen": "Q", "jack": "J", "ten": "T", "nine": "9", "eight": "8",
               "seven": "7", "six": "6", "five": "5", "four": "4", "three": "3", "two": "2", "deuce": "2"}
_PAIR_WORDS = {"aces": "A", "kings": "K", "queens": "Q", "jacks": "J", "tens": "T", "nines": "9", "eights": "8",
               "sevens": "7", "sixes": "6", "fives": "5", "fours": "4", "threes": "3", "twos": "2", "deuces": "2"}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_VERBS = {"fold": "fold", "folded": "fold", "call": "call", "called": "call", "check": "check", "checked": "check",
          "bet": "bet", "raise": "raise", "raised": "raise", "shove": "allin", "shoved": "allin", "jam": "allin",
          "jammed": "allin"}

_RANGE_TOKEN = r"[AKQJT2-9]{2}[so]?(?:\+|-[AKQJT2-9]{2}[so]?)?"
_RANGE_RE = re.compile(
    r"(?:\brange(?:\s+(?:is|of))?|\bif\s+(?:he|she|they|villain|\w+)\s+(?:has|had|holds?|is\s+on|plays?)|"
    r"\bagainst|\bvs\.?|\bversus)\s*:?\s*"
    rf"({_RANGE_TOKEN}(?:\s*(?:,|\bor\b|\band\b)\s*{_RANGE_TOKEN})*)", re.I)
_BLUFF_RE = re.compile(
    r"bluff(?:s|ing)?\s+(?:about\s+|around\s+)?(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)|"
    r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)\s+(?:of\s+the\s+time\s+)?(?:a\s+)?bluff", re.I)
_CARDS_RE = re.compile(r"(?<![A-Za-z])((?:[AKQJT2-9][shdc]){2})(?![A-Za-z])")
_COMBO_RE = re.compile(r"(?<![A-Za-z0-9])([AKQJT2-9]{2})([so])?(?![A-Za-z0-9+])")
_WORD_COMBO_RE = re.compile(rf"\b({'|'.join(_RANK_WORDS)})[\s-]+({'|'.join(_RANK_WORDS)})\b", re.I)
_PAIR_RE = re.compile(rf"\b(?:pocket\s+)?({'|'.join(_PAIR_WORDS)})\b", re.I)
_STREET_RE = re.compile(r"\b(pre-?flop|flop|turn|river)\b", re.I)
_HERO_VERB_RE = re.compile(
    rf"\b(?:i|my)\s+(?:(?:just|then|had|have|did)\s+)?(?:(?!got\b|get\b|was\b|been\b)\w+\s+)?({'|'.join(_VERBS)})\b",
    re.I)
_RAISE_RE = re.compile(r"\b(?:re-?|check-?)?rais(?:e|ed|es|ing)\b", re.I)
_HAND_RE = re.compile(r"\bhand\s*(?:#|no\.?|number)?\s*(\d{1,4})\b", re.I)
_NUMBER_RE = re.compile(r"(?:#|\b(?:option|number|no\.)\s*)(\d{1,4})\b", re.I)
_ORDINAL_RE = re.compile(rf"\b({'|'.join(_ORDINALS)}|last)\s+one\b", re.I)
_LATEST_RE = re.compile(r"\b(?:last|latest|most\s+recent)\b", re.I)


class Query(BaseModel):
    hand_index: int | None = None
    number: int | None = None  # "#2": a candidate position if candidates were offered, else a hand number
    ordinal: int | None = None  # "the second one"; -1 means "the last one"
    latest: bool = False
    cards: list[str] | None = None  # exact hole cards, e.g. AhQd
    ranks: str | None = None  # e.g. "AQ"
    suited: bool | None = None
    player: str | None = None
    street: Street | None = None
    hero_verb: str | None = None
    faced_raise: bool = False
    villain_range: str | None = None
    bluff_share: float | None = None
    range_error: str | None = None

    def identifies_hand(self) -> bool:
        return bool(self.hand_index or self.cards or self.ranks or self.player or self.latest)

    def describe(self) -> str:
        parts = []
        if self.hand_index:
            parts.append(f"hand {self.hand_index}")
        if self.ordinal:
            parts.append("the last one" if self.ordinal == -1 else f"choice {self.ordinal}")
        if self.number:
            parts.append(f"#{self.number}")
        if self.cards:
            parts.append("".join(self.cards))
        elif self.ranks:
            parts.append(self.ranks + ("s" if self.suited else "o" if self.suited is False else ""))
        if self.player:
            parts.append(f"vs {self.player}")
        if self.hero_verb:
            parts.append(f"you {self.hero_verb}")
        if self.street:
            parts.append(f"on the {self.street}")
        if self.faced_raise:
            parts.append("facing a raise")
        if self.latest:
            parts.append("most recent")
        if self.villain_range:
            parts.append(f"his range {self.villain_range}")
        if self.bluff_share is not None:
            parts.append(f"what-if he bluffs {self.bluff_share:.0%}")
        return ", ".join(parts) or "nothing specific"


def _canonical_ranks(a: str, b: str) -> str:
    return "".join(sorted((a.upper(), b.upper()), key=_RANK.index))


def _cut(text: str, m: re.Match) -> str:
    return text[: m.start()] + " " + text[m.end():]


def parse_question(text: str, aliases: list[str] = ()) -> Query:
    q = Query()
    rest = text

    if m := _RANGE_RE.search(rest):
        tokens = re.split(r"\s*(?:,|\bor\b|\band\b)\s*", m.group(1), flags=re.I)
        q.villain_range = ",".join(t[:2].upper() + t[2:].lower() if len(t) > 2 and t[2] in "sSoO" else t.upper()
                                   for t in tokens if t)
        rest = _cut(rest, m)
    if m := _BLUFF_RE.search(rest):
        share = float(m.group(1) or m.group(2))
        if share > 100:
            q.range_error = "a bluff share must be between 0% and 100%"
        else:
            q.bluff_share = share / 100
        rest = _cut(rest, m)

    if m := _HAND_RE.search(rest):
        q.hand_index = int(m.group(1))
        rest = _cut(rest, m)
    elif m := _NUMBER_RE.search(rest):
        q.number = int(m.group(1))
        rest = _cut(rest, m)
    if m := _ORDINAL_RE.search(rest):
        word = m.group(1).lower()
        q.ordinal = -1 if word == "last" else _ORDINALS[word]
        rest = _cut(rest, m)
    elif _LATEST_RE.search(rest):
        q.latest = True

    if m := _CARDS_RE.search(rest):
        try:
            q.cards = parse_cards(m.group(1))
        except ValueError:
            pass
        rest = _cut(rest, m)
    elif m := _COMBO_RE.search(rest):
        q.ranks = _canonical_ranks(*m.group(1))
        q.suited = {"s": True, "o": False}.get(m.group(2) or "")
        rest = _cut(rest, m)
    elif m := _WORD_COMBO_RE.search(rest):
        q.ranks = _canonical_ranks(_RANK_WORDS[m.group(1).lower()], _RANK_WORDS[m.group(2).lower()])
    elif m := _PAIR_RE.search(rest):
        q.ranks = _PAIR_WORDS[m.group(1).lower()] * 2

    lowered = rest.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        if alias.lower() != "hero" and alias.lower() in lowered:
            q.player = alias
            break

    if m := _STREET_RE.search(rest):
        q.street = "preflop" if "pre" in m.group(1).lower() else m.group(1).lower()

    for m in _HERO_VERB_RE.finditer(rest):
        before = rest[: m.start()].lower().split()
        if before and before[-1] in ("should", "would", "could", "shouldn't", "wouldn't", "if"):
            continue  # a counterfactual, not what happened
        q.hero_verb = _VERBS[m.group(1).lower()]
        rest = _cut(rest, m)
        break
    q.faced_raise = bool(_RAISE_RE.search(rest))
    return q
