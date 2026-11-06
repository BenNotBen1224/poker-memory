"""Native PokerGO development observations; never turn incomplete pixels into a hand.

Calibrated to the candidate63 854x480 development window. The top row's
positions are graphic slots, not verified physical seat numbers. Times describe
sampled graphic appearances, not the underlying physical poker actions.
"""

from __future__ import annotations

import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, NoReturn

import av
import numpy as np
from pydantic import BaseModel, Field

from ..engine import replay
from ..model import HandRecord, Hero, Seat, Stakes
from .hand import Draft, clock
from .overlay import Catalog, Ocr, parse_rank
from .suits import read_suit

_COLUMNS = ((92, 193), (194, 294), (295, 395), (396, 497), (498, 599))
_EVERY = 6


class Observation(BaseModel):
    field: str
    owner: str | None = None
    value: str | int | list[str] | None = None
    raw: str = ""
    frame_index: int
    pts: int
    time_base: str
    earliest_seconds: float
    latest_seconds: float
    kind: Literal["observed", "unknown", "inferred"] = "observed"
    issue: str | None = None


class _ActionLabel(Observation):
    owner: str
    value: str


class PokerGoObservations(BaseModel):
    layout: str = "pokergo-native-development"
    status: str = "uncertain"
    stakes: Stakes | None = None
    observations: list[Observation] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    seconds: dict[str, float] = Field(default_factory=dict)
    sampled_frames: int = 0
    decoded_frames: int = 0
    review_status: str = "NOT REVIEWED"


class UncertainHand(ValueError):
    def __init__(self, evidence: PokerGoObservations):
        self.stage = "pokergo_sequence"
        self.evidence = evidence
        super().__init__("PokerGO observations are incomplete; no justified HandRecord: " + "; ".join(evidence.issues))


def parse_amount(text: str) -> int | None:
    """Parse chip text without deleting decimals or repairing OCR magnitudes."""
    match = re.fullmatch(r"\s*(\d[\d,]*(?:\.\d+)?)([KM]?)\s*", text.upper())
    if not match or ",," in match[1]:
        return None
    integer = match[1].split(".")[0]
    if any(len(integer[pos + 1:].replace(",", "")) % 3 != 0 or not integer[pos + 1:]
           for pos, ch in enumerate(integer) if ch == ","):
        return None
    try:
        value = Decimal(match[1].replace(",", "")) * {"": 1, "K": 1000, "M": 1000000}[match[2]]
    except InvalidOperation:
        return None
    return int(value) if value == value.to_integral_value() and value >= 0 else None


def parse_action(text: str, maximum: int | None) -> tuple[str, int | None] | None:
    compact = re.sub(r"\s+", "", text.upper())
    if compact in ("FOLD", "CHECK", "CALL", "ALLIN"):
        return compact.lower(), None
    match = re.fullmatch(r"(RAISETO|BET|CALL)(.+)", compact)
    if not match:
        return None
    amount = parse_amount(match[2])
    if amount is None or amount <= 0 or maximum is None or amount > maximum:
        return None
    return ("raise" if match[1] == "RAISETO" else match[1].lower()), amount


def _crop(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    return np.ascontiguousarray(rgb[top:bottom, left:right].mean(axis=2).astype(np.uint8))


def _native_frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (854, 480):
            raise ValueError("PokerGO development observations require native 854x480 video")
        previous = None
        index = -1
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("PokerGO frame has no presentation timestamp")
            if index % _EVERY == 0:
                rgb = frame.to_ndarray(format="rgb24")
                yield index, frame.pts, str(frame.time_base), float(frame.pts * frame.time_base), rgb, previous
            # The suit pair uses an actual nearby decoded frame, not a repeated current crop.
            if index % _EVERY == _EVERY - 2:
                previous = frame.to_ndarray(format="rgb24")
        if index >= 0 and index % _EVERY:
            yield index, frame.pts, str(frame.time_base), float(frame.pts * frame.time_base), frame.to_ndarray(format="rgb24"), previous


def recognizes(rgb: np.ndarray, ocr: Ocr) -> bool:
    labels = ocr([_crop(rgb, (94, 26, 149, 39)), _crop(rgb, (247, 26, 304, 39))])
    return [re.sub(r"\s+", "", label.upper()) for label in labels] == ["BLINDS", "BBANTE"]


def _rank_texts(crops: list[np.ndarray], ocr: Ocr) -> list[str]:
    texts = ocr(crops)
    unsupported = [i for i, (crop, text) in enumerate(zip(crops, texts))
                   if parse_rank(text) is None and np.any(crop > 180)]
    padded = [np.pad(crops[i], ((0, 4), (0, 0)),
                     constant_values=int(np.median(crops[i][crops[i] > 180]))) for i in unsupported]
    for i, text in zip(unsupported, ocr(padded) if padded else []):
        if parse_rank(text) is not None:
            texts[i] = text
    return texts


def observe(path: Path, ocr: Ocr) -> PokerGoObservations:
    started = time.perf_counter()
    frames = list(_native_frames(path))
    if not frames:
        raise ValueError("PokerGO video decoded no frames")
    catalogs: dict[str, Catalog] = {}
    cells: list[dict[str, int]] = []

    def number(kind: str, crop: np.ndarray, dark: bool = False) -> int:
        catalog = catalogs.setdefault(kind, Catalog())
        return catalog.add(crop < 150 if dark else crop > 150, crop)

    for _, _, _, _, rgb, _ in frames:
        row = {}
        for slot, (left, right) in enumerate(_COLUMNS):
            for field, box, dark in (
                ("name", (left + 12, 43, right - 8, 57), False),
                ("stack", (left + 12, 58, right - 8, 73), False),
                ("position", (left - 2, 52, left + 12, 66), True),
            ):
                row[f"{field}:{slot}"] = number(f"{field}:{slot}", _crop(rgb, box), dark)
        for field, box in (
            ("blinds", (153, 26, 235, 39)), ("ante", (307, 26, 350, 39)),
            ("ante_label", (247, 26, 304, 39)), ("pot", (706, 388, 823, 408)),
        ):
            row[field] = number(field, _crop(rgb, box))
        for panel, top in enumerate((298, 370)):
            row[f"owner:{panel}"] = number("owner", _crop(rgb, (72, top + (23 if panel == 0 else 22), 175, top + 39)))
            row[f"action:{panel}"] = number("action", _crop(rgb, (72, top + 40, 175, top + 61)))
            for card, left in enumerate((73, 108)):
                row[f"rank:{panel}:{card}"] = number("rank", _crop(rgb, (left, top, left + 16, top + 18)), True)
        cells.append(row)
    scan_seconds = time.perf_counter() - started
    texts = {}
    for kind, catalog in catalogs.items():
        settled = catalog.settled()
        crops = [catalog.text(i) if kind in ("owner", "action") else catalog.crops[i] for i in settled]
        texts[kind] = dict(zip(settled, _rank_texts(crops, ocr) if kind == "rank" else ocr(crops)))
    result = PokerGoObservations(sampled_frames=len(frames), decoded_frames=frames[-1][0] + 1)
    seen: dict[tuple[str, str | None], tuple[object, str]] = {}
    previous_headers: dict[int, tuple[str, int, str | None]] = {}
    previous_labels: dict[int, tuple[str, tuple[str, int | None]]] = {}
    label_onsets: dict[int, float] = {}
    displayed_stacks: dict[str, int] = {}
    stakes_candidates: set[tuple[int, int, int]] = set()
    previous_time = frames[0][3]

    def add(field, owner, value, raw, frame, issue=None, earliest=None):
        signature = (value, raw) if value is None else (value, "")
        key = (field, owner)
        if seen.get(key) == signature:
            return
        seen[key] = signature
        index, pts, time_base, seconds, _, _ = frame
        result.observations.append(Observation(
            field=field, owner=owner, value=value, raw=raw, frame_index=index, pts=pts,
            time_base=time_base, earliest_seconds=previous_time if earliest is None else earliest, latest_seconds=seconds,
            kind="unknown" if value is None else "observed", issue=issue,
        ))

    for frame, row in zip(frames, cells):
        _, _, _, _, rgb, previous_rgb = frame

        def text(key, kind=None):
            return texts[kind or key].get(row[key], "").strip()

        header = {}
        for slot in range(5):
            name = text(f"name:{slot}").upper()
            stack_text = text(f"stack:{slot}")
            stack = parse_amount(stack_text)
            if len(re.sub(r"[^0-9]", "", stack_text)) < 3:
                stack = None
            position_text = text(f"position:{slot}").upper()
            position = position_text if position_text in ("D", "SB", "BB") else None
            if re.fullmatch(r"[A-Z][A-Z .'-]{1,39}", name) and stack is not None and stack > 0:
                header[slot] = name, stack, position
                if previous_headers.get(slot) == header[slot]:
                    displayed_stacks[name] = stack
                    add("displayed_stack", name, stack, stack_text, frame)
                    add("graphic_slot", name, slot, name, frame)
            elif re.fullmatch(r"[A-Z][A-Z .'-]{1,39}", name):
                add("displayed_stack", name, None, stack_text, frame, "incomplete/unsupported numeric graphic")
        roles = {position: slot for slot, (_, _, position) in header.items() if position}
        positions = {position: name for name, _, position in header.values() if position}
        if (len(header) == 5 and len(roles) == 3
                and (roles["SB"] - roles["D"]) % 5 == 1 and (roles["BB"] - roles["SB"]) % 5 == 1
                and all(previous_headers.get(s) == h for s, h in header.items())):
            for position, name in positions.items():
                add("position", name, position, position, frame)
        previous_headers = header
        blinds = text("blinds").split("/")
        ante = parse_amount(text("ante"))
        if len(blinds) == 2:
            sb, bb = map(parse_amount, blinds)
            if sb and bb and sb <= bb and ante is not None and re.sub(r"\s+", "", text("ante_label")) == "BBANTE":
                stakes_candidates.add((sb, bb, ante))
                add("stakes", None, [str(sb), str(bb), str(ante), "big_blind"],
                    text("blinds") + " BB ANTE " + text("ante"), frame)
        pot_text = text("pot")
        pot = parse_amount(pot_text)
        if pot is not None:
            add("displayed_pot", None, pot, pot_text, frame)
        labels = {}
        for panel, top in enumerate((298, 370)):
            owner = text(f"owner:{panel}", "owner").upper()
            add(f"panel_owner_{panel + 1}", None, owner if owner in displayed_stacks else None, owner, frame,
                "absent/unsupported owner crop or unmatched header name" if owner not in displayed_stacks else None)
            if owner not in displayed_stacks or previous_rgb is None:
                continue
            cards = []
            for card, left in enumerate((73, 108)):
                rank = parse_rank(text(f"rank:{panel}:{card}", "rank"))
                suit_top = top + (2 if panel == 0 else 1)
                suit = read_suit(rgb[suit_top:suit_top + (17 if panel == 0 else 16), left + 16:left + 32],
                                 previous_rgb[suit_top:suit_top + (17 if panel == 0 else 16), left + 16:left + 32])
                add(f"hole_rank_{card + 1}", owner, rank, text(f"rank:{panel}:{card}", "rank"), frame,
                    "rank uncertain" if rank is None else None)
                add(f"hole_suit_{card + 1}", owner, suit, "", frame,
                    "glyph absent/transition/unsupported" if suit is None else None)
                cards.append(rank + suit if rank and suit else None)
            if all(cards):
                add("camera_cards", owner, cards, "", frame)
            else:
                add("camera_cards", owner, None, "", frame, "rank/suit absent, changing, or unsupported")
            raw = text(f"action:{panel}", "action")
            action = parse_action(raw, displayed_stacks[owner])
            if action:
                labels[panel] = owner, action
                if previous_labels.get(panel) == labels[panel]:
                    verb, amount = action
                    add("action_label", owner, verb if amount is None else f"{verb} {amount}", raw, frame,
                        earliest=label_onsets[panel])
                else:
                    label_onsets[panel] = previous_time
                    add("action_label", owner, None, raw, frame, "label not yet confirmed across samples")
            elif raw.startswith("+"):
                gain = parse_amount(raw[1:])
                if gain is not None:
                    add("win_label", owner, gain, raw, frame)
            elif raw:
                add("action_label", owner, None, raw, frame, "unsupported/inconsistent action text or amount")
        previous_labels = labels
        previous_time = frame[3]
    if len(stakes_candidates) == 1:
        sb, bb, ante = next(iter(stakes_candidates))
        result.stakes = Stakes(sb=sb, bb=bb, ante=ante, ante_kind="big_blind")
    else:
        result.issues.append("stakes missing or inconsistent")
    result.issues.extend([
        "top-row stack/position updates are asynchronous; starting stacks and hand boundary are not certified",
        "graphic slots are not recorded physical seat numbers",
        "panel disappearance is not a recorded fold; a complete action sequence is not established",
        "board-region recognition is unsupported in this development observation path",
        "camera cards are hindsight observations, not decision-time opponent knowledge",
        "time intervals bracket sampled graphic recognition, not certified physical-action or visibility-onset times",
    ])
    add("board", None, None, "", frames[-1], "no validated board-region detector for this layout")
    result.seconds = {"decode_and_catalog": scan_seconds, "total": time.perf_counter() - started}
    return result


def build_preflop(evidence: PokerGoObservations) -> Draft:
    """Support one unopened preflop raise won by explicit folds, not postflop play.

    The caller supplies a pre-cut hand. Logical seats follow the header's relative
    cyclic order; they are application identifiers, not printed chair numbers.
    """
    def reject(message: str) -> NoReturn:
        failed = evidence.model_copy(deep=True)
        failed.issues.append(message)
        raise UncertainHand(failed)

    if evidence.stakes is None:
        reject("preflop builder: missing or inconsistent stakes")
    observed = [o for o in evidence.observations if o.kind == "observed"]
    if any(o.field == "board" and o.value for o in observed):
        reject("preflop builder: observed board cards/postflop input are unsupported")
    slots: dict[str, int] = {}
    positions: dict[str, str] = {}
    stacks: dict[str, int] = {}
    for o in observed:
        if o.field == "graphic_slot" and isinstance(o.value, int) and o.owner:
            if o.owner in slots and slots[o.owner] != o.value:
                reject("preflop builder: graphic order changed within the cut")
            slots[o.owner] = o.value
        elif o.field == "position" and isinstance(o.value, str) and o.owner:
            if o.value in positions and positions[o.value] != o.owner:
                reject("preflop builder: position markers changed within the cut")
            positions[o.value] = o.owner
        elif o.field == "displayed_stack" and isinstance(o.value, int) and o.owner:
            stacks.setdefault(o.owner, o.value)
    count = len(slots)
    if count != 5 or set(slots.values()) != set(range(count)) or set(stacks) != set(slots):
        reject("preflop builder: requires five complete, stable header players/stacks")
    if set(positions) != {"D", "SB", "BB"} or any(name not in slots for name in positions.values()):
        reject("preflop builder: missing button/blind roles")
    button, sb, bb = (slots[positions[role]] for role in ("D", "SB", "BB"))
    if (sb - button) % count != 1 or (bb - sb) % count != 1:
        reject("preflop builder: role/order contradiction")
    actions: list[_ActionLabel] = []
    last_label: dict[str, str] = {}
    for o in observed:
        if o.field != "action_label" or not o.owner or not isinstance(o.value, str):
            continue
        if last_label.get(o.owner) != o.value:
            actions.append(_ActionLabel.model_validate(o.model_dump()))
            last_label[o.owner] = o.value
    if len(actions) != count or len({o.owner for o in actions}) != count:
        reject("preflop builder: missing/repeated actions; disappearance is not a fold")
    if any(o.value != "fold" and not o.value.startswith("raise ") for o in actions):
        reject("preflop builder: calls, checks, all-ins and postflop sequences are unsupported")
    raises = [o for o in actions if o.value.startswith("raise ")]
    if len(raises) != 1 or actions[-1].value != "fold":
        reject("preflop builder: requires one explicit opening raise followed by a fold")
    winner = raises[0].owner
    expected_order = sorted(slots, key=lambda name: (slots[name] - bb - 1) % count)
    if [o.owner for o in actions] != expected_order:
        reject("preflop builder: observed labels do not follow preflop action order")
    first_action, last_action = actions[0].frame_index, actions[-1].frame_index
    if any(o.owner in stacks and o.field in ("graphic_slot", "position", "displayed_stack")
           and o.frame_index >= first_action
           and (o.field != "displayed_stack" or o.value != stacks[o.owner]) for o in observed):
        reject("preflop builder: initial header not established before actions")
    cards: dict[str, list[str]] = {}
    issues = [
        "development-calibrated PokerGO preflop-only reader; pre-cut hand required",
        "logical seat IDs follow relative graphic order; not printed physical chair numbers",
        "starting stacks use stable pre-action header values, checked against replay/pot/win labels",
        "timestamps bracket sampled graphic recognition, not physical actions; manual segmentation is external",
        "opponent camera cards are hindsight only; no recorded showdown reveals",
        "no board dealt: complete explicit preflop fold sequence terminates the replay; postflop recognition unsupported",
    ]
    for o in observed:
        if o.field == "camera_cards" and o.owner and isinstance(o.value, list):
            if o.owner in cards and cards[o.owner] != o.value:
                reject("preflop builder: conflicting camera cards within the cut")
            cards[o.owner] = o.value
    if winner not in cards:
        reject("preflop builder: winner's required hero cards are unread")
    for name in slots:
        if name not in cards:
            issues.append(f"{name}: camera cards incomplete/unsupported; retained as unknown")
    seat_of = {name: slot + 1 for name, slot in slots.items()}
    record = HandRecord(
        index=1, button=button + 1, hero=Hero(seat=seat_of[winner], cards="".join(cards[winner])),
        seats=[Seat(seat=seat_of[name], alias=name.title(), stack=stacks[name],
                    note="logical ID from relative header order; pre-action displayed stack",
                    hole_cam="".join(cards[name]) if name != winner and name in cards else None)
               for name in sorted(slots, key=slots.get)],
        events=[f"s{seat_of[o.owner]} {o.value} @{clock(o.latest_seconds)}" for o in actions],
    )
    validated = replay(record, evidence.stakes)
    if validated.status != "verified" or validated.board or validated.payoffs is None:
        reject("preflop builder: replay rejected sequence: " + "; ".join(i.message for i in validated.issues))
    initial_pot = validated.steps[0].before.pot
    raised_pot = next(step.after.pot for step in validated.steps if step.verb == "raise")
    pots = [o for o in observed if o.field == "displayed_pot"]
    if (not any(o.value == initial_pot and o.frame_index < raises[0].frame_index for o in pots)
            or not any(o.value == raised_pot and o.frame_index <= last_action for o in pots)
            or any(o.value not in (initial_pot, raised_pot) for o in pots)):
        reject("preflop builder: displayed pots disagree with blind/ante/action replay")
    win_labels = [o for o in observed if o.field == "win_label"]
    if (not win_labels or any(o.owner != winner or o.value != validated.payoffs[seat_of[winner]]
                              or o.frame_index < raises[0].frame_index for o in win_labels)):
        reject("preflop builder: missing/inconsistent winner net-gain label")
    evidence.status = "preflop_replay_verified"
    evidence.issues = issues.copy()
    evidence.observations.append(actions[-1].model_copy(update={
        "field": "board_from_replay", "owner": None, "value": [], "raw": "", "kind": "inferred",
        "issue": "no board dealt because the explicit preflop action stream terminates the validated replay",
    }))
    return Draft(record, issues, {seat_of[winner]: validated.payoffs[seat_of[winner]]})
