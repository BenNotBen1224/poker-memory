"""Replays a HandRecord through PokerKit. PokerKit types never leave this module.

Replay stops at the first blocking problem, so no later number is derived from a
broken state. Opponent hole cards stay unknown ("????") until a showdown reveal.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Literal

from pokerkit import Automation, Mode, NoLimitTexasHoldem
from pydantic import BaseModel

from .model import STREETS, Event, HandRecord, Stakes, VideoTime, parse_cards, parse_event

ENGINE_VERSION = f"pokerkit-{version('pokerkit')}/replay-1"

_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)
_REVEALS = ("show", "muck")


class Snapshot(BaseModel):
    street: str
    board: list[str]
    pot: int
    pots: list[int]  # collected main pot then side pots; current-street bets are not included
    stacks: dict[int, int]
    committed: dict[int, int]
    actor: int | None
    to_call: int | None
    min_raise_to: int | None
    max_raise_to: int | None


class Step(BaseModel):
    seq: int | None  # None for Hero's automatic showdown reveal
    raw: str
    kind: str
    seat: int | None
    verb: str | None = None
    label: str
    time: VideoTime | None
    before: Snapshot
    after: Snapshot


class Issue(BaseModel):
    seq: int | None
    code: str
    message: str


class Replay(BaseModel):
    status: Literal["verified", "invalid"]
    steps: list[Step]
    issues: list[Issue]
    payoffs: dict[int, int] | None
    revealed: dict[int, list[str]]
    board: list[str]
    engine_version: str = ENGINE_VERSION


class _Stop(Exception):
    def __init__(self, seq: int | None, code: str, message: str):
        self.issue = Issue(seq=seq, code=code, message=message)


def replay(record: HandRecord, stakes: Stakes) -> Replay:
    seats = sorted(s.seat for s in record.seats)
    b = seats.index(record.button)
    order = seats[b + 1 :] + seats[: b + 1]  # PokerKit index 0 = first seat after the button
    index_of = {seat: i for i, seat in enumerate(order)}
    start = {s.seat: s.stack for s in record.seats}
    hero = record.hero.seat
    hero_cards = parse_cards(record.hero.cards)
    camera_cards = {c for s in record.seats if s.hole_cam for c in parse_cards(s.hole_cam)}  # only for the reuse check
    state = NoLimitTexasHoldem.create_state(
        _AUTOMATIONS, True, stakes.ante, (stakes.sb, stakes.bb), stakes.bb,
        tuple(start[seat] for seat in order), len(order), mode=Mode.CASH_GAME,
    )
    while state.hole_dealee_index is not None:
        seat = order[state.hole_dealee_index]
        state.deal_hole("".join(hero_cards) if seat == hero else "????")

    steps: list[Step] = []
    revealed: dict[int, list[str]] = {}

    def snap() -> Snapshot:
        acting = state.actor_index is not None
        stacks = {seat: state.stacks[index_of[seat]] for seat in order}
        return Snapshot(
            street=STREETS[state.street_index] if state.street_index is not None else "showdown",
            board=[repr(c) for cards in state.board_cards for c in cards],
            pot=state.total_pot_amount,
            pots=[p for p in state.pot_amounts if p] if state.status else [],
            stacks=stacks,
            committed={seat: (start[seat] - stacks[seat]) if state.status else 0 for seat in order},
            actor=order[state.actor_index] if acting else None,
            to_call=state.checking_or_calling_amount if acting else None,
            min_raise_to=state.min_completion_betting_or_raising_to_amount if acting else None,
            max_raise_to=state.max_completion_betting_or_raising_to_amount if acting else None,
        )

    def add_step(ev: Event | None, before: Snapshot, label: str, seat: int | None = None) -> None:
        steps.append(Step(
            seq=ev.seq if ev else None, raw=ev.raw if ev else "", kind=ev.kind if ev else "show",
            seat=ev.seat if ev else seat, verb=ev.verb if ev else None, label=label,
            time=ev.time if ev else None, before=before, after=snap(),
        ))

    # Parse everything up front; reveals (show/muck) must come after all actions and board cards.
    events: list[Event] = []
    reveals: dict[int, Event] = {}
    deferred: _Stop | None = None
    for seq, raw in enumerate(record.events):
        try:
            ev = parse_event(seq, raw)
        except ValueError as e:
            deferred = _Stop(seq, "unreadable_event", str(e))
            break
        if ev.seat is not None and ev.seat not in index_of:
            deferred = _Stop(seq, "unknown_seat", f"seat {ev.seat} is not at the table")
            break
        if ev.kind in _REVEALS:
            if ev.seat in reveals:
                deferred = _Stop(seq, "duplicate_reveal", f"seat {ev.seat} shows or mucks twice")
                break
            reveals[ev.seat] = ev
        elif reveals:
            deferred = _Stop(seq, "reveal_not_last", "show/muck lines must come after all actions and board cards")
            break
        else:
            events.append(ev)

    def settle_showdown() -> None:
        while state.showdown_index is not None:
            seat = order[state.showdown_index]
            before = snap()
            ev = reveals.pop(seat, None)
            if ev is None and seat == hero:
                state.show_or_muck_hole_cards("".join(hero_cards))
                revealed[seat] = hero_cards
                add_step(None, before, f"{record.alias(seat)} shows {' '.join(hero_cards)}", seat)
            elif ev is None:
                raise _Stop(None, "showdown_missing",
                            f"{record.alias(seat)} (seat {seat}) reached showdown but no 'show' or 'muck' was recorded")
            elif ev.kind == "muck":
                state.show_or_muck_hole_cards(False)
                add_step(ev, before, f"{record.alias(seat)} mucks")
            else:
                if seat == hero and sorted(ev.cards) != sorted(hero_cards):
                    raise _Stop(ev.seq, "hero_cards_mismatch", "Hero shows different cards than were dealt")
                in_play = set(before.board) | {c for s, cs in revealed.items() for c in cs} | (
                    set() if seat == hero else set(hero_cards))
                if dup := in_play & set(ev.cards):
                    raise _Stop(ev.seq, "card_reused", f"card already in play: {', '.join(sorted(dup))}")
                state.show_or_muck_hole_cards("".join(ev.cards))
                revealed[seat] = ev.cards
                add_step(ev, before, f"{record.alias(seat)} shows {' '.join(ev.cards)}")

    def apply_board(ev: Event) -> None:
        settle_showdown()  # all-in hands may reveal before the runout
        before = snap()
        if state.actor_index is not None:
            raise _Stop(ev.seq, "board_out_of_order",
                        f"{ev.street} dealt while {record.alias(before.actor)} (seat {before.actor}) still has to act")
        if state.street_index is None:
            raise _Stop(ev.seq, "board_out_of_order", f"no more board cards are dealt in this hand")
        expected = STREETS[state.street_index]
        if ev.street != expected:
            raise _Stop(ev.seq, "wrong_street", f"expected the {expected}, got the {ev.street}")
        in_play = set(before.board) | set(hero_cards) | camera_cards | {c for cs in revealed.values() for c in cs}
        if dup := in_play & set(ev.cards):
            raise _Stop(ev.seq, "card_reused", f"card already in play: {', '.join(sorted(dup))}")
        while state.can_burn_card("??"):
            state.burn_card("??")
        if not state.can_deal_board("".join(ev.cards)):
            raise _Stop(ev.seq, "board_out_of_order", f"the {ev.street} cannot be dealt now")
        state.deal_board("".join(ev.cards))
        add_step(ev, before, f"{ev.street.capitalize()} {' '.join(ev.cards)}")

    def apply_action(ev: Event) -> None:
        if state.actor_index is None:
            raise _Stop(ev.seq, "no_action_pending", "no one can act now; the next event should deal a card")
        actor = order[state.actor_index]
        if ev.seat != actor:
            raise _Stop(ev.seq, "out_of_turn", f"it is {record.alias(actor)} (seat {actor})'s turn, not seat {ev.seat}")
        before = snap()
        name = record.alias(actor)
        to_call = state.checking_or_calling_amount
        facing = max(state.bets) > 0
        verb = ev.verb
        if verb == "allin":
            if state.can_complete_bet_or_raise_to():
                verb, amount = ("raise" if facing else "bet"), state.max_completion_betting_or_raising_to_amount
            elif to_call:
                verb = "call"
            else:
                raise _Stop(ev.seq, "illegal_allin", f"{name} cannot go all-in here")
        else:
            amount = ev.amount

        if verb == "fold":
            if not to_call:
                raise _Stop(ev.seq, "illegal_fold", f"{name} has nothing to call; record a check")
            state.fold()
            label = f"{name} folds"
        elif verb == "check":
            if to_call:
                raise _Stop(ev.seq, "illegal_check", f"{name} faces {to_call} to call and cannot check")
            state.check_or_call()
            label = f"{name} checks"
        elif verb == "call":
            if not to_call:
                raise _Stop(ev.seq, "illegal_call", "nothing to call; record a check")
            if ev.amount is not None and ev.amount != to_call:
                raise _Stop(ev.seq, "wrong_call_amount", f"{name} calls {to_call}, not {ev.amount}")
            state.check_or_call()
            label = f"{name} calls {to_call}"
        else:
            if verb == "bet" and facing:
                raise _Stop(ev.seq, "bet_when_facing", "there is already a bet; record 'raise N'")
            if verb == "raise" and not facing:
                raise _Stop(ev.seq, "raise_without_bet", "there is no bet to raise; record 'bet N'")
            if not state.can_complete_bet_or_raise_to(amount):
                raise _Stop(ev.seq, "illegal_amount",
                            f"{name} cannot {verb} to {amount}; legal range is {before.min_raise_to}..{before.max_raise_to}")
            state.complete_bet_or_raise_to(amount)
            label = f"{name} {'raises to' if facing else 'bets'} {amount}"
        if state.stacks[index_of[actor]] == 0 and verb != "fold":
            label += " (all-in)"
        add_step(ev, before, label)

    try:
        for ev in events:
            if not state.status:
                raise _Stop(ev.seq, "hand_over", "the hand is already over; extra event")
            if ev.kind == "board":
                apply_board(ev)
            else:
                apply_action(ev)
        if deferred:
            raise deferred
        settle_showdown()
        if reveals:
            ev = min(reveals.values(), key=lambda e: e.seq)
            raise _Stop(ev.seq, "not_at_showdown", f"seat {ev.seat} is not part of a showdown")
        if state.status:
            if state.actor_index is not None:
                seat = order[state.actor_index]
                need = f"{record.alias(seat)} (seat {seat}) to act"
            elif state.street_index is not None:
                need = f"the {STREETS[state.street_index]} to be dealt"
            else:
                need = "the showdown"
            raise _Stop(None, "hand_incomplete", f"the hand is not finished; waiting for {need}")
    except _Stop as stop:
        issues = [stop.issue]
        if stop.issue.seq is not None and stop.issue.seq + 1 < len(record.events):
            issues.append(Issue(seq=stop.issue.seq, code="replay_stopped",
                                message=f"events after #{stop.issue.seq} were not replayed"))
        return Replay(status="invalid", steps=steps, issues=issues, payoffs=None, revealed=revealed,
                      board=_last_board(steps))

    payoffs = {seat: state.payoffs[index_of[seat]] for seat in order}
    issues = [
        Issue(seq=None, code="result_mismatch",
              message=f"recorded result for seat {seat} is {net:+}, replay gives {payoffs.get(seat, 0):+}")
        for seat, net in (record.result or {}).items()
        if payoffs.get(seat) != net
    ]
    return Replay(status="invalid" if issues else "verified", steps=steps, issues=issues, payoffs=payoffs,
                  revealed=revealed, board=_last_board(steps))


def _last_board(steps: list[Step]) -> list[str]:
    for step in reversed(steps):
        if step.after.board:
            return step.after.board
    return []
