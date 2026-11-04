"""Application facade: import, end, correct, and ask about ended sessions.

Every number in an answer comes from a recorded tool run against one exact hand
revision. Opponent cards revealed later are reported as hindsight and never used.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .engine import Step
from .gto import SAMPLES, NotSolved, RiverGTO, river_gto
from .interpret import Query, parse_question
from .model import HandRecord, SessionDoc, VideoTime, parse_cards
from .odds import bluff_share_what_if, equity, pot_odds, river_partition, showdown, winnable_if_call
from .search import find_decision, hero_cards, search, summary, you_label
from .store import Conflict, NotFound, StoredHand, StoredSession, Store, ToolRun

FactKind = Literal["observed", "derived", "estimated", "unknown"]


class Fact(BaseModel):
    name: str
    kind: FactKind
    text: str
    value: Any = None


class Evidence(BaseModel):
    label: str
    start_ms: int
    end_ms: int


class Candidate(BaseModel):
    index: int
    revision: int
    summary: str


class Decision(BaseModel):
    seq: int
    street: str
    label: str
    time: VideoTime | None


class Answer(BaseModel):
    kind: Literal["analysis", "candidates", "not_found", "needs_fix", "refused"]
    text: str
    understood: str = ""
    conversation_id: int | None = None
    session_key: str | None = None
    hand: int | None = None
    revision: int | None = None
    decision: Decision | None = None
    facts: list[Fact] = []
    candidates: list[Candidate] = []
    evidence: list[Evidence] = []
    video_url: str | None = None
    assumptions: list[str] = []
    tools: list[ToolRun] = []


LIVE_REFUSAL = ("This session is still marked live. Poker Memory only reviews hands after a session has ended, "
                "so it never gives help during play. End the session first.")
NEED_ASSUMPTION = ("Your equity is unknown: I need an assumption about his hands. Give a range "
                   "(\"if he has 77, AK\") or, on the river, a what-if (\"if he bluffs 10%\").")


def _gto_text(g: RiverGTO, currency: str) -> str:
    unit = 10 ** max(0, len(str(g.pot)) - 4)  # four digits of the pot's size; hides 32-bit float noise

    def chips(x: float, sign: str = "+") -> str:
        return f"{round(x / unit) * unit:{sign},} {currency}"

    text = f"Solver (river only, chips counted from the start of the river): with {g.hand} "
    if g.ev_lost <= g.exploitability * g.pot:
        text += (f"your {g.moves[g.taken]} ({chips(g.ev[g.taken])}) was as good as the best move, "
                 f"{g.moves[g.best]} ({chips(g.ev[g.best])}), at the solver's accuracy.")
    else:
        text += (f"the best move was {g.moves[g.best]}, worth {chips(g.ev[g.best])}; your {g.moves[g.taken]} "
                 f"got {chips(g.ev[g.taken])}, about {chips(g.ev_lost, '')} less.")
    if g.beats_you is not None:
        text += f" About {g.beats_you:.0%} of the hands he plays this way beat yours."
    return text


def _gto_assumptions(g: RiverGTO, names: dict[int, str]) -> list[str]:
    sizes = ", ".join(f"{names[s]} {n}" for s, n in g.range_hands.items())
    out = [f"Solver ranges come from the betting only: both players start with every hand, and each call keeps "
           f"the hands with enough equity against the other player's range to pay the price ({SAMPLES} random "
           f"boards per call before the turn). Cards shown later are not used. River hands: {sizes}.",
           f"The solver could check or bet {', '.join(f'{x:,}' for x in g.bet_sizes)} (the last is all-in), "
           "with up to two raises after that.",
           "Chips only: tournament payout effects (ICM) are ignored.",
           f"Solved in {g.seconds:.2f} s ({g.iterations} rounds); a perfect opponent could win at most "
           f"{g.exploitability:.2%} of the pot more against this solution."]
    if not g.hero_in_range:
        out.append("Your hand falls outside the range the betting rule gives you, so it was added with a tiny "
                   "weight to get a plan for it.")
    return out


def _gap(a: VideoTime, b: VideoTime) -> tuple[int, int]:
    return max(0, b.earliest_ms - a.latest_ms), b.latest_ms - a.earliest_ms


def _secs(lo: int, hi: int) -> str:
    text = f"{lo / 1000:.1f}-{hi / 1000:.1f} s"
    if lo >= 60_000:
        minutes, seconds = divmod(round((lo + hi) / 2000), 60)
        text += f" (about {minutes} min {seconds} s)"
    return text


def _hand_words(label: str) -> str:
    return f"a {label.lower()}" if label in {"High card", "Straight", "Flush", "Full house", "Straight flush"} else label.lower()


class PokerMemory:
    def __init__(self, store: Store, data_dir: Path):
        self.store = store
        self.data_dir = data_dir
        self._solved: dict[tuple, RiverGTO | NotSolved] = {}  # (session, hand, revision, seq); the solver is deterministic

    @classmethod
    def open(cls, data_dir: str | Path) -> "PokerMemory":
        data_dir = Path(data_dir)
        return cls(Store(data_dir / "poker_memory.db"), data_dir)

    # writing

    def import_session(self, doc: SessionDoc, source_dir: str | Path | None = None) -> StoredSession:
        """Store a session. A consented video next to the source file is copied into the data folder."""
        video = None
        if doc.session.video:
            if source_dir is None or not (Path(source_dir) / doc.session.video).is_file():
                raise NotFound(f"video {doc.session.video!r} not found next to the session file")
            video = Path(source_dir) / doc.session.video
        stored = self.store.import_session(doc)
        if video:
            target = self.video_path(doc.session.key, doc.session.video)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video, target)
        return stored

    def end_session(self, key: str) -> StoredSession:
        return self.store.end_session(key)

    def correct_hand(self, key: str, index: int, base_revision: int, record: HandRecord, note: str) -> StoredHand:
        return self.store.correct(key, index, record, base_revision, note)

    def video_path(self, key: str, name: str | None = None) -> Path | None:
        name = name or self.store.session(key).info.video
        return self.data_dir / "media" / key / Path(name).name if name else None

    # asking

    def ask(self, question: str, session_key: str | None = None, conversation_id: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("ask a question")
        state: dict = {}
        if conversation_id is not None:
            key, state = self.store.conversation(conversation_id)
            if session_key and session_key != key:
                raise Conflict("that conversation belongs to another session")
        else:
            key = session_key or self._default_session()
        session = self.store.session(key)
        if session.status != "ended":
            return Answer(kind="refused", text=LIVE_REFUSAL, session_key=key, conversation_id=conversation_id)

        hands = self.store.hands(key)
        q = parse_question(question, sorted({s.alias for h in hands for s in h.record.seats}))
        answer, runs, new_state = self._answer(session, hands, q, state)
        answer.session_key, answer.understood, answer.tools = key, q.describe(), runs
        if conversation_id is None:
            conversation_id = self.store.new_conversation(key)
        answer.conversation_id = conversation_id
        self.store.save_turn(conversation_id, question, answer.model_dump(mode="json"), new_state, runs)
        return answer

    def _default_session(self) -> str:
        sessions = self.store.sessions()
        if not sessions:
            raise NotFound("no sessions yet; import one first")
        ended = [s for s in sessions if s.status == "ended"]
        return (ended or sessions)[0].info.key

    def _answer(self, session: StoredSession, hands: list[StoredHand], q: Query, state: dict):
        by_index = {h.index: h for h in hands}
        offered: list[int] = state.get("candidates", [])
        focus: dict | None = state.get("focus")
        keep = {"candidates": offered, "focus": focus, "assumption": state.get("assumption")}

        def candidates(found: list[StoredHand], text: str):
            listed = [Candidate(index=h.index, revision=h.revision, summary=summary(h)) for h in found]
            return Answer(kind="candidates", text=text, candidates=listed), [], {**keep, "candidates": [h.index for h in found]}

        pick = q.ordinal or (q.number if offered else None)
        if q.hand_index is not None or (q.number and not offered):
            n = q.hand_index or q.number
            if n not in by_index:
                return Answer(kind="not_found", text=f"There is no hand {n} in this session."), [], keep
            target = by_index[n]
        elif pick and offered:
            pos = len(offered) if pick == -1 else pick
            if not 1 <= pos <= len(offered):
                return Answer(kind="not_found", text=f"I only offered {len(offered)} hands."), [], keep
            target = by_index[offered[pos - 1]]
        elif q.ordinal == -1 or q.identifies_hand() or (not focus and (q.street or q.hero_verb or q.faced_raise)):
            if q.ordinal == -1:
                q.latest = True
            found = search(hands, q)
            if not found:
                return Answer(kind="not_found", text=f"No hand matches ({q.describe()})."), [], keep
            if len(found) > 1:
                return candidates(found, f"{len(found)} hands match ({q.describe()}). Which one? "
                                         "Say \"the first one\", \"#2\", or \"hand N\".")
            target = found[0]
        elif focus and focus["hand"] in by_index:
            target = by_index[focus["hand"]]
        else:
            return candidates(hands, "Which hand do you mean? Mention the cards, the street, what you did, "
                                     "or pick one of these.")

        if target.status != "verified":
            issue = target.replay.issues[0].message if target.replay.issues else "it does not replay"
            return Answer(kind="needs_fix", hand=target.index, revision=target.revision,
                          text=f"Hand {target.index} has no verified version yet ({issue}). Fix it first."), [], keep

        step = None
        same_hand = bool(focus and focus["hand"] == target.index)
        if same_hand and not (q.street or q.hero_verb or q.faced_raise):
            step = next((s for s in target.replay.steps if s.seq == focus.get("seq") and s.kind == "action"
                         and s.seat == target.record.hero.seat), None)
        step = step or find_decision(target, q)
        if step is None:
            return Answer(kind="not_found", hand=target.index, revision=target.revision,
                          text=f"In hand {target.index} I can't find a decision of yours ({q.describe()})."), [], keep

        assumption, reused = None, False
        if q.villain_range:
            assumption = {"range": q.villain_range}
        elif q.bluff_share is not None:
            assumption = {"bluff": q.bluff_share}
        elif same_hand and state.get("assumption"):
            assumption, reused = state["assumption"], True

        answer, runs = self._analyze(session, target, step, assumption, reused, q.range_error)
        return answer, runs, {"candidates": offered, "focus": {"hand": target.index, "seq": step.seq},
                              "assumption": assumption}

    def _analyze(self, session: StoredSession, h: StoredHand, step: Step, assumption: dict | None, reused: bool,
                 range_error: str | None) -> tuple[Answer, list[ToolRun]]:
        key, info = session.info.key, session.info
        hero = h.record.hero.seat
        alias = h.record.alias
        cards = hero_cards(h)
        steps = h.replay.steps
        i = next(n for n, s in enumerate(steps) if s.seq == step.seq)
        before, board = step.before, step.before.board
        faced = next((s for s in reversed(steps[:i]) if s.kind == "action" and s.seat != hero
                      and s.before.street == before.street), None)
        prior = steps[steps.index(faced) - 1] if faced and steps.index(faced) > 0 else None

        def run(tool: str, inp: dict, out: dict) -> None:
            runs.append(ToolRun(tool=tool, hand_index=h.index, revision=h.revision, input=inp, output=out))

        runs: list[ToolRun] = []
        facts: list[Fact] = []
        assumptions: list[str] = []
        run("replay", {"session": key, "hand": h.index, "revision": h.revision},
            {"status": h.replay.status, "engine_version": h.replay.engine_version, "decision_seq": step.seq})

        facts.append(Fact(name="hero_cards", kind="observed", text=f"You held {' '.join(cards)}.", value=cards))
        facts.append(Fact(name="board", kind="observed", value=board,
                          text=f"Board at this point: {' '.join(board)}." if board else "No board cards yet (preflop)."))
        if faced:
            facts.append(Fact(name="faced", kind="observed", text=f"{faced.label}.", value=faced.label))
        facts.append(Fact(name="your_action", kind="observed", text=f"Then {you_label(step.label, alias(hero))}.",
                          value=step.verb))

        po = None
        if before.to_call:
            call, winnable = winnable_if_call(before.committed, hero, before.to_call, before.stacks[hero])
            po = pot_odds(call, winnable)
            run("pot_odds", {"committed": before.committed, "to_call": before.to_call,
                             "hero_stack": before.stacks[hero]}, po.model_dump())
            facts.append(Fact(name="pot_odds", kind="derived", value=round(po.required_equity, 4),
                              text=f"Calling cost {call} to win a pot of {winnable}, so a call needed "
                                   f"{po.required_equity:.1%} equity."))
        else:
            facts.append(Fact(name="pot_odds", kind="unknown",
                              text="You were not facing a bet here, so there are no pot odds."))

        eq = None
        if range_error:
            facts.append(Fact(name="equity", kind="unknown", text=f"Equity is unknown: {range_error}."))
        elif assumption is None:
            facts.append(Fact(name="equity", kind="unknown", text=NEED_ASSUMPTION))
        elif "bluff" in assumption:
            share = assumption["bluff"]
            if len(board) != 5:
                facts.append(Fact(name="equity", kind="unknown",
                                  text="The bluff what-if only works on the river. Give a range instead."))
            else:
                part = river_partition(tuple(cards), board)
                eq = bluff_share_what_if(share)
                counts = {"beats_you": len(part.beats_hero), "ties": len(part.ties), "you_beat": len(part.loses)}
                run("bluff_what_if", {"share": share, "hero": cards, "board": board}, {**eq.model_dump(), **counts})
                facts.append(Fact(name="equity", kind="estimated", value=round(eq.equity, 4),
                                  text=f"If {share:.0%} of his hands are bluffs you beat and the rest beat you, "
                                       f"your equity is {eq.equity:.1%}."))
                assumptions.append(eq.assumption)
                assumptions.append(f"On this board {counts['beats_you']} two-card hands beat you, {counts['ties']} tie, "
                                   f"and {counts['you_beat']} lose to you; ties are left out of the what-if.")
        else:
            rng = assumption["range"]
            try:
                eq = equity(tuple(cards), rng, board)
            except ValueError as e:
                run("equity", {"hero": cards, "board": board, "range": rng}, {"error": str(e)})
                facts.append(Fact(name="equity", kind="unknown", text=f"Equity is unknown: {e}."))
            else:
                run("equity", {"hero": cards, "board": board, "range": rng}, eq.model_dump())
                how = (f"exact count over {eq.trials} hands" if eq.method == "exact" else
                       f"simulated {eq.trials} runouts, ±{1.96 * eq.std_error:.1%}, seed {eq.seed}")
                facts.append(Fact(name="equity", kind="estimated", value=round(eq.equity, 4),
                                  text=f"Against {rng} your equity is {eq.equity:.1%} ({how})."))
                assumptions.append(f"His range is {rng} ({eq.combos_used} hands left after removing known cards).")
        if reused and eq:
            assumptions.insert(0, "Using the assumption from your earlier question.")
        folded = {s.seat for s in steps[:i] if s.verb == "fold"}
        still_in = len(h.record.seats) - 1 - len(folded - {hero})
        if eq and still_in > 1:
            assumptions.append(f"Equity is against one range, but {still_in} opponents were still in the hand.")

        if po and eq:
            better = "call" if eq.equity >= po.required_equity else "fold"
            if step.verb in ("fold", "call"):
                ok = better == step.verb
                verdict = f"the {step.verb} was reasonable" if ok else f"a {better} would have been better"
            else:
                verdict = f"calling would {'' if better == 'call' else 'not '}have been profitable"
            facts.append(Fact(name="verdict", kind="estimated", value=better,
                              text=f"Under this assumption {verdict}: you needed {po.required_equity:.1%} "
                                   f"and had {eq.equity:.1%}."))
            if len(board) < 5:
                assumptions.append("Pot odds ignore betting on later streets.")
        spot = (key, h.index, h.revision, step.seq)
        if spot not in self._solved:
            try:
                self._solved[spot] = river_gto(steps, i, hero, cards, [s.seat for s in h.record.seats])
            except NotSolved as e:
                self._solved[spot] = e
        g = self._solved[spot]
        if isinstance(g, NotSolved):
            facts.append(Fact(name="gto", kind="unknown", text=f"No solver result for this spot: {g}."))
        else:
            run("gto_river", {"hero": cards, "board": h.replay.board, "decision_seq": step.seq}, g.model_dump())
            facts.append(Fact(name="gto", kind="estimated", value=g.model_dump(),
                              text=_gto_text(g, info.stakes.currency)))
            names = {s: "you" if s == hero else alias(s) for s in g.range_hands}
            assumptions += _gto_assumptions(g, names)

        evidence: list[Evidence] = []
        timing: dict = {}
        if faced and faced.time and prior and prior.time:
            lo, hi = _gap(prior.time, faced.time)
            timing["villain_ms"] = [lo, hi]
            facts.append(Fact(name="villain_time", kind="derived", value=[lo, hi],
                              text=f"{alias(faced.seat)} took {_secs(lo, hi)} before acting "
                                   f"(video {prior.time.text} to {faced.time.text})."))
        else:
            facts.append(Fact(name="villain_time", kind="unknown",
                              text="His thinking time is unknown: " + (
                                  "no video times were recorded for it." if faced else
                                  "no opponent acted before you on this street.")))
        if faced and faced.time and step.time:
            lo, hi = _gap(faced.time, step.time)
            timing["your_ms"] = [lo, hi]
            facts.append(Fact(name="your_time", kind="derived", value=[lo, hi],
                              text=f"You took {_secs(lo, hi)} to decide."))
        if timing:
            run("timing", {"times": {s.label: s.time.model_dump() for s in (prior, faced, step) if s and s.time}},
                timing)
        for s in (prior, faced, step):
            if s and s.time and info.video:
                evidence.append(Evidence(label=f"{s.time.text} {s.label}", start_ms=s.time.earliest_ms,
                                         end_ms=s.time.latest_ms))

        for seat, shown in h.replay.revealed.items():
            if seat != hero:
                facts.append(Fact(name="hindsight", kind="observed", value=shown,
                                  text=f"Later {alias(seat)} showed {' '.join(shown)}. That is hindsight and "
                                       "is not used in any number above."))
        for s in h.record.seats:
            if not s.hole_cam or s.seat == hero or s.seat in folded or s.seat in h.replay.revealed:
                continue
            theirs = parse_cards(s.hole_cam)
            facts.append(Fact(name="hole_cam", kind="observed", value=theirs,
                              text=f"A hole-card camera in the broadcast showed {alias(s.seat)} holding "
                                   f"{' '.join(theirs)}. You could not see that at the table; it is hindsight and "
                                   "is not used in any number above."))
            if len(h.replay.board) == 5:
                sd = showdown(tuple(cards), tuple(theirs), h.replay.board)
                run("showdown", {"hero": cards, "villain": theirs, "board": h.replay.board}, sd.model_dump())
                wins = {"villain": f"{alias(s.seat)}'s hand wins", "hero": "your hand wins", "tie": "it is a tie"}
                text = (f"On the final board {alias(s.seat)} has {_hand_words(sd.villain_hand)} and you have "
                        f"{_hand_words(sd.hero_hand)}: {wins[sd.winner]}.")
                if before.street == "river" and step.verb == "fold" and still_in == 1:
                    outcome = {"villain": "lost", "hero": "won", "tie": "split"}[sd.winner]
                    text += f" Calling here would have {outcome} the pot."
                facts.append(Fact(name="hindsight_showdown", kind="derived", value=sd.model_dump(), text=text))
        net = (h.replay.payoffs or {}).get(hero)
        if net is not None:
            facts.append(Fact(name="result", kind="derived", value=net,
                              text=f"Your result in this hand: {net:+} {info.stakes.currency}."))

        head = (f"Hand {h.index}, {before.street} (revision {h.revision}): "
                f"{you_label(step.label, alias(hero))}.")
        text = "\n".join([head] + [f.text for f in facts])
        return Answer(kind="analysis", text=text, hand=h.index, revision=h.revision,
                      decision=Decision(seq=step.seq, street=before.street, label=step.label, time=step.time),
                      facts=facts, evidence=evidence, assumptions=assumptions,
                      video_url=f"/api/sessions/{key}/video" if info.video else None), runs
