"""Read-only review of the pinned, manually transcribed WPT example."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .engine import replay
from .model import HandRecord, SessionDoc, Source, Stakes, parse_cards
from .odds import bluff_share_what_if, pot_odds, showdown, winnable_if_call
from .store import Conflict, NotFound, StoredHand, StoredSession

KEY = "wpt-rolling-thunder-final-table"
CASE_SHA256 = "f24b55f1ca09bc37decd1d443ff3f4db8ea05249764c39966a6d1825999e1e18"
VIDEO_SHA256 = "5d0d3f43e883affe89413562a60a5f11ec7aa69afc295defd211db25f51b4f79"
Lens = Literal["decision", "broadcast"]


class Scenario(BaseModel):
    share_percent: int
    call_advantage: float


class Decision(BaseModel):
    hero: str
    opponent: str
    hero_cards: list[str]
    board: list[str]
    pot_before_call: int
    extra_call: int
    called_final_pot: int
    break_even_share: float
    fold_future_ev: int = 0
    stack_if_fold: int
    stack_if_losing_call: int
    scenarios: list[Scenario]


class Moment(BaseModel):
    seq: int
    label: str
    street: str
    time_ms: int
    time_text: str
    pot_after: int


class PlayerContext(BaseModel):
    alias: str
    position: str
    position_kind: Literal["inferred"] = "inferred"
    initial_stack: int
    observed_actions: int
    hand_count: int = 1
    long_run_profile: Literal["unknown"] = "unknown"


class Hypothesis(BaseModel):
    title: str
    observation: str
    possible_explanation: str
    limitation: str
    evidence_needed: str
    kind: Literal["hypothesis"] = "hypothesis"


class Hindsight(BaseModel):
    opponent_cards: list[str]
    hero_hand: str
    opponent_hand: str
    actual_hero_net: int
    known_card_call_net: int
    additional_loss_avoided: int
    video_url: str


class ReportedContext(BaseModel):
    title: str
    text: str
    author: str
    published: str
    url: str
    kind: Literal["attributed_report"] = "attributed_report"
    retrieval: str = "Selected public search-index excerpt; full article fetch unavailable during verification."


class Review(BaseModel):
    session_key: str
    revision: int
    case_sha256: str
    lens: Lens
    source: Source
    decision: Decision
    timeline: list[Moment]
    players: list[PlayerContext]
    hypotheses: list[Hypothesis]
    reported_context: list[ReportedContext]
    limitations: list[str]
    hindsight: Hindsight | None = None


def case_digest(doc: SessionDoc) -> str:
    payload = json.dumps(doc.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def call_advantage(share: float, call_cost: int, final_pot: int) -> float:
    if not math.isfinite(share):
        raise ValueError("scenario share must be finite")
    scenario = bluff_share_what_if(share)
    pot_odds(call_cost, final_pot)
    return scenario.win * final_pot - call_cost


def decision_from_record(record: HandRecord, stakes: Stakes) -> Decision:
    # Camera cards must not enter the decision-time replay, even as card-reuse constraints.
    decision_record = record.model_copy(update={
        "seats": [s.model_copy(update={"hole_cam": None}) for s in record.seats],
    })
    checked = replay(decision_record, stakes)
    if checked.status != "verified":
        raise Conflict("review needs a complete verified replay")
    last = checked.steps[-1]
    state = last.before
    if (last.verb != "fold" or last.seat != record.hero.seat or state.street != "river"
            or len(state.board) != 5 or not state.to_call):
        raise Conflict("review supports a completed river fold facing a bet only")
    opponents = {s.seat for s in checked.steps if s.verb == "allin"}
    if len(opponents) != 1 or record.hero.seat in opponents:
        raise Conflict("review requires one observed opponent shove")
    opponent = opponents.pop()
    cost, final_pot = winnable_if_call(
        state.committed, record.hero.seat, state.to_call, state.stacks[record.hero.seat])
    odds = pot_odds(cost, final_pot)
    return Decision(
        hero=record.alias(record.hero.seat), opponent=record.alias(opponent),
        hero_cards=parse_cards(record.hero.cards), board=state.board,
        pot_before_call=state.pot, extra_call=cost, called_final_pot=final_pot,
        break_even_share=odds.required_equity,
        stack_if_fold=state.stacks[record.hero.seat],
        stack_if_losing_call=state.stacks[record.hero.seat] - cost,
        scenarios=[Scenario(share_percent=p, call_advantage=call_advantage(p / 100, cost, final_pot))
                   for p in range(101)],
    )


def verify_case(session: StoredSession, hand: StoredHand, video: Path | None) -> str:
    if session.status != "ended":
        raise Conflict("demo is post-session only; this session has not ended")
    if session.info.consent != "open_license" or not session.info.source:
        raise Conflict("this demo requires the credited, openly licensed source")
    if (hand.status != "verified" or hand.revision != hand.active_revision
            or hand.latest_revision != hand.active_revision):
        raise Conflict("demo requires the active verified revision with no pending corrections")
    doc = SessionDoc(session=session.info, hands=[hand.record])
    digest = case_digest(doc)
    if digest != CASE_SHA256:
        raise Conflict("reviewed transcription or source changed; curated analysis is unavailable")
    if video is None or not video.is_file():
        raise NotFound("reviewed source video unavailable; run the isolated demo setup")
    with video.open("rb") as stream:
        digest_video = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest_video != VIDEO_SHA256:
        raise Conflict("source video changed; refusing mismatched evidence")
    return digest


def build_review(session: StoredSession, hand: StoredHand, video: Path | None, lens: Lens) -> Review:
    digest = verify_case(session, hand, video)
    record = hand.record
    checked = replay(record, session.info.stakes)
    if checked.status != "verified" or checked.payoffs is None:
        raise Conflict("current engine cannot verify the reviewed hand")
    decision = decision_from_record(record, session.info.stakes)
    timeline = []
    for step in checked.steps:
        if step.seq is None or step.time is None:
            raise Conflict("reviewed action timing is missing")
        timeline.append(Moment(seq=step.seq, label=step.label, street=step.before.street,
                               time_ms=step.time.earliest_ms, time_text=step.time.text,
                               pot_after=step.after.pot))
    hindsight = None
    if lens == "broadcast":
        camera = next(s.hole_cam for s in record.seats if s.alias == decision.opponent)
        if not camera:
            raise Conflict("broadcast camera evidence is unavailable")
        cards = parse_cards(camera)
        result = showdown(tuple(decision.hero_cards), tuple(cards), decision.board)
        if result.winner != "villain":
            raise Conflict("known-card outcome no longer matches the curated explanation")
        net = checked.payoffs[record.hero.seat]
        hindsight = Hindsight(
            opponent_cards=cards, hero_hand=result.hero_hand, opponent_hand=result.villain_hand,
            actual_hero_net=net, known_card_call_net=net - decision.extra_call,
            additional_loss_avoided=decision.extra_call, video_url="/api/demo/wpt/video")
    source = session.info.source
    if source is None:
        raise Conflict("source credit is missing")
    return Review(
        session_key=session.info.key, revision=hand.revision, case_sha256=digest, lens=lens, source=source,
        decision=decision, timeline=timeline,
        players=[PlayerContext(alias=s.alias, position=s.note, initial_stack=s.stack,
                               observed_actions=sum(step.seat == s.seat for step in checked.steps))
                 for s in record.seats if s.seat in (1, 2)],
        hypotheses=[
            Hypothesis(title="A strong hand is not an automatic call",
                       observation="Ian bets the river, faces Joe's shove, then folds.",
                       possible_explanation="Ian may assign fewer winning matchups than the call price requires.",
                       limitation="His post-hand account is reported reasoning, not a measured range or proof of optimality.",
                       evidence_needed="Independently reviewed comparable river shoves and evidence of decision-time beliefs."),
            Hypothesis(title="The betting story may matter",
                       observation="Joe calls preflop and flop, bets the turn, then shoves over the river bet.",
                       possible_explanation="Ian may treat this line as stronger than a typical single river bet.",
                       limitation="One line is not a measured bluff frequency or a stable player tendency.",
                       evidence_needed="A larger sample of this same line, including reviewed showdowns and selection bias."),
            Hypothesis(title="Tournament pressure is a possible factor",
                       observation="This is a five-player tournament final table, not a cash game.",
                       possible_explanation="Payout incentives could change chip-risk preferences.",
                       limitation="Exact physical stacks are disputed; payout and validated ICM inputs are unavailable.",
                       evidence_needed="Verified physical-table stacks, payout schedule and a validated tournament model."),
        ],
        reported_context=[
            ReportedContext(title="Published correction: broadcast stacks disputed",
                            text="PokerNews attributes a correction to McKeehen: the displayed stacks were wrong and the players started with similar stacks. We retain the broadcast record, not replace it with an invented exact stack. Remaining stacks and arithmetic here are conditional on that record; no precise actual survival or ICM conclusion follows.",
                            author="Frank Op de Woerd / PokerNews", published="6 March 2018; updated 12 March 2018",
                            url="https://www.pokernews.com/news/2018/03/ian-steinman-kings-set-fold-mckeehen-30175.htm"),
            ReportedContext(title="Ian's reported reasoning after the hand",
                            text="In Brian Pempus's Card Player interview, Ian described initially thinking Joe was almost never bluffing, referring to pay jumps/ICM and Joe's skill. This is Ian's retrospective account, not a measured bluff frequency or a validated player profile. PokerNews also relays Ian's uncertainty that the fold might still have been poor.",
                            author="Brian Pempus / Card Player", published="12 March 2018",
                            url="https://www.cardplayer.com/poker-news/22600-poker-pro-unsure-if-epic-fold-was-a-smart-play"),
        ],
        limitations=[
            "Reviewed manual transcription, not a fully accurate automated read: WPT calibration remains 26/27, with a known +3.1s fold-time error.",
            "Video clocks are approximate (about 1-3s) and may include edits; they are not reliable thinking times.",
            "Logical seats and positions are inferred, not printed physical chair numbers.",
            "One hand only. Long-run profile, measured shove/bluff frequency and opponent range are unknown.",
            "What-if p is the share of shoves Ian beats, including bluffs or weaker value; ties excluded, remaining hands beat Ian.",
            "Conditional chip EV only: no payout/ICM estimate. Prior broad-range solver +EV is not a player-specific verdict.",
            "Under the transcribed broadcast stacks only, folding leaves Ian 4,175,000 chips and a losing call 2,035,000. Published reporting disputes the starting stacks; actual survival and ICM implications are not established.",
            "A good realized outcome does not establish optimal play against an unknown range.",
        ],
        hindsight=hindsight,
    )
