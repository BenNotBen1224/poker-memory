"""Check the fast reader against a hand read by eye, frame by frame.

    python scripts/check_fast_read.py examples/real/wpt_rolling_thunder.yaml

Reads the session's video with `poker_memory.reader.read_hand` and compares the result with the YAML's first
hand: stakes, button, hero, each seat's name and hole cards, and every event, which must match word for word and
land within --slack seconds. Stacks are not compared because the broadcast never shows them. The record that was
read is also replayed through the engine, and its payoffs must equal the YAML's result. Exits 1 if anything
differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poker_memory.engine import replay
from poker_memory.model import SessionDoc, VideoTime, parse_cards
from poker_memory.reader import read_hand


def _event(line: str) -> tuple[str, float | None]:
    body, _, when = line.partition("@")
    body = " ".join(body.split()).lower().replace("all-in", "allin")
    return body, VideoTime.parse(when).earliest_ms / 1000 if when.strip() else None


def _cards(text: str | None) -> frozenset[str] | None:
    return frozenset(parse_cards(text)) if text else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the fast reader against a hand read by eye.")
    parser.add_argument("yaml", type=Path)
    parser.add_argument("--slack", type=float, default=3.0, help="how many seconds an event may be off")
    args = parser.parse_args()

    doc = SessionDoc.load(args.yaml)
    want = doc.hands[0]
    reading = read_hand(args.yaml.parent / doc.session.video)
    got = reading.record
    rows: list[tuple[str, str, str, bool]] = []

    def check(what: str, expected: object, actual: object, ok: bool | None = None) -> None:
        rows.append((what, str(expected), str(actual), expected == actual if ok is None else ok))

    s, r = doc.session.stakes, reading.stakes
    check("stakes", (s.sb, s.bb, s.ante), (r.sb, r.bb, r.ante))
    check("button", want.button, got.button)
    check("hero", (want.hero.seat, _cards(want.hero.cards)), (got.hero.seat, _cards(got.hero.cards)))
    seats = {seat.seat: seat for seat in got.seats}
    for seat in want.seats:
        mine = seats.get(seat.seat)
        check(f"seat {seat.seat}", (seat.alias.lower(), _cards(seat.hole_cam)),
              (mine.alias.lower(), _cards(mine.hole_cam)) if mine else None)
    check("players", len(want.seats), len(got.seats))

    def timed(what: str, expected: tuple[str, float | None], actual: tuple[str, float | None] | None) -> None:
        if actual is None:
            rows.append((what, expected[0], "missing", False))
            return
        off = actual[1] - expected[1] if expected[1] is not None and actual[1] is not None else None
        ok = expected[0] == actual[0] and off is not None and abs(off) <= args.slack
        shown = f"{actual[0]}  {off:+.1f} s" if off is not None else actual[0]
        rows.append((what, expected[0], shown, ok))

    want_dealt = VideoTime.parse(want.dealt_at).earliest_ms / 1000 if want.dealt_at else None
    got_dealt = VideoTime.parse(got.dealt_at).earliest_ms / 1000 if got.dealt_at else None
    timed("dealt", ("dealt", want_dealt), ("dealt", got_dealt))
    wanted, read = [_event(e) for e in want.events], [_event(e) for e in got.events]
    for i, expected in enumerate(wanted):
        timed(f"event {i + 1}", expected, read[i] if i < len(read) else None)
    for i in range(len(wanted), len(read)):
        rows.append((f"event {i + 1}", "nothing", read[i][0], False))
    check("result", want.result, replay(got, reading.stakes).payoffs)

    width = max(len(row[1]) for row in rows)
    for what, expected, actual, ok in rows:
        print(f"{'ok ' if ok else 'BAD'}  {what:9s} {expected:{width}s}  {actual}")
    bad = [row for row in rows if not row[3]]
    print(f"{len(rows) - len(bad)} of {len(rows)} checks pass; read in {reading.seconds['total']:.2f} s, "
          f"{reading.pictures} pictures, {reading.ocr_reads} OCR reads")
    for issue in reading.issues:
        print("issue:", issue)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
