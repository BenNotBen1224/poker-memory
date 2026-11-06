"""Seed the reviewed WPT example into a NEW isolated data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_memory.memory import PokerMemory
from poker_memory.model import SessionDoc
from poker_memory.review import CASE_SHA256, build_review, case_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    if (args.data / "poker_memory.db").exists():
        parser.error("database already exists; choose a fresh isolated directory")
    source = Path(__file__).resolve().parents[1] / "examples" / "real"
    doc = SessionDoc.load(source / "wpt_rolling_thunder.yaml")
    if case_digest(doc) != CASE_SHA256:
        parser.error("reviewed transcription changed")
    memory = PokerMemory.open(args.data)
    memory.import_session(doc, source)
    session = memory.end_session(doc.session.key)
    review = build_review(session, memory.store.hand(session.info.key, 1),
                          memory.video_path(session.info.key), "decision")
    receipt = {"data": str(args.data.resolve()), "case_sha256": review.case_sha256,
               "session": session.info.key, "revision": review.revision, "status": "verified"}
    (args.data / "setup-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
