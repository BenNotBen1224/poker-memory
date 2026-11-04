"""Command line: poker-memory demo | import FILE | end KEY | ask QUESTION | read VIDEO... | serve"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .memory import PokerMemory
from .model import SessionDoc
from .store import Conflict

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "home_game.yaml"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="poker-memory")
    p.add_argument("--data", default=os.environ.get("POKER_MEMORY_DATA", "data"), help="data folder (default: data)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="import the synthetic example session and mark it ended")
    imp = sub.add_parser("import", help="import a session YAML file")
    imp.add_argument("file", type=Path)
    end = sub.add_parser("end", help="mark a session as ended so it can be reviewed")
    end.add_argument("key")
    ask = sub.add_parser("ask", help="ask a question about an ended session")
    ask.add_argument("question")
    ask.add_argument("--session")
    read = sub.add_parser("read", help="read one hand from each broadcast video (needs the 'vision' extra)")
    read.add_argument("videos", type=Path, nargs="+")
    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        os.environ["POKER_MEMORY_DATA"] = args.data
        uvicorn.run("poker_memory.api:app_from_env", factory=True, host=args.host, port=args.port)
        return
    if args.cmd == "read":
        _read(args.videos)
        return

    memory = PokerMemory.open(args.data)
    if args.cmd == "demo":
        doc = SessionDoc.load(EXAMPLE)
        try:
            memory.import_session(doc, EXAMPLE.parent)
        except Conflict:
            pass
        s = memory.end_session(doc.session.key)
        print(f"{s.info.key}: {len(memory.store.hands(s.info.key))} hands, {s.status}")
    elif args.cmd == "import":
        doc = SessionDoc.load(args.file)
        s = memory.import_session(doc, args.file.parent)
        for h in memory.store.hands(s.info.key):
            issues = "; ".join(i.message for i in h.replay.issues)
            print(f"hand {h.index}: {h.status}" + (f" ({issues})" if issues else ""))
    elif args.cmd == "end":
        print(memory.end_session(args.key).status)
    elif args.cmd == "ask":
        a = memory.ask(args.question, args.session)
        print(a.text)
        for c in a.candidates:
            print(f"  - {c.summary}")
        for s in a.assumptions:
            print(f"  assumption: {s}")


def _read(videos: list[Path]) -> None:
    """Print each video's stakes and hand as YAML, in the shapes a session file uses. The second video onwards
    reads faster, as the OCR model is already loaded."""
    import yaml

    from .reader import read_hand

    for n, video in enumerate(videos):
        reading = read_hand(video)
        if n:
            print("---")
        print(f"# {video.name}: read in {reading.seconds['total']:.2f} s from {reading.pictures} pictures")
        for issue in reading.issues:
            print(f"# {issue}")
        hand = {"stakes": reading.stakes.model_dump(), "hands": [reading.record.model_dump(exclude_none=True)]}
        print(yaml.safe_dump(hand, sort_keys=False), end="")


if __name__ == "__main__":
    main()
