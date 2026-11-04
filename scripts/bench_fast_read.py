"""Time the fast reader on one video, both in fresh processes and warm in one process.

    python scripts/bench_fast_read.py examples/real/wpt_rolling_thunder_fold.webm --cold 5 --runs 7

A fresh process pays for starting Python, the imports and loading the OCR model; a warm process (a server, or one
command reading many videos) pays for those once. "cores" is CPU time over wall time while reading: how many cores
a read kept busy on average. Every read must give the same hand as the first one, or the script exits 1.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from poker_memory.reader import Reading, read_hand

FRESH = """
import json, sys, time
start = time.perf_counter()
from pathlib import Path
from poker_memory.reader import read_hand
imported = time.perf_counter()
r = read_hand(Path(sys.argv[1]))
print(json.dumps({"import": imported - start, "read": time.perf_counter() - imported,
                  "hand": r.record.model_dump_json() + repr((r.stakes, r.issues, r.wins))}))
"""


def _hand(r: Reading) -> str:
    return r.record.model_dump_json() + repr((r.stakes, r.issues, r.wins))


def _spread(values: list[float]) -> str:
    return f"median {statistics.median(values):.3f} s, range {min(values):.3f}-{max(values):.3f} s"


def main() -> int:
    parser = argparse.ArgumentParser(description="Time the fast reader on one video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--cold", type=int, default=5, help="fresh processes to time")
    parser.add_argument("--runs", type=int, default=7, help="warm reads to time")
    args = parser.parse_args()

    first = read_hand(args.video)
    want, differ = _hand(first), 0

    walls, imports, reads = [], [], []
    for _ in range(args.cold):
        start = time.perf_counter()
        out = subprocess.run([sys.executable, "-c", FRESH, str(args.video)], capture_output=True, text=True, check=True)
        walls.append(time.perf_counter() - start)
        fresh = json.loads(out.stdout.strip().splitlines()[-1])
        imports.append(fresh["import"])
        reads.append(fresh["read"])
        differ += fresh["hand"] != want
    if args.cold:
        print(f"fresh process, {args.cold} runs: {_spread(walls)}")
        print(f"  of which imports {statistics.median(imports):.3f} s, first read {statistics.median(reads):.3f} s")

    totals, cores, laps = [], [], {}
    for _ in range(args.runs):
        cpu, start = time.process_time(), time.perf_counter()
        reading = read_hand(args.video)
        totals.append(time.perf_counter() - start)
        cores.append((time.process_time() - cpu) / totals[-1])
        differ += _hand(reading) != want
        for name, seconds in reading.seconds.items():
            laps.setdefault(name, []).append(seconds)
    if args.runs:
        print(f"warm, {args.runs} runs: {_spread(totals)}; {statistics.median(cores):.1f} of {os.cpu_count()} cores busy")
        print("  " + ", ".join(f"{name} {statistics.median(v):.3f}" for name, v in laps.items() if name != "total"))
    print(f"{first.pictures} pictures read, {first.ocr_reads} OCR reads; "
          + ("every read gave the same hand" if not differ else f"{differ} reads gave a different hand"))
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
