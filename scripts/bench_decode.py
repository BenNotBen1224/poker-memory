"""Time the ways of getting pictures out of a recording, so the fast reader uses the cheapest one.

    python scripts/bench_decode.py VIDEO [--runs 5]

Prints one line per method: median seconds over the runs, frames produced, and frames per second.
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av


def list_keyframes(path: Path) -> list[float]:
    """Keyframe times in seconds, found by reading packet headers only (nothing is decoded)."""
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        return [float(p.pts * p.time_base) for p in box.demux(stream) if p.pts is not None and p.is_keyframe]


def decode_keyframes(path: Path, thread_type: str) -> int:
    count = 0
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        stream.thread_type = thread_type
        for packet in box.demux(stream):
            if packet.pts is None or not packet.is_keyframe:
                continue
            count += len(stream.codec_context.decode(packet))
        count += len(stream.codec_context.decode(None))
    return count


def decode_span(path: Path, start: float, end: float, threads: int = 0) -> int:
    """Decode every frame shown from `start` up to `end` (start should be a keyframe time).

    threads=0 lets the decoder use every core; threads=1 keeps it to one core so many spans can run side by side.
    """
    count = 0
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        if threads == 1:
            stream.thread_type = "NONE"
        else:
            stream.thread_type = "AUTO"
            stream.thread_count = threads
        box.seek(int(start / stream.time_base), stream=stream, backward=True)
        for frame in box.decode(stream):
            if frame.time is None or frame.time < start - 1e-3:
                continue
            if frame.time >= end:
                break
            count += 1
    return count


def decode_all(path: Path) -> int:
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        stream.thread_type = "AUTO"
        return sum(1 for _ in box.decode(stream))


def timed(runs: int, work) -> tuple[float, int]:
    seconds, produced = [], 0
    for _ in range(runs):
        began = time.perf_counter()
        produced = work()
        seconds.append(time.perf_counter() - began)
    return statistics.median(seconds), produced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--gaps", type=int, default=16, help="how many keyframe gaps to decode in parallel")
    args = parser.parse_args()

    keys = list_keyframes(args.video)
    spans = list(zip(keys, keys[1:]))
    step = max(1, len(spans) // args.gaps)
    chosen = spans[::step][: args.gaps]

    def gaps_parallel(workers: int, threads: int) -> int:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return sum(pool.map(lambda span: decode_span(args.video, *span, threads), chosen))

    methods = {
        "list keyframes (no decode)": lambda: len(list_keyframes(args.video)),
        "keyframes only, slice threads": lambda: decode_keyframes(args.video, "SLICE"),
        "keyframes only, auto threads": lambda: decode_keyframes(args.video, "AUTO"),
        "one keyframe gap": lambda: decode_span(args.video, *chosen[0]),
        f"{len(chosen)} gaps, all cores each": lambda: gaps_parallel(len(chosen), 0),
        f"{len(chosen)} gaps, 1 core each": lambda: gaps_parallel(len(chosen), 1),
        f"{len(chosen)} gaps, 2 cores each, 8 at a time": lambda: gaps_parallel(8, 2),
        "every frame": lambda: decode_all(args.video),
    }
    print(f"{args.video.name}: {len(keys)} keyframes; median of {args.runs} runs")
    for name, work in methods.items():
        runs = 1 if name == "every frame" else args.runs
        seconds, produced = timed(runs, work)
        print(f"  {name:<34} {seconds:7.3f} s  {produced:6d} frames  {produced / seconds:8.0f} fps")


if __name__ == "__main__":
    main()
