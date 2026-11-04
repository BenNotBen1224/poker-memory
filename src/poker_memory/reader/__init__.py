"""Read one televised poker hand from its TV graphics, on a CPU, in well under a second.

1. Read the whole file, then decode each keyframe (one every few seconds) on the workers.
2. Once every keyframe is read: where two neighbouring keyframes differ, the first one's decoder goes on through the
   stretch between them, sampling about four pictures a second, and stops as soon as the graphics match the second.
   The last keyframe's decoder goes on to the end of the recording. Workers only decode; one thread reads the
   graphics on every picture.
3. Number every distinct crop of one kind (status line, name, card rank), then OCR that kind's settled ones once,
   on a worker, while this thread numbers the next kind.
4. Hand the timeline to `hand.build`.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import av

from ..model import HandRecord, Stakes
from .hand import BLANK, Moment, Seen, build
from .overlay import TEXT_CUT, Card, Catalog, Look, Ocr, look_all, parse_blinds, parse_rank, parse_status, same_look
from .video import Clip, keyframe, stretch

EVERY = 6  # every 6th picture: about four a second at 24 fps
BATCH = 64  # most pictures read in one go
LEASH = 3  # most pictures a stretch decodes before the graphics on them are read
LATE = 6  # how many of the last stretches to start get a decoder with threads
TAIL_THREADS = 4  # threads for each of those: a 1280-wide picture has 4 tile columns


@dataclass
class Reading:
    record: HandRecord
    stakes: Stakes
    issues: list[str]
    wins: dict[int, int]  # seat -> WIN amount shown on screen
    moments: list[Moment]
    seconds: dict[str, float]
    pictures: int  # pictures read: keyframes plus samples between them
    ocr_reads: int


_OCR: Ocr | None = None


def _ocr() -> Ocr:
    global _OCR
    if _OCR is None:
        _OCR = Ocr()
    return _OCR


def _scan(pool: ThreadPoolExecutor, path: Path) -> tuple[list[Look], list[Look]]:
    """The looks of every keyframe, and of the pictures sampled between neighbouring keyframes that differ and after
    the last keyframe.

    This thread reads the whole file before any decoding starts: reading it while workers decode runs slower, as
    every packet waits for the interpreter lock. Then only this thread reads pictures. numpy lets go of the lock
    inside every call and has to win it back after, so reading on many threads at once runs slower than reading on
    one while the others decode; and reading whatever pictures are waiting together takes fewer calls than reading
    them one by one.

    Stretches start once every keyframe is read: starting each as soon as its two keyframes are in slowed the
    keyframes still decoding. A stretch decodes at most LEASH pictures before this thread reads them, so it stops
    close to the first picture that matches; without the leash a sixth of the decoding was spent past it.

    The workers take stretches in order, so the last ones to start end the scan while other cores sit idle. Those
    decode their keyframe again with a decoder that splits each picture over TAIL_THREADS threads: giving every
    stretch threads made the scan slower, as threads cost more work per picture while every core is busy."""
    clip = Clip.read(path)
    # (what, keyframe number, payload): ("key", k, picture), ("gap", k, picture or None once done) and
    # ("error", 0, exception)
    inbox: queue.SimpleQueue[tuple[str, int, Any]] = queue.SimpleQueue()
    decoders: dict[int, av.CodecContext] = {}  # each keyframe's, until it is known whether the stretch after is needed
    stop: dict[int, bool] = {}
    leash: dict[int, threading.Semaphore] = {}

    def job(work: Callable[[], None]) -> None:
        try:
            work()
        except BaseException as error:  # else this thread would wait for its pictures forever
            inbox.put(("error", 0, error))

    def decode_key(k: int) -> None:
        decoders[k], picture = keyframe(clip, k)
        inbox.put(("key", k, picture))

    def decode_gap(k: int, late: bool) -> None:
        try:
            codec = decoders.pop(k)
            if late:
                codec = keyframe(clip, k, TAIL_THREADS)[0]
            pictures = stretch(clip, k, codec, EVERY)
            while leash[k].acquire() and not stop[k]:
                picture = next(pictures, None)
                if picture is None:
                    break
                inbox.put(("gap", k, picture))
        finally:
            inbox.put(("gap", k, None))

    keys: dict[int, Look] = {}
    between: dict[int, list[Look]] = {}
    running = 0  # stretches still decoding

    def start_gaps() -> None:
        """Decode the stretch after each keyframe unless the next one looks the same, and the pictures after the
        last keyframe, which have no next one to match. An empty screen never ends a gap early: before the flop,
        each player's panel shows only briefly, with empty screens between."""
        nonlocal running
        gaps = []
        for k in range(len(keys) - 1):
            a, b = keys[k], keys[k + 1]
            if a.panels and b.panels and same_look(a, b):
                del decoders[k]
            else:
                gaps.append(k)
        if clip.keys[-1] + 1 < len(clip.packets):
            gaps.append(len(keys) - 1)
        for n, k in enumerate(gaps):
            stop[k], between[k], leash[k] = False, [], threading.Semaphore(LEASH)
            running += 1
            pool.submit(job, partial(decode_gap, k, len(gaps) - n <= LATE))

    for k in range(len(clip.keys)):
        pool.submit(job, partial(decode_key, k))
    try:
        while len(keys) < len(clip.keys) or running:
            batch = [inbox.get()]
            while len(batch) < BATCH and not inbox.empty():
                batch.append(inbox.get_nowait())
            todo = []
            for what, k, payload in batch:
                if what == "error":
                    raise payload
                if payload is None:
                    running -= 1
                elif what == "key" or not stop[k]:
                    todo.append((what, k, payload))
            for (what, k, _), seen in zip(todo, look_all([picture for _, _, picture in todo])):
                if what == "key":
                    keys[k] = seen
                    if len(keys) == len(clip.keys):
                        start_gaps()
                elif not stop[k]:
                    between[k].append(seen)
                    stop[k] = k + 1 < len(keys) and bool(keys[k + 1].panels) and same_look(seen, keys[k + 1])
            for what, k, payload in batch:
                if what == "gap" and payload is not None:
                    leash[k].release()
    except BaseException:  # else the pool would wait forever for workers held on their leash
        for k in leash:
            stop[k] = True
            leash[k].release()
        raise
    return [keys[k] for k in range(len(keys))], [seen for k in sorted(between) for seen in between[k]]


def _stakes(keys: list[Look], ocr: Ocr) -> Stakes:
    preflop = sorted((k for k in keys if all(c is None for c in k.board)),
                     key=lambda k: -int((k.blinds > TEXT_CUT).sum()))[:3]
    texts = ocr([crop for k in preflop for crop in (k.blinds, k.ante)])
    for blinds, ante in zip(texts[::2], texts[1::2]):
        if parsed := parse_blinds(blinds, ante):
            sb, bb, ante_ = parsed
            return Stakes(sb=sb, bb=bb, ante=ante_)
    raise ValueError(f"could not read the blinds: {texts}")


Cell = tuple[tuple[int, ...], Any, Any]  # where on screen, mask, crop

# Each kind of text in a look, one cell per place it can be: None where nothing is shown. Status lines come first:
# they take OCR longest, so their reading overlaps numbering the rest.
_CELLS: dict[str, Callable[[Look], list[Cell | None]]] = {
    "status": lambda k: [None if p.status is None else ((p.slot, p.slid), p.status, p.status_crop) for p in k.panels],
    "name": lambda k: [((p.slot, p.slid), p.name, p.name_crop) for p in k.panels],
    "hole": lambda k: [None if c is None else ((p.slot, p.slid, j), c.glyph, c.rank)
                       for p in k.panels for j, c in enumerate(p.cards)],
    "board": lambda k: [None if c is None else ((j,), c.glyph, c.rank) for j, c in enumerate(k.board)],
}


def _number(looks: list[Look], cells: Callable[[Look], list[Cell | None]]) -> tuple[list[list[int | None]], Catalog]:
    """Every look's cells of one kind as numbers in one Catalog. Each crop is tried first against the number the
    crop in the same place got last time, which is nearly always right."""
    catalog, last, ids = Catalog(), {}, []
    for k in looks:
        row = []
        for cell in cells(k):
            if cell is None:
                row.append(None)
                continue
            where, mask, crop = cell
            last[where] = catalog.add(mask, crop, last.get(where))
            row.append(last[where])
        ids.append(row)
    return ids, catalog


def _state(i: int, text: str) -> str:
    parsed = parse_status(text)
    if not parsed:
        return f"#{i}"
    return parsed[0] if parsed[1] is None else f"{parsed[0]} {parsed[1]}"


def read_hand(path: Path) -> Reading:
    path = Path(path)
    clock = [time.perf_counter()]
    seconds: dict[str, float] = {}

    def lap(name: str) -> None:
        clock.append(time.perf_counter())
        seconds[name] = clock[-1] - clock[-2]

    with ThreadPoolExecutor(os.cpu_count() or 1) as pool:
        loading = pool.submit(_ocr)  # the first read in a process loads the OCR model while it scans
        keys, between = _scan(pool, path)
        looks = sorted(keys + between, key=lambda k: k.time)
        lap("scan")
        ocr = loading.result()
        calls = ocr.calls
        lap("ocr_wait")
        ids, reading = {}, {}
        for kind, cells in _CELLS.items():
            ids[kind], catalog = _number(looks, cells)
            settled = catalog.settled()
            crops = [catalog.text(i) if kind in ("name", "status") else catalog.crops[i] for i in settled]
            reading[kind] = settled, pool.submit(ocr, crops)
        stakes_read = pool.submit(_stakes, keys, ocr)
        lap("match")
        texts = {kind: dict(zip(settled, job.result())) for kind, (settled, job) in reading.items()}
        stakes = stakes_read.result()
    lap("ocr")

    who = {i: " ".join(re.sub(r"[^A-Z .'-]", "", text.upper()).split()) for i, text in texts["name"].items()}
    states = {i: _state(i, text) for i, text in texts["status"].items()}

    def card(kind: str, i: int | None, shown: Card) -> str | None:
        if i is None or i not in texts[kind]:
            return None
        rank = parse_rank(texts[kind][i])
        return rank + shown.suit if rank and shown.suit in "shdc" else None

    moments = []
    for k, names, statuses, holes, board in zip(looks, ids["name"], ids["status"], ids["hole"], ids["board"]):
        hole = iter(holes)  # two per panel
        seen = []
        for p, name, status in zip(k.panels, names, statuses):
            cards = tuple(card("hole", next(hole), c) for c in p.cards)
            if any(ch.isalpha() for ch in who.get(name, "")):
                seen.append(Seen(who[name], BLANK if status is None else states.get(status), cards))
        moments.append(Moment(k.time, tuple(seen),
                              tuple(None if c is None else card("board", i, c) or "" for i, c in zip(board, k.board)),
                              k.strip))
    draft = build(moments, stakes)
    lap("build")
    seconds["total"] = clock[-1] - clock[0]
    return Reading(draft.record, stakes, draft.issues, draft.wins, moments, seconds, len(looks), ocr.calls - calls)
