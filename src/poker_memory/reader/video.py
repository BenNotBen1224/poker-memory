"""Get pictures out of a recording without decoding all of it.

The file is read once, into its compressed packets. Each keyframe starts an independent decoder, which may need
later packets before returning reordered pictures. Its iterator retains that decoder for the stretch after the
keyframe; stretches can run side by side and stop once the caller has seen enough.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class Picture:
    time: float
    y: np.ndarray  # brightness, full size
    u: np.ndarray  # blue-difference colour, half size
    v: np.ndarray  # red-difference colour, half size


@dataclass(frozen=True)
class Clip:
    """A recording's compressed packets in decode order, with presentation times and keyframe packet indexes."""

    codec: str
    extradata: bytes | None
    packets: list[av.Packet]
    times: list[float]
    keys: list[int]  # where the keyframes are in `packets`

    @classmethod
    def read(cls, path: Path) -> Clip:
        with av.open(str(path)) as box:
            stream = box.streams.video[0]
            clip = cls(stream.codec_context.name, stream.codec_context.extradata, [], [], [])
            # the same floats as float(p.pts * time_base), without slow Fraction maths
            num, den = stream.time_base.numerator, stream.time_base.denominator
            for p in box.demux(stream):
                if p.pts is None or not p.size:
                    continue
                clip.packets.append(p)
                clip.times.append(p.pts * num / den)
                if p.is_keyframe:
                    clip.keys.append(len(clip.packets) - 1)
        return clip


def _decoder(clip: Clip, threads: int) -> av.CodecContext:
    codec = av.CodecContext.create(clip.codec, "r")
    if clip.extradata:
        codec.extradata = clip.extradata
    # Threads split each picture into its tiles. Frame threads would hand each picture back a few packets late.
    codec.thread_type = "NONE" if threads == 1 else "SLICE"
    codec.thread_count = threads
    return codec


def _picture(frame: av.VideoFrame, time: float) -> Picture:
    if frame.format.name != "yuv420p":
        frame = frame.reformat(format="yuv420p")
    planes = []
    for i, plane in enumerate(frame.planes):
        h, w = (frame.height, frame.width) if i == 0 else (frame.height // 2, frame.width // 2)
        planes.append(np.frombuffer(plane, np.uint8).reshape(h, plane.line_size)[:, :w])
    return Picture(time, *planes)


def _time(frame: av.VideoFrame) -> float:
    if frame.pts is None or frame.time_base is None:
        raise ValueError("decoded video frame has no presentation timestamp")
    return frame.pts * frame.time_base.numerator / frame.time_base.denominator


def _frames(clip: Clip, k: int, threads: int) -> Iterator[av.VideoFrame]:
    codec = _decoder(clip, threads)
    start = clip.keys[k]
    end_time = clip.times[clip.keys[k + 1]] if k + 1 < len(clip.keys) else float("inf")
    # Look ahead past a GOP's packet boundary to drain reordered pictures; None flushes the final GOP at EOF.
    for packet in chain(islice(clip.packets, start, None), (None,)):
        for frame in codec.decode(packet):
            if packet is None:
                # PyAV's bare-codec EOF flush preserves PTS but does not attach the packet stream's time base.
                frame.time_base = clip.packets[start].time_base
            time = _time(frame)
            if time >= end_time:
                return
            if time >= clip.times[start]:
                yield frame


def keyframe(clip: Clip, k: int, threads: int = 1) -> tuple[Iterator[av.VideoFrame], Picture]:
    """Keyframe k's picture and its retained presentation-order decoder iterator."""
    frames = _frames(clip, k, threads)
    frame = next(frames, None)
    if frame is None or _time(frame) != clip.times[clip.keys[k]]:
        raise ValueError(f"keyframe {k} did not decode at its presentation timestamp")
    return frames, _picture(frame, _time(frame))


def stretch(clip: Clip, k: int, frames: Iterator[av.VideoFrame], every: int) -> Iterator[Picture]:
    """Every `every`-th picture after keyframe k, up to the next keyframe, going on with the decoder that decoded k.
    After the last keyframe it goes on to the end of the recording, and gives the last picture too."""
    last = None
    last_index = 0
    for i, frame in enumerate(frames, 1):
        if i % every == 0:
            yield _picture(frame, _time(frame))
        last, last_index = frame, i
    if k == len(clip.keys) - 1 and last is not None and last_index % every:
        yield _picture(last, _time(last))
