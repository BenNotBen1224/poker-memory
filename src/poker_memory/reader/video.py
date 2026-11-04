"""Get pictures out of a recording without decoding all of it.

The file is read once, into its compressed packets. A keyframe decodes on its own, so decoding just those gives a
picture every few seconds, each on any free core. The stretch after a keyframe can only be decoded from that
keyframe, so `stretch` goes on with the decoder that decoded it, on one core, so many stretches can run side by
side; the caller stops it as soon as it has seen enough.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
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
    """A recording's compressed packets, in decode order; each one decodes to one shown picture."""

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


def keyframe(clip: Clip, k: int, threads: int = 1) -> tuple[av.CodecContext, Picture]:
    """Keyframe k's picture, and the decoder that decoded it, ready to go on into the stretch after it."""
    codec = _decoder(clip, threads)
    i = clip.keys[k]
    frames = codec.decode(clip.packets[i])
    if len(frames) != 1:
        raise ValueError(f"keyframe {k} decoded to {len(frames)} pictures")
    return codec, _picture(frames[0], clip.times[i])


def stretch(clip: Clip, k: int, codec: av.CodecContext, every: int) -> Iterator[Picture]:
    """Every `every`-th picture after keyframe k, up to the next keyframe, going on with the decoder that decoded k.
    After the last keyframe it goes on to the end of the recording, and gives the last picture too."""
    start = clip.keys[k]
    end = clip.keys[k + 1] if k + 1 < len(clip.keys) else len(clip.packets)
    for i in range(start + 1, end):
        for frame in codec.decode(clip.packets[i]):
            if (i - start) % every == 0 or i == len(clip.packets) - 1:
                yield _picture(frame, clip.times[i])
