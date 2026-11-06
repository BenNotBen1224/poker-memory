"""Render the TikTok: video + overlays + audio mix.

  python render.py                      full video  -> out/steinman-fold-tiktok.mp4 (+ frames.json, mix-stats.json)
  python render.py --preview hook+1 ... stills     -> out/preview.jpg   (time = seconds or segment+offset)
  python render.py --range 20 30        a section  -> out/range.mp4
  python render.py --audio              mix only   -> out/mix.wav
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from fractions import Fraction

import av
import numpy as np
from av.audio.frame import AudioFrame
from av.video.frame import VideoFrame
from PIL import Image, ImageDraw
from scipy.io import wavfile

import gfx
import mix
import scenes
import timeline as tl
from timeline import FPS, SR

OUT = tl.HERE / "out"
FINAL = OUT / "steinman-fold-tiktok.mp4"
X264 = {"crf": "18", "preset": "medium", "profile": "high",
        "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709"}


def parse_time(spec: str) -> float:
    try:
        return float(spec)
    except ValueError:
        pass
    m = re.fullmatch(r"([a-z_0-9]+)(?:([+-])([\d.]+))?", spec)
    if not m or m.group(1) not in tl.BY_NAME:
        raise SystemExit(f"bad time spec: {spec}")
    off = float(m.group(3) or 0) * (-1 if m.group(2) == "-" else 1)
    return tl.T(m.group(1), off)


def frame_of(t: float) -> int:
    return min(tl.NFRAMES - 1, max(0, math.floor(t * FPS + 1e-6)))


def compose(k: int, src: tl.Source, sc: scenes.Scene) -> tuple[np.ndarray, dict]:
    canvas, info = tl.render_video(k, src)
    t = tl.frame_time(k)
    scenes.draw(sc, canvas, t)
    tl.flash(canvas, t)
    return np.clip(canvas + 0.5, 0, 255).astype(np.uint8), info


def audio_packets(stream, audio: np.ndarray) -> list:
    audio = np.concatenate([audio, np.zeros(((-len(audio)) % 1024, 2), np.float32)])
    pkts = []
    for i in range(0, len(audio), 1024):
        fr = AudioFrame.from_ndarray(np.ascontiguousarray(audio[i:i + 1024].T), format="fltp", layout="stereo")
        fr.sample_rate, fr.pts, fr.time_base = SR, i, Fraction(1, SR)
        pkts.extend(stream.encode(fr))
    pkts.extend(stream.encode(None))
    return pkts


def encode(path, k0: int, k1: int, audio: np.ndarray) -> list[dict]:
    src, sc = tl.Source(), scenes.build()
    c = av.open(str(path), "w", options={"movflags": "+faststart"})
    vs = c.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = tl.W, tl.H, "yuv420p"
    vs.options = dict(X264)
    for attr in ("color_primaries", "color_trc", "colorspace", "color_range"):
        setattr(vs.codec_context, attr, 1)
    ast = c.add_stream("aac", rate=SR, layout="stereo")
    ast.bit_rate = 192000
    apk = audio_packets(ast, audio)
    ai, infos, t0 = 0, [], time.time()
    for k in range(k0, k1):
        rgb, info = compose(k, src, sc)
        fr = VideoFrame.from_ndarray(rgb, "rgb24").reformat(
            format="yuv420p", dst_colorspace="ITU709", src_color_range="JPEG", dst_color_range="MPEG")
        fr.pts, fr.time_base = k - k0, 1 / FPS
        for p in vs.encode(fr):
            c.mux(p)
        limit = float((k - k0 + 1) / FPS) + 0.3
        while ai < len(apk) and float(apk[ai].pts * apk[ai].time_base) <= limit:
            c.mux(apk[ai])
            ai += 1
        infos.append(info)
        done = k - k0 + 1
        if done % 120 == 0 or k == k1 - 1:
            el = time.time() - t0
            print(f"  frame {done}/{k1 - k0}  {done / el:.1f} fps  eta {(k1 - k0 - done) * el / done:.0f}s", flush=True)
    for p in vs.encode(None):
        c.mux(p)
    for p in apk[ai:]:
        c.mux(p)
    c.close()
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s, seeks {src.seeks})")
    return infos


def build_audio() -> tuple[np.ndarray, dict]:
    t0 = time.time()
    sc = scenes.build()
    out, stats = mix.mix(sc.cues)
    stats["seconds"] = round(time.time() - t0, 1)
    print("mix:", {k: v for k, v in stats.items() if k != "clips"})
    return out[:stats["end_sample"]], stats


def preview(specs: list[str], scale: float, cols: int, name: str = "preview") -> None:
    src, sc = tl.Source(), scenes.build()
    tiles = []
    for spec in specs:
        k = frame_of(parse_time(spec))
        rgb, info = compose(k, src, sc)
        im = Image.fromarray(rgb).resize((round(tl.W * scale), round(tl.H * scale)), Image.LANCZOS)
        tiles.append((f"{spec}  {tl.frame_time(k):.2f}s  {info['seg']}", im))
    tw, th, lab = tiles[0][1].width, tiles[0][1].height, 34
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + lab + 8) + 8), (40, 40, 40))
    d, font = ImageDraw.Draw(sheet), gfx.tt(22, "Bold")
    for i, (label, im) in enumerate(tiles):
        x, y = 8 + (i % cols) * (tw + 8), 8 + (i // cols) * (th + lab + 8)
        d.text((x + 4, y + 6), label, font=font, fill=(255, 255, 255))
        sheet.paste(im, (x, y + lab))
    path = OUT / f"{name}.jpg"
    sheet.save(path, quality=90)
    print(f"wrote {path} ({len(tiles)} stills)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs="+")
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--name", default="preview")
    ap.add_argument("--range", nargs=2)
    ap.add_argument("--audio", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.preview:
        preview(a.preview, a.scale, a.cols, a.name)
        return 0
    audio, stats = build_audio()
    if a.audio:
        wavfile.write(OUT / "mix.wav", SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
        (OUT / "mix-stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        print(f"wrote {OUT / 'mix.wav'}")
        return 0
    if a.range:
        k0, k1 = frame_of(parse_time(a.range[0])), frame_of(parse_time(a.range[1])) + 1
        s0, s1 = round(k0 / FPS * SR), round(k1 / FPS * SR)
        encode(OUT / "range.mp4", k0, k1, audio[s0:s1])
        return 0
    infos = encode(FINAL, 0, tl.NFRAMES, audio)
    (OUT / "frames.json").write_text(json.dumps(infos), encoding="utf-8")
    (OUT / "mix-stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
