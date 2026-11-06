"""Check the rendered TikTok file itself. Re-runnable; writes out/verify.json, out/verify.txt and out/sheet-*.jpg.

  python verify.py
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import av
import numpy as np
import pyloudnorm
from PIL import Image, ImageDraw
from scipy.signal import correlate, resample_poly

import align
import captions
import gfx
import mix
import overlays
import scenes
import timeline as tl
from timeline import FPS, SR

OUT = tl.HERE / "out"
FINAL = OUT / "steinman-fold-tiktok.mp4"
REVEAL_K = tl.BY_NAME["reveal"].f0
T_REVEAL = tl.T("reveal")
SRC_FRAME = float(1 / FPS)
BANNED = ["queen", "ten", "q 10", "q10", "straight", "broadway", "nuts", "spicy float", "gut shot", "gutshot",
          "oh no", "devastating", "trap", "poor ian", "worst card", "all the chips", "victory lap", "dream card",
          "death card", "knows hes beat", "joe is sick", "camouflaged", "camo"]
NAMES = ["vince", "van patten", "patten", "tony", "dunst", "lynn", "gilmartin"]


def tokens(text: str) -> list[str]:
    return [t.replace("'", "") for t in re.findall(r"[a-z0-9']+", text.lower()) if t.replace("'", "")]


def phrase_hits(words: list[dict], phrases: list[str]) -> list[dict]:
    toks = [(n, w["t"]) for w in words for n in tokens(w["w"])]
    hits = []
    for ph in phrases:
        p = ph.split()
        for i in range(len(toks) - len(p) + 1):
            if [x for x, _ in toks[i:i + len(p)]] == p:
                ctx = " ".join(x for x, _ in toks[max(0, i - 5):i + len(p) + 5])
                hits.append({"phrase": ph, "t": round(toks[i][1], 2), "context": ctx})
    return hits


def top_boxes(path) -> list[str]:
    names, size = [], path.stat().st_size
    with open(path, "rb") as f:
        pos = 0
        while pos < size:
            f.seek(pos)
            head = f.read(16)
            n, kind = int.from_bytes(head[:4], "big"), head[4:8].decode("latin-1")
            n = int.from_bytes(head[8:16], "big") if n == 1 else (size - pos if n == 0 else n)
            names.append(kind)
            pos += n
    return names


def decode(sample_ks: set[int]) -> dict:
    c = av.open(str(FINAL))
    vs, ast = c.streams.video[0], c.streams.audio[0]
    cc = vs.codec_context
    fmt = {"container_s": round(c.duration / 1e6, 3), "codec": cc.name, "profile": cc.profile,
           "size": [cc.width, cc.height], "pix_fmt": cc.pix_fmt, "avg_rate": str(vs.average_rate),
           "color": [cc.color_primaries, cc.color_trc, cc.colorspace, cc.color_range],
           "audio": [ast.codec_context.name, ast.codec_context.sample_rate, ast.codec_context.layout.nb_channels],
           "bytes": FINAL.stat().st_size, "boxes": top_boxes(FINAL)}
    ks, stills = [], {}
    for fr in c.decode(vs):
        k = round(fr.time * float(FPS))
        ks.append(k)
        if k in sample_ks:
            im = fr.to_image()
            stills[k] = im.resize((im.width // 5, im.height // 5), Image.LANCZOS)
    c.close()
    c = av.open(str(FINAL))
    chunks, a_first = [], None
    for fr in c.decode(audio=0):
        a_first = fr.time if a_first is None else a_first
        chunks.append(fr.to_ndarray())
    c.close()
    audio = np.concatenate(chunks, axis=1).T.astype(np.float32)
    fmt.update(video_frames=len(ks), first_video_time=None, audio_first_time=a_first,
               audio_samples=len(audio), audio_s=round(len(audio) / SR, 3))
    return {"fmt": fmt, "ks": ks, "stills": stills, "audio": audio}


def check_log(infos: list[dict]) -> dict:
    hi = SRC_FRAME + 0.0045 + 1e-4
    return {"frames": len(infos),
            "contiguous": [f["i"] for f in infos] == list(range(tl.NFRAMES)),
            "unmasked_before_reveal": [f["i"] for f in infos if f["i"] < REVEAL_K and f["mask"] == "none"],
            "masks_before_reveal": sorted({f["mask"] for f in infos if f["i"] < REVEAL_K}),
            "segment_mismatch": [f["i"] for f in infos if f["seg"] != tl.seg_of(f["i"]).name],
            "wrong_source_frame": [f["i"] for f in infos if not -0.0046 <= f["src_t"] - f["pts_time"] <= hi]}


def clip_spans() -> dict:
    return {name: (at, at + sum(b - a for a, b in parts) + tl.GAP * (len(parts) - 1))
            for name, parts, at, _ in tl.VOICE}


def source_scan(spans: dict) -> dict:
    pools = {name: captions.model_words(f) for name, f in captions.SOURCES if (captions.PLAN / f).exists()}
    hits, heard = [], {}
    for name, parts, at, _ in tl.VOICE:
        if spans[name][0] >= T_REVEAL:
            continue
        for model, ws in pools.items():
            sel = [w for w in ws if any(w["end"] > a - 0.15 and w["start"] < b + 0.15 for a, b in parts)]
            words = [{"w": w["w"], "t": at + (tl.clip_offset(parts, w["start"]) or 0.0)} for w in sel]
            heard[f"{name}/{model}"] = " ".join(w["w"] for w in sel)
            hits += [dict(h, clip=name, model=model) for h in phrase_hits(words, BANNED + NAMES)]
    return {"models": list(pools), "hits": hits, "heard": heard}


def whisper(a16: np.ndarray) -> list[dict]:
    from faster_whisper import WhisperModel
    m = WhisperModel("medium.en", device="cpu", compute_type="int8", local_files_only=True)
    segs, _ = m.transcribe(a16, language="en", word_timestamps=True, beam_size=5,
                           condition_on_previous_text=False, vad_filter=False)
    return [{"w": w.word.strip(), "t": round(w.start, 2), "end": round(w.end, 2), "p": round(w.probability, 2)}
            for s in segs for w in s.words]


def loudness(x: np.ndarray) -> dict:
    tp = max(np.abs(resample_poly(x[:, ch].astype(np.float64), 4, 1)).max() for ch in range(x.shape[1]))
    return {"lufs": round(pyloudnorm.Meter(SR).integrated_loudness(x.astype(np.float64)), 2),
            "true_peak_dbtp": round(20 * np.log10(tp), 2),
            "sample_peak_dbfs": round(20 * np.log10(np.abs(x).max()), 2)}


def av_offset(dec: np.ndarray, ref: np.ndarray, max_lag: int = 4800) -> dict:
    n = min(len(dec), len(ref))
    a, b = dec[:n].mean(axis=1).astype(np.float64), ref[:n].mean(axis=1).astype(np.float64)
    z = correlate(a, b, mode="full", method="fft")[n - 1 - max_lag:n + max_lag]
    lag = int(np.argmax(z)) - max_lag
    a2 = a[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag), a])
    m = min(len(a2), len(b))
    res = a2[:m] - b[:m]
    return {"lag_samples": lag, "lag_ms": round(lag / SR * 1000, 2),
            "corr": round(float(np.corrcoef(a2[:m], b[:m])[0, 1]), 5),
            "residual_db": round(float(20 * np.log10(np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(b[:m] ** 2)))), 1),
            "length_diff_samples": len(dec) - len(ref)}


def caption_checks(a16: np.ndarray, heard: list[dict], spans: dict) -> list[dict]:
    words = overlays.voice_words()
    rows = []
    for name, (t0, t1) in spans.items():
        ws = [w for w in words if w["clip"] == name]
        w0 = max(0.0, t0 - 0.25)
        seg = a16[int(w0 * 16000):int((t1 + 0.25) * 16000)]
        raw = align.align(seg, [w["text"] for w in ws])
        keep = [i for i, r in enumerate(raw) if r["start"] is not None]
        res = align.refine_starts(seg, [raw[i] for i in keep])
        d_ref = [w0 + r["start"] - ws[i]["start"] for i, r in zip(keep, res)]
        d_ctc = [w0 + r["ctc_start"] - ws[i]["start"] for i, r in zip(keep, res)]
        cap = [t for w in ws for t in tokens(w["text"])]
        got = [t for w in heard if t0 - 0.3 <= w["t"] < t1 + 0.3 for t in tokens(w["w"])]
        rows.append({"clip": name, "start": round(t0, 2), "words": len(ws), "aligned": len(d_ref),
                     "median_offset_s": round(float(np.median(d_ref)), 3) if d_ref else None,
                     "median_offset_ctc_s": round(float(np.median(d_ctc)), 3) if d_ctc else None,
                     "max_abs_offset_s": round(float(np.max(np.abs(d_ref))), 3) if d_ref else None,
                     "text_match": round(difflib.SequenceMatcher(a=cap, b=got, autojunk=False).ratio(), 2),
                     "caption": " ".join(w["text"] for w in ws),
                     "heard": " ".join(w["w"] for w in heard if t0 - 0.3 <= w["t"] < t1 + 0.3)})
    return rows


def sheets(stills: dict, ks: list[int]) -> list[str]:
    paths, per, cols = [], 24, 8
    font = gfx.tt(18, "Bold")
    for n in range(0, len(ks), per):
        group = [k for k in ks[n:n + per] if k in stills]
        tw, th = stills[group[0]].size
        rows = -(-len(group) // cols)
        sheet = Image.new("RGB", (cols * (tw + 6) + 6, rows * (th + 30) + 6), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for i, k in enumerate(group):
            x, y = 6 + (i % cols) * (tw + 6), 6 + (i // cols) * (th + 30)
            label = f"{tl.frame_time(k):5.1f}s {tl.seg_of(k).name}"
            d.text((x + 2, y + 4), label, font=font, fill=(255, 255, 255) if k < REVEAL_K else (120, 255, 120))
            sheet.paste(stills[k], (x, y + 30))
        p = OUT / f"sheet-{n // per + 1:02d}.jpg"
        sheet.save(p, quality=88)
        paths.append(p.name)
    return paths


def main() -> int:
    t_start = time.time()
    sample_t = list(np.arange(0.25, T_REVEAL, 0.5)) + list(np.arange(T_REVEAL + 0.3, tl.DURATION, 1.0))
    sample_ks = sorted({min(tl.NFRAMES - 1, int(t * float(FPS))) for t in sample_t})
    dec = decode(set(sample_ks))
    fmt = dec["fmt"]
    fmt["first_video_time"] = round(dec["ks"][0] / float(FPS), 4) if dec["ks"] else None
    infos = json.loads((OUT / "frames.json").read_text(encoding="utf-8"))
    log = check_log(infos)
    spans = clip_spans()
    crossing = [n for n, (a, b) in spans.items() if a < T_REVEAL < b - 0.01]
    src = source_scan(spans)
    ref, mix_stats = mix.mix(scenes.build().cues)
    av_res = av_offset(dec["audio"], ref[:mix_stats["end_sample"]])
    loud = loudness(dec["audio"])
    a16 = resample_poly(dec["audio"].mean(axis=1).astype(np.float64), 1, 3).astype(np.float32)
    heard = whisper(a16)
    pre = [w for w in heard if w["t"] < T_REVEAL]
    mix_hits, name_hits = phrase_hits(pre, BANNED), phrase_hits(heard, NAMES)
    caps = caption_checks(a16, heard, spans)
    sheet_paths = sheets(dec["stills"], sample_ks)

    checks = [
        ("format", fmt["codec"] == "h264" and fmt["size"] == [tl.W, tl.H] and fmt["pix_fmt"] == "yuv420p"
         and fmt["avg_rate"] == "24000/1001" and fmt["color"] == [1, 1, 1, 1] and fmt["audio"] == ["aac", SR, 2]
         and {"moov", "mdat"} <= set(fmt["boxes"]) and fmt["boxes"].index("moov") < fmt["boxes"].index("mdat"),
         f"{fmt['size']} {fmt['avg_rate']} "
         f"{fmt['codec']}/{fmt['profile']} {fmt['audio']} color={fmt['color']} boxes={fmt['boxes']}"),
        ("all frames decode", dec["ks"] == list(range(tl.NFRAMES)),
         f"{fmt['video_frames']} frames, first at {fmt['first_video_time']}s, container {fmt['container_s']}s"),
        ("frame log", log["contiguous"] and not log["segment_mismatch"] and not log["wrong_source_frame"],
         f"{log['frames']} rows, wrong source frame {len(log['wrong_source_frame'])}"),
        ("hole cards masked before reveal", not log["unmasked_before_reveal"],
         f"{REVEAL_K} frames before {T_REVEAL:.2f}s all masked ({log['masks_before_reveal']})"),
        ("no voice clip crosses the reveal", not crossing, str(crossing or "none")),
        ("no spoiler words in source clips", not src["hits"], f"{len(src['hits'])} hits over {src['models']}"),
        ("no spoiler words heard before reveal", not mix_hits, f"{len(mix_hits)} hits in {len(pre)} heard words"),
        ("no commentator names heard", not name_hits, f"{len(name_hits)} hits in {len(heard)} heard words"),
        ("loudness", -15 <= loud["lufs"] <= -13 and loud["true_peak_dbtp"] <= -1.0, str(loud)),
        ("audio in sync with the timeline", abs(av_res["lag_samples"]) <= 48 and av_res["corr"] >= 0.98, str(av_res)),
        ("audio length", abs(fmt["audio_s"] - tl.DURATION) <= 0.05,
         f"audio {fmt['audio_s']}s vs video {tl.DURATION:.3f}s, first audio at {fmt['audio_first_time']}"),
        ("captions on time", all(r["median_offset_s"] is not None and abs(r["median_offset_s"]) <= 0.2 for r in caps),
         "worst median offset " + str(max((9 if r["median_offset_s"] is None else abs(r["median_offset_s"]), r["clip"])
                                          for r in caps))),
    ]
    verdict = all(ok for _, ok, _ in checks)
    report = {"file": FINAL.relative_to(tl.HERE).as_posix(), "verdict": "PASS" if verdict else "FAIL", "seconds": round(time.time() - t_start),
              "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks], "format": fmt, "log": log,
              "source_scan": src, "mix_hits": mix_hits, "name_hits": name_hits, "loudness": loud, "av": av_res,
              "captions": caps, "sheets": sheet_paths, "heard": heard}
    (OUT / "verify.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = [f"{'PASS' if ok else 'FAIL'}  {n}: {d}" for n, ok, d in checks]
    lines += ["", "captions (clip, start, median offset s, ctc-only, max |offset|, text match):"]
    lines += [f"  {r['clip']:9s} {r['start']:6.2f}  {r['median_offset_s']!s:>7s} {r['median_offset_ctc_s']!s:>7s} "
              f"{r['max_abs_offset_s']!s:>6s}  {r['text_match']:.2f}  heard: {r['heard']}" for r in caps]
    lines += ["", "spoiler hits:"] + [f"  {h}" for h in src["hits"] + mix_hits + name_hits] + ["", f"sheets: {sheet_paths}",
              f"verdict: {report['verdict']}"]
    (OUT / "verify.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
