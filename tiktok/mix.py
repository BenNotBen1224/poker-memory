"""Audio: commentary clips, synthesized effects, a quiet tension bed, and mastering."""
from __future__ import annotations

import av
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

import sfx
import timeline as tl
from timeline import SR, S, T

CACHE = tl.HERE / "cache" / "src48k.npy"
METER = pyln.Meter(SR)

SOUNDS = {
    "pop": sfx.pop, "chips": sfx.chips, "tick_hi": lambda: sfx.tick(True), "tick_lo": lambda: sfx.tick(False),
    "boom": sfx.boom, "sub": sfx.sub_drop, "heart": sfx.heartbeat, "riser": sfx.riser, "ding": sfx.ding,
    "whoosh": sfx.whoosh, "rewind": sfx.rewind,
}


def source_audio() -> np.ndarray:
    """Whole source soundtrack as (n, 2) float32 at 48 kHz; sample index = seconds * 48000."""
    if CACHE.exists():
        return np.load(CACHE)
    rs = av.AudioResampler(format="fltp", layout="stereo", rate=SR)
    chunks, first = [], None
    with av.open(str(tl.SRC)) as c:
        for fr in c.decode(c.streams.audio[0]):
            if first is None:
                first = fr.pts
            chunks.extend(o.to_ndarray() for o in rs.resample(fr))
        chunks.extend(o.to_ndarray() for o in rs.resample(None))
    assert first == 0, f"audio does not start at 0 (pts {first})"
    a = np.ascontiguousarray(np.concatenate(chunks, axis=1).T.astype(np.float32))
    CACHE.parent.mkdir(exist_ok=True)
    np.save(CACHE, a)
    return a


def _fade(x: np.ndarray, fin: float = 0.025, fout: float = 0.07) -> np.ndarray:
    env = np.ones(len(x), np.float32)
    a, b = int(fin * SR), int(fout * SR)
    env[:a] = np.linspace(0, 1, a)
    env[-b:] = np.minimum(env[-b:], np.linspace(1, 0, b))
    return x * env[:, None]


def build_clip(src: np.ndarray, parts) -> np.ndarray:
    pieces = []
    for i, (a, b) in enumerate(parts):
        if i:
            pieces.append(np.zeros((int(round(tl.GAP * SR)), 2), np.float32))
        pieces.append(_fade(np.array(src[int(round(a * SR)):int(round(b * SR))], np.float32)))
    return np.concatenate(pieces)


def _add(dst: np.ndarray, x: np.ndarray, i0: int, gain: float = 1.0) -> None:
    if i0 < 0:
        x, i0 = x[-i0:], 0
    n = min(len(x), len(dst) - i0)
    if n > 0:
        dst[i0:i0 + n] += x[:n] * gain


def energy_db(x: np.ndarray, win: float) -> np.ndarray:
    p = uniform_filter1d(np.mean(x.astype(np.float64) ** 2, axis=1), max(1, int(win * SR)))
    return 10 * np.log10(np.maximum(p, 0.0) + 1e-12)


def snr_db(x: np.ndarray) -> float:
    p10, p95 = np.percentile(energy_db(x, 0.05), [10, 95])
    return float(p95 - p10)


def expand(x: np.ndarray, max_cut: float = 9.0) -> tuple[np.ndarray, float]:
    """Gentle 1:2 downward expander for quiet table-mic clips: room noise between words drops, speech stays."""
    e = energy_db(x, 0.02)
    p10, p95 = np.percentile(e, [10, 95])
    thr = min(p10 + 8, p95 - 6)
    hop = SR // 1000
    target = -np.clip(thr - e[::hop], 0, max_cut)
    att, rel = np.exp(-1 / 5), np.exp(-1 / 120)
    g, cur = np.empty_like(target), 0.0
    for i, tg in enumerate(target):
        a = att if tg > cur else rel
        cur = a * cur + (1 - a) * tg
        g[i] = cur
    gs = np.interp(np.arange(len(x)), np.arange(len(g)) * hop, g)
    return (x * (10 ** (gs / 20))[:, None]).astype(np.float32), float(thr)


def big_cues() -> list[tuple[float, str, float]]:
    c = [(0.0, "boom", -14), (T("rewind"), "rewind", -8),
         (T("river"), "riser", -10), (T("river", 1.4), "boom", -9),
         (S("bet_a", 197.12), "chips", -6)]
    c += [(T("shove", o), "heart", -8) for o in (0.2, 1.05, 1.9)]
    c += [(T("shove", 1.05), "riser", -10), (T("shove", 2.45), "boom", -7), (T("shove", 2.45), "sub", -10)]
    for name in ("tank_joe", "fold_a"):
        c += [(T(name, 0.1), "heart", -9), (T(name, 0.75), "heart", -9)]
    c += [(T("fold_b", 0.78), "boom", -13), (T("fold_b", 0.78), "sub", -16), (T("regret", 0.1), "whoosh", -14)]
    c += [(T("build"), "riser", -6), (T("build", 0.2), "heart", -8), (T("build", 0.8), "heart", -8)]
    c += [(T("reveal"), "boom", -6), (T("reveal"), "sub", -9), (T("reveal", 1.5), "ding", -12)]
    c += [(T("twist") - 0.3, "whoosh", -10), (T("outro", 1.2), "pop", -16)]
    return c


def bed_gain(n: int) -> np.ndarray:
    keys = [(0, 0.10), (T("rewind"), 0.0), (T("ian"), 0.08), (T("river"), 0.16), (T("wow"), 0.12),
            (T("poll"), 0.30), (T("fold_b", 0.78), 0.10), (T("regret"), 0.24), (T("build"), 0.0),
            (T("reveal"), 0.12), (T("twist"), 0.28), (T("outro"), 0.14)]
    g = np.zeros(n, np.float32)
    for tk, v in keys:
        g[int(tk * SR):] = v
    g = uniform_filter1d(g, size=int(0.15 * SR))
    end, fo = int(tl.DURATION * SR), int(1.5 * SR)
    g[end - fo:end] *= np.linspace(1, 0, fo, dtype=np.float32)
    g[end:] = 0
    return g


def limit(x: np.ndarray, ceiling: float) -> np.ndarray:
    peak = np.max(np.abs(x), axis=1)
    g = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    w = int(0.008 * SR)
    g = uniform_filter1d(minimum_filter1d(g, size=2 * w + 1), size=w)
    return (x * g[:, None]).astype(np.float32)


def true_peak_db(x: np.ndarray) -> float:
    m = max(float(np.max(np.abs(signal.resample_poly(x[:, c].astype(np.float64), 4, 1)))) for c in range(2))
    return 20 * np.log10(m + 1e-12)


def lufs(x: np.ndarray) -> float:
    return float(METER.integrated_loudness(x.astype(np.float64)))


def mix(cues: list[tuple[float, str, float]]) -> tuple[np.ndarray, dict]:
    end = int(round(tl.DURATION * SR))
    n = end + (-end) % 1024
    src = source_audio()
    voice = np.zeros((n, 2), np.float32)
    stats = {"clips": []}
    for name, parts, at, _ in tl.VOICE:
        clip = build_clip(src, parts)
        level_in, snr_in, thr = lufs(clip), snr_db(clip), None
        if -20.0 - level_in > 10:
            clip, thr = expand(clip)
        level = lufs(clip)
        gain = float(np.clip(-20.0 - level, -6, 24))
        _add(voice, clip, int(round(at * SR)), 10 ** (gain / 20))
        stats["clips"].append({"name": name, "parts": parts, "at": round(at, 4), "len": round(len(clip) / SR, 4),
                               "lufs_in": round(level_in, 2), "gain_db": round(gain, 2),
                               "lufs_out": round(level + gain, 2), "snr_in_db": round(snr_in, 1),
                               "snr_out_db": round(snr_db(clip), 1),
                               "expander_thr_db": None if thr is None else round(thr, 1)})

    fx = np.zeros((n, 2), np.float32)
    flicks = 0
    for t, kind, gain in sorted(cues + big_cues()):
        if kind == "flick":
            snd = sfx.flick(6 + flicks % 5, pan=((flicks * 7) % 5 - 2) * 0.12)
            flicks += 1
        else:
            snd = SOUNDS[kind]()
        _add(fx, snd, int(round(t * SR)), 10 ** (gain / 20))

    rms = np.sqrt(np.maximum(uniform_filter1d(np.mean(voice ** 2, axis=1), size=int(0.1 * SR)), 0.0))
    duck = 1 - 0.6 * np.clip(rms / 0.02, 0, 1)
    duck = uniform_filter1d(minimum_filter1d(duck, size=int(0.3 * SR)), size=int(0.1 * SR))
    bed = sfx.drone(n / SR)[:n] * (bed_gain(n) * duck)[:, None]

    raw = voice + fx + bed
    gain_db, ceiling_db = -14.0 - lufs(raw[:end]), -1.5
    for _ in range(6):
        out = limit(raw * 10 ** (gain_db / 20), 10 ** (ceiling_db / 20))
        level, tp = lufs(out[:end]), true_peak_db(out)
        if tp > -1.0:
            ceiling_db -= 0.5
        elif level < -14.4:
            gain_db += -14.0 - level
        else:
            break
    out[end:] = 0
    if not np.isfinite(out).all():
        raise ValueError("mix contains non-finite samples")
    pre = np.max(np.abs(raw[:end] * 10 ** (gain_db / 20)), axis=1)
    loud = pre > 0.05
    gr = 20 * np.log10(np.max(np.abs(out[:end]), axis=1)[loud] / pre[loud])
    stats["limiter"] = {"max_gr_db": round(float(-gr.min()), 2),
                        "pct_time_gr_over_1db": round(100 * float(np.mean(gr < -1)) * loud.mean(), 2),
                        "pct_time_gr_over_3db": round(100 * float(np.mean(gr < -3)) * loud.mean(), 2),
                        "gr_over_4db_at_s": sorted({round(float(t), 1) for t in np.flatnonzero(loud)[gr < -4] / SR})}
    stats.update({"lufs": round(level, 2), "true_peak_dbtp": round(tp, 2), "ceiling_db": ceiling_db,
                  "master_gain_db": round(gain_db, 2), "sample_peak": round(float(np.max(np.abs(out))), 4),
                  "samples": n, "end_sample": end, "cues": len(cues) + len(big_cues())})
    return out, stats
