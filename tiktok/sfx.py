"""Sound effects synthesized from scratch (48 kHz, stereo float32). No samples, no music."""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 48000


def _n(d: float) -> int:
    return max(1, int(round(d * SR)))


def _t(d: float) -> np.ndarray:
    return np.arange(_n(d)) / SR


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def _filt(x, kind, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype=kind, fs=SR, output="sos"), x)


def _phase(f: np.ndarray) -> np.ndarray:
    return 2 * np.pi * np.cumsum(f) / SR


def _norm(x: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return x * (peak / m)


def _stereo(x: np.ndarray, pan=0.0) -> np.ndarray:
    """Constant-power pan; pan may be a scalar or a per-sample array in [-1, 1]."""
    p = (np.asarray(pan, dtype=np.float64) + 1) * np.pi / 4
    return np.stack([x * np.cos(p) * np.sqrt(2), x * np.sin(p) * np.sqrt(2)], axis=1).astype(np.float32)


def sweep_noise(dur: float, f0: float, f1: float, width_oct: float = 0.7, seed: int = 0) -> np.ndarray:
    """Noise through a band-pass whose centre glides from f0 to f1 (STFT mask)."""
    n = _n(dur)
    x = _noise(n + 4096, seed)
    f, tt, z = signal.stft(x, fs=SR, nperseg=1024, noverlap=768)
    fc = f0 * (f1 / f0) ** np.clip(tt / dur, 0, 1)
    lf = np.log2(np.maximum(f, 1.0)[:, None] / fc[None, :])
    z = z * np.exp(-0.5 * (lf / width_oct) ** 2)
    _, y = signal.istft(z, fs=SR, nperseg=1024, noverlap=768)
    return y[:n]


def whoosh(dur: float = 0.45, up: bool = True, seed: int = 1) -> np.ndarray:
    t = _t(dur)
    p = t / dur
    x = sweep_noise(dur, *((250, 4500) if up else (4500, 250)), 0.6, seed)
    env = (p ** 1.4) * (1 - p) ** 0.6
    pan = np.linspace(-0.6, 0.6, len(t)) if up else np.linspace(0.6, -0.6, len(t))
    return _stereo(_norm(x * env, 0.8), pan)


def boom(dur: float = 1.8, f0: float = 95, f1: float = 36, seed: int = 2) -> np.ndarray:
    t = _t(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.12)
    body = np.sin(_phase(f)) * np.exp(-t / 0.55)
    click = _filt(_noise(len(t), seed), "lowpass", 1800) * np.exp(-t / 0.018)
    rumble = _filt(_noise(len(t), seed + 1), "lowpass", 120) * np.exp(-t / 0.5)
    x = body + 0.35 * _norm(click, 1) + 0.35 * _norm(rumble, 1)
    x *= 1 - np.exp(-t / 0.002)
    return _stereo(_norm(np.tanh(1.6 * x), 0.95))


def sub_drop(dur: float = 2.2) -> np.ndarray:
    t = _t(dur)
    f = 30 + 34 * np.exp(-t / 0.5)
    x = np.sin(_phase(f)) * np.exp(-t / 0.9) * (1 - np.exp(-t / 0.01))
    return _stereo(_norm(np.tanh(1.3 * x), 0.9))


def heartbeat() -> np.ndarray:
    def thump(f0, f1, amp, dur=0.24):
        t = _t(dur)
        f = f1 + (f0 - f1) * np.exp(-t / 0.04)
        return amp * np.sin(_phase(f)) * (1 - np.exp(-t / 0.004)) * np.exp(-t / 0.06)

    out = np.zeros(_n(0.6))
    a, b = thump(72, 42, 1.0), thump(64, 40, 0.7)
    out[:len(a)] += a
    off = _n(0.19)
    out[off:off + len(b)] += b
    return _stereo(_norm(_filt(out, "lowpass", 170), 0.9))


def tick(high: bool = True, seed: int = 4) -> np.ndarray:
    t = _t(0.05)
    f = 3100 if high else 2300
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.006)
    x += 0.5 * _filt(_noise(len(t), seed), "highpass", 2500) * np.exp(-t / 0.003)
    return _stereo(_norm(x, 0.6), 0.15 if high else -0.15)


def riser(dur: float = 1.4, seed: int = 5) -> np.ndarray:
    t = _t(dur)
    p = t / dur
    nz = _norm(sweep_noise(dur, 300, 7000, 0.8, seed), 1)
    f = 180 * 6 ** p
    tone = sum(np.sin(_phase(f * k)) / k for k in (1, 2, 3)) / 1.8
    trem = 0.75 + 0.25 * np.sin(_phase(4 + 16 * p))
    x = (0.6 * nz + 0.4 * tone) * p ** 2 * trem
    return _stereo(_norm(x, 0.85))


def ding(f0: float = 1046.5) -> np.ndarray:
    t = _t(1.6)
    parts = [(1.0, 1.0, 0.9), (2.0, 0.5, 0.5), (3.0, 0.25, 0.3), (4.2, 0.15, 0.18)]
    x = sum(a * np.sin(2 * np.pi * f0 * m * t) * np.exp(-t / d) for m, a, d in parts)
    return _stereo(_norm(x * (1 - np.exp(-t / 0.002)), 0.6))


def pop() -> np.ndarray:
    t = _t(0.09)
    f = 380 + 900 * np.exp(-t / 0.012)
    x = np.sin(_phase(f)) * np.exp(-t / 0.025) * (1 - np.exp(-t / 0.0015))
    return _stereo(_norm(x, 0.7))


def flick(seed: int = 6, pan: float = 0.0) -> np.ndarray:
    t = _t(0.07)
    x = _filt(_noise(len(t), seed), "bandpass", [1800, 9000]) * np.exp(-t / 0.012) * (1 - np.exp(-t / 0.001))
    x += 0.3 * _filt(_noise(len(t), seed + 1), "bandpass", [400, 1500]) * np.exp(-t / 0.02)
    return _stereo(_norm(x, 0.6), pan)


def chips(n: int = 4, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(_n(0.5))
    pos = 0
    for i in range(n):
        t = _t(0.06)
        f = rng.uniform(4200, 6200)
        c = 0.6 * np.sin(2 * np.pi * f * t) + _filt(_noise(len(t), seed + i), "highpass", 3000)
        out[pos:pos + len(t)] += c * np.exp(-t / rng.uniform(0.008, 0.016)) * rng.uniform(0.5, 1.0)
        pos += _n(rng.uniform(0.03, 0.08))
    return _stereo(_norm(out, 0.6))


def rewind(dur: float = 0.55, seed: int = 8) -> np.ndarray:
    t = _t(dur)
    p = t / dur
    f = 900 + 2600 * p
    chatter = 0.5 + 0.5 * np.sign(np.sin(_phase(28 + 40 * p)))
    tone = 0.5 * np.sin(_phase(f)) + 0.25 * np.sin(_phase(f * 1.5))
    nz = _norm(_filt(_noise(len(t), seed), "bandpass", [1500, 6000]), 0.5)
    x = (tone + nz) * (0.4 + 0.6 * chatter) * np.sin(np.pi * p) ** 0.5
    return _stereo(_norm(x, 0.6))


def drone(dur: float, root: float = 55.0, seed: int = 10) -> np.ndarray:
    """Quiet minor-chord pad with slow movement, used as a tension bed."""
    t = _t(dur)
    x = np.zeros(len(t))
    for i, (ratio, amp) in enumerate([(1, 1.0), (1.4983, 0.6), (2, 0.5), (2 * 1.1892, 0.3), (3, 0.15)]):
        det = 1 + 0.002 * np.sin(2 * np.pi * 0.07 * (i + 1) * t + i)
        x += amp * np.sin(2 * np.pi * root * ratio * np.cumsum(det) / SR)
    x *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.11 * t)
    x += 0.08 * _norm(_filt(_noise(len(t), seed), "lowpass", 400), 1)
    left = x
    right = np.roll(x, _n(0.012))
    return np.stack([_norm(left, 0.5), _norm(right, 0.5)], axis=1).astype(np.float32)
