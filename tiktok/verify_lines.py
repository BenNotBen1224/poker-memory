import json
import sys
from pathlib import Path

import av
import numpy as np
from faster_whisper import WhisperModel

SRC = Path(__file__).resolve().parents[1] / "examples" / "real" / "wpt_rolling_thunder_fold.webm"
OUT = Path(__file__).resolve().parent / "plan"
WINDOWS = [
    (22.5, 27.5), (45.0, 54.5), (61.5, 70.0), (204.5, 214.5), (213.5, 227.5), (227.0, 240.0),
    (243.0, 253.0), (272.0, 287.0), (311.0, 318.0), (327.0, 334.0), (367.0, 380.0),
    (397.0, 418.0), (420.0, 447.5),
]
PROMPT = "World Poker Tour final table. Ian Steinman and Joe McKeehen. Pocket kings, all in, time extensions, time chips."


def load_audio() -> np.ndarray:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(str(SRC)) as c:
        for frame in c.decode(c.streams.audio[0]):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
    chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def main(model_name: str) -> None:
    audio = load_audio()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    out = []
    for a, b in WINDOWS:
        segs, _ = model.transcribe(audio[int(a * 16000):int(b * 16000)], language="en", beam_size=5,
                                   word_timestamps=True, initial_prompt=PROMPT,
                                   condition_on_previous_text=False, vad_filter=False)
        for s in segs:
            out.append({
                "start": round(a + s.start, 3), "end": round(a + s.end, 3), "text": s.text.strip(),
                "words": [{"start": round(a + w.start, 3), "end": round(a + w.end, 3), "word": w.word,
                           "p": round(w.probability, 3)} for w in (s.words or [])],
            })
    safe = model_name.replace("/", "_")
    (OUT / f"windows-{safe}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in out:
        print(f"{r['start']:7.2f}-{r['end']:7.2f}  {r['text']}")


if __name__ == "__main__":
    main(sys.argv[1])
