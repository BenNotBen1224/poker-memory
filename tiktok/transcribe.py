import json
import sys
import time
from pathlib import Path

import av
import numpy as np
from faster_whisper import WhisperModel

SRC = Path(__file__).resolve().parents[1] / "examples" / "real" / "wpt_rolling_thunder_fold.webm"
OUT = Path(__file__).resolve().parent / "plan"
PROMPT = "World Poker Tour final table. Ian Steinman and Joe McKeehen. Pocket kings, set of kings, all in, time bank, Broadway."


def load_audio() -> np.ndarray:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(str(SRC)) as c:
        for frame in c.decode(c.streams.audio[0]):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
    chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def main(model_name: str) -> None:
    started = time.time()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        load_audio(),
        language="en",
        beam_size=5,
        word_timestamps=True,
        initial_prompt=PROMPT,
        condition_on_previous_text=False,
    )
    rows = []
    for seg in segments:
        rows.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 3),
            "no_speech_prob": round(seg.no_speech_prob, 3),
            "words": [
                {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word, "p": round(w.probability, 3)}
                for w in (seg.words or [])
            ],
        })
    result = {
        "model": model_name,
        "source": SRC.relative_to(SRC.parents[2]).as_posix(),
        "duration": info.duration,
        "seconds": round(time.time() - started, 1),
        "segments": rows,
    }
    safe = model_name.replace("/", "_")
    (OUT / f"transcript-{safe}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    with (OUT / f"transcript-{safe}.txt").open("w", encoding="utf-8") as fh:
        for r in rows:
            m, s = divmod(r["start"], 60)
            fh.write(f"{int(m):02d}:{s:05.2f}  {r['text']}\n")
    print(model_name, "segments", len(rows), "seconds", result["seconds"])


if __name__ == "__main__":
    main(sys.argv[1])
