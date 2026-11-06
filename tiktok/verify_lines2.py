import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_lines import OUT, load_audio  # noqa: E402

WINDOWS = [
    (193.5, 204.0), (229.0, 240.0), (294.5, 302.5), (368.5, 377.5), (377.0, 386.0), (396.0, 404.0),
    (408.5, 420.5), (420.5, 434.5), (434.0, 447.5),
]


def main(model_name: str, tag: str) -> None:
    audio = load_audio()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    out = []
    for a, b in WINDOWS:
        segs, _ = model.transcribe(audio[int(a * 16000):int(b * 16000)], language="en", beam_size=5,
                                   word_timestamps=True, condition_on_previous_text=False, vad_filter=False)
        for s in segs:
            out.append({
                "start": round(a + s.start, 3), "end": round(a + s.end, 3), "text": s.text.strip(),
                "words": [{"start": round(a + w.start, 3), "end": round(a + w.end, 3), "word": w.word,
                           "p": round(w.probability, 3)} for w in (s.words or [])],
            })
    (OUT / f"windows2-{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in out:
        print(f"{r['start']:7.2f}-{r['end']:7.2f}  {r['text']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
