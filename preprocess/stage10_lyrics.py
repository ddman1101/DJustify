#!/usr/bin/env python3
"""Stage 10: lyrics for the story-set planner (Whisper on the vocal stem).

Writes, under $AIDJ_LYRICS_POOL_ROOT:
  whisper_json/<tid>.json   Whisper output with segments (start, end, text) and words
  lyrics_txt/<tid>.txt      one line per segment

The listening-test pool was transcribed this way (large-v3 on the Demucs vocal
stem) and then corrected by hand against published lyrics; correct
lyrics_txt/ by hand if you can, the selector quotes them verbatim as evidence.
Then build the chorus database: python stage10_chorus_db.py --allow-incomplete

usage: python stage10_lyrics.py [--model large-v3] [--language zh] [tid ...]
       (no tid = every track in the registry that has a vocal stem)
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(os.environ.get("AIDJ_RUNTIME_ROOT") or os.path.abspath("runtime"))
POOL = Path(os.environ.get("AIDJ_LYRICS_POOL_ROOT") or "lyrics")
STEMS = ROOT / "dataset" / "normalized_stems"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tids", nargs="*")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default=None, help="force a language code; default lets Whisper detect it")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import whisper
    model = whisper.load_model(a.model, device=a.device)
    (POOL / "whisper_json").mkdir(parents=True, exist_ok=True)
    (POOL / "lyrics_txt").mkdir(parents=True, exist_ok=True)
    tids = a.tids or sorted(p.name for p in STEMS.iterdir() if (p / "vocals.mp3").is_file() or (p / "vocals.wav").is_file())
    for tid in tids:
        out_json = POOL / "whisper_json" / f"{tid}.json"
        if out_json.is_file():
            continue
        stem = next((STEMS / tid / f for f in ("vocals.mp3", "vocals.wav") if (STEMS / tid / f).is_file()), None)
        if stem is None:
            print(f"  [lyrics] SKIP {tid}: no vocal stem", flush=True)
            continue
        r = model.transcribe(str(stem), language=a.language, word_timestamps=True, verbose=False)
        segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip(),
                 "words": [{"start": float(w["start"]), "end": float(w["end"]), "word": w["word"]} for w in s.get("words", [])]}
                for s in r["segments"] if s["text"].strip()]
        json.dump({"file": f"{tid}.mp3", "audio_used": str(stem), "language": r.get("language"),
                   "duration": segs[-1]["end"] if segs else 0.0, "segments": segs},
                  open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        (POOL / "lyrics_txt" / f"{tid}.txt").write_text("\n".join(s["text"] for s in segs) + "\n", encoding="utf-8")
        print(f"  [lyrics] {tid}: {len(segs)} lines", flush=True)
    print("=== STAGE10 LYRICS DONE ===")


if __name__ == "__main__":
    main()
