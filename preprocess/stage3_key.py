#!/usr/bin/env python
"""Stage 3: key and confidence (madmom CNN key recognition).

Env:  allin1 environment (see README.md)
Weights: the CNN bundled with madmom (CNNKeyRecognitionProcessor).
Input:  audio folder
Output: {out_json} = {tid: {"key": "B:min", "conf": 0.936}}   (song_library parses root, mode = key.split(":"))

Key spelling: sharps plus a colon ("A#:maj", "C#:min"); the Camelot position is derived by the consumer, not stored here.

usage: python stage3_key.py <audio_dir> <out_json>
"""
import sys, os, json, glob
import numpy as np


def main():
    audio_dir, out_json = sys.argv[1], sys.argv[2]
    from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
    proc = CNNKeyRecognitionProcessor()

    out = json.load(open(out_json, encoding="utf-8")) if os.path.isfile(out_json) else {}
    files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")) + glob.glob(os.path.join(audio_dir, "*.wav")))
    for f in files:
        tid = os.path.splitext(os.path.basename(f))[0]
        if tid in out:
            continue
        try:
            pred = proc(f)                                  # (24,) probabilities (12 roots x major/minor)
            label = key_prediction_to_label(pred)           # e.g. "B minor"
            conf = float(np.max(pred))
            key = label.replace(" major", ":maj").replace(" minor", ":min")
            out[tid] = {"key": key, "conf": round(conf, 3)}
            json.dump(out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  [key] {tid}: {key} ({conf:.2f})", flush=True)
        except Exception as e:
            print(f"  [key] SKIP {tid}: {repr(e)[:80]}", flush=True)
    print("=== STAGE3 KEY DONE ===")


if __name__ == "__main__":
    main()
