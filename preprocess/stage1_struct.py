#!/usr/bin/env python
"""Stage 1: beats / BPM / downbeats and section structure (all-in-one).

Env:  allin1 environment (see README.md)
Weights: model='harmonix-all', downloaded on first run; all-in-one separates stems with Demucs internally.
Input:  a folder of .mp3/.wav
Output: {out_json} = {track_id: {bpm, beat_times, downbeat_times, bars, segments:[{index,start,end,label}]}}

Section labels use the Harmonix vocabulary (start/intro/verse/chorus/bridge/inst/solo/outro/end).
Section energy is added later by assemble_reg.py.

usage: python stage1_struct.py <audio_dir> <out_json> [--device cuda]
"""
import sys, os, json, glob, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("out_json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import allin1fix  # noqa: an import error in natten/DiNAT means a torch/natten version mismatch

    files = sorted(glob.glob(os.path.join(a.audio_dir, "*.mp3")) +
                   glob.glob(os.path.join(a.audio_dir, "*.wav")))
    out = json.load(open(a.out_json, encoding="utf-8")) if os.path.isfile(a.out_json) else {}
    for f in files:
        tid = os.path.splitext(os.path.basename(f))[0]
        if tid in out:
            continue
        try:
            r = allin1fix.analyze(f, out_dir=None, device=a.device, overwrite=False)
            r = r[0] if isinstance(r, list) else r
            beats = [float(x) for x in r.beats]
            downs = [float(x) for x in r.downbeats]
            segs = [{"index": i, "start": float(s.start), "end": float(s.end), "label": s.label}
                    for i, s in enumerate(r.segments)]
            # bpm from the median beat interval (as in the registry); r.bpm is the fallback
            bpm = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 1 else float(r.bpm)
            out[tid] = {"bpm": round(bpm, 1), "beat_times": beats, "downbeat_times": downs,
                        "bars": len(downs), "segments": segs}
            json.dump(out, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  [struct] {tid}: bpm={bpm:.0f} bars={len(downs)} segs={len(segs)}", flush=True)
        except Exception as e:
            print(f"  [struct] SKIP {tid}: {repr(e)[:80]}", flush=True)
    print("=== STAGE1 STRUCT DONE ===")


if __name__ == "__main__":
    main()
