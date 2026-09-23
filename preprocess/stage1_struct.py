#!/usr/bin/env python
"""Stage 1 — 節拍/BPM/小節線 + 段落結構(allin1fix all-in-one).

Env:  allin1 environment (see README.md)
權重: model='harmonix-all',首次執行自動下載到 torch-hub/HF cache;另用 Demucs 分軌。
輸入: 一個資料夾的 .mp3/.wav
輸出: {out_json}  = {track_id: {bpm, beat_times, downbeat_times, bars, segments:[{index,start,end,label}]}}

段落標籤 = Harmonix 詞彙(start/intro/verse/chorus/bridge/inst/solo/outro/end),對得上 REG。
energy 欄位不在這裡算(段落能量在 assemble_reg.py 用 librosa 補)。

用法: python stage1_struct.py <audio_dir> <out_json> [--device cuda]
"""
import sys, os, json, glob, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("out_json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import allin1fix  # noqa: 若 import 掛在 natten/DiNAT,是 torch/natten 版本不合,非本檔問題

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
            # bpm 由拍距中位數導出(與 REG 一致);allin1 也回 r.bpm 可當備援
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
