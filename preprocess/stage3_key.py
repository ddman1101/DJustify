#!/usr/bin/env python
"""Stage 3 — 調性 key + 信心值(madmom CNN key recognition).

Env:  allin1 environment (see README.md)
權重: madmom 套件內建 CNN(CNNKeyRecognitionProcessor)。
輸入: 音檔資料夾
輸出: {out_json} = {tid: {"key": "B:min", "conf": 0.936}}   ← 消費端 agent_v0 以 root,mode=key.split(":") 解析

key 記法:sharp 拼字 + 冒號(如 "A#:maj" / "C#:min");Camelot 由消費端(agent_v0.cam)自算,不存這裡。

用法: python stage3_key.py <audio_dir> <out_json>
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
            pred = proc(f)                                  # (24,) 機率(12 音 x 大小調)
            label = key_prediction_to_label(pred)           # 例 "B minor"
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
