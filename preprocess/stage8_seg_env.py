# -*- coding: utf-8 -*-
"""Stage 8: 4-point energy envelope and trend label per section -> outputs/seg_env.json
{tid: {"segs": [{"i":0,"env":[..4..],"trend":"平穩|漸強|漸弱|fade至無聲"}...],   # steady | rising | falling | fade to silence (labels are read as-is by the planner)
       "fade_start": seconds or null}}"""
import os, sys, json
import os, sys
ROOT = os.environ.get("AIDJ_RUNTIME_ROOT") or os.path.abspath("runtime")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "djustify"))
import numpy as np
import librosa
import song_library as A

OUT = "../outputs/seg_env.json"
res = json.load(open(OUT)) if os.path.isfile(OUT) else {}
tids = [t for t in A.REG if A.REG[t].get("segments")]
print("tracks:", len(tids), "done:", len(res), flush=True)

for n, tid in enumerate(tids):
    if str(tid) in res: continue
    try:
        path = A.REG[tid].get("audio_path") or f"{A.ROOT}/dataset/normalized_pool/{tid}.mp3"
        if not os.path.isfile(path):
            path = f"{A.ROOT}/dataset/normalized_pool/{tid}.mp3"
        y, sr = librosa.load(path, sr=22050, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=2205)[0]  # 0.1s hop
        ref = np.percentile(rms, 95) + 1e-9
        rn = np.clip(rms / ref, 0, 1.2)
        dur = len(y) / sr
        segs = []
        for si, sg in enumerate(A.REG[tid]["segments"]):
            s0, e0 = float(sg["start"]), float(sg["end"])
            i0, i1 = int(s0 * 10), max(int(e0 * 10), int(s0 * 10) + 4)
            v = rn[i0:i1]
            if len(v) == 0:
                v = np.zeros(4)
            elif len(v) < 4:
                v = np.pad(v, (0, 4 - len(v)), mode="edge")
            q = [round(float(v[int(len(v) * k / 4):int(len(v) * (k + 1) / 4)].mean()), 2)
                 for k in range(4)]
            is_last = (e0 >= dur - 2.0)
            if is_last and q[3] < 0.15 and q[0] > q[3] + 0.12 and q[0] >= q[1] >= q[2] * 0.8:
                tr = "fade至無聲"
            elif q[3] - q[0] > 0.15: tr = "漸強"
            elif q[0] - q[3] > 0.15: tr = "漸弱"
            else: tr = "平穩"
            segs.append({"i": si, "env": q, "trend": tr})
        # fade_start: the point after which the smoothed dB curve never climbs back above ref-5 dB;
        # requires a total drop >= 15 dB, an ending <= -25 dB and a span >= 6 s
        fade_start = None
        db = 20 * np.log10(np.maximum(rms, 1e-6))
        ref = np.percentile(db, 90)
        sm = np.convolve(db - ref, np.ones(20) / 20, mode="same")   # 2 s smoothing on a 0.1 s grid
        if len(sm) > 100 and sm[-15:].mean() < -25:
            above = np.where(sm > -5)[0]
            if len(above):
                k = int(above[-1])                                   # last frame still above -5 dB
                span = (len(sm) - k) * 0.1
                drop = sm[k] - sm[-15:].mean()
                if span >= 6 and drop >= 15:
                    fade_start = round(k * 0.1, 1)
        res[str(tid)] = {"segs": segs, "fade_start": fade_start}
    except Exception as e:
        res[str(tid)] = {"err": repr(e)[:80]}
    if n % 25 == 0:
        json.dump(res, open(OUT, "w"), ensure_ascii=False)
        print(f"{n}/{len(tids)}", flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False)
nf = sum(1 for v in res.values() if v.get("fade_start"))
print("=== SEG ENV DONE ===", len(res), "with fade_start:", nf, flush=True)
