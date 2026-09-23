"""Voiced segments per song = UNION of trained-SVD voiced ∪ energy-threshold voiced.
The union is conservative on purpose: if EITHER detector hears singing, treat it as voiced, so a
cut is never placed where the SVD has a false-negative (e.g. soft Mandarin outros it misses).
-> svd/voiced_segments.json {tid: [[s,e],...]}.  env: dj   usage: CUDA_VISIBLE_DEVICES=2 python precompute_voiced.py"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from infer import SVD
import vad_cues as VC

STEMS = os.path.join(os.environ.get("AIDJ_RUNTIME_ROOT", "runtime"), "dataset", "normalized_stems")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voiced_segments.json")


def union(segs, merge_gap=0.25):
    segs = sorted([list(x) for x in segs])
    out = []
    for s, e in segs:
        if out and s <= out[-1][1] + merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [[round(s, 3), round(e, 3)] for s, e in out]


svd = SVD()
out = {}
dirs = sorted(glob.glob(os.path.join(STEMS, "*")))
for i, d in enumerate(dirs):
    tid = os.path.basename(d)
    vp = os.path.join(d, "vocals.mp3")
    if not os.path.isfile(vp):
        continue
    try:
        svd_segs = svd.voiced_segments(path=vp)
        env, t = VC._vocal_env(tid)
        e_segs = VC.voiced_segments(env, t) if env is not None else []
        out[tid] = union(list(svd_segs) + [tuple(x) for x in e_segs])
    except Exception as ex:
        print(f"  skip {tid}: {ex}")
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(dirs)}", flush=True)

json.dump(out, open(OUT, "w"), ensure_ascii=False)
print(f"wrote {OUT}  ({len(out)} songs, SVD∪energy union)")
