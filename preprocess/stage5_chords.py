# -*- coding: utf-8 -*-
"""Stage 5: CREMA chords for the pool (602 classes incl. sevenths and inversions) plus the main repeated progression.

Runs in the crema environment (tensorflow-cpu < 2.16).
Input: variants/chords_manifest.json (tid -> audio_path + downbeats).

Votes at half-bar resolution: on slow songs the downbeat grid often runs at half speed (one grid bar = two real bars),
while CREMA changes chords every ~1.7 s; grid = downbeats + midpoints keeps ii-V motion visible.
Main loop: try n-grams with n in {4, 8}, keep the one with the highest coverage (no "N", no single-chord loops).
Output: variants/chords_crema/{tid}.json = {segments, grid_chords, main_loop, n, occurrences, coverage}
usage: CUDA_VISIBLE_DEVICES= python chords_crema.py [--shard i/n]   (venv_crema)
Marker: === CREMA CHORDS DONE ===
"""
import os
ROOT = os.environ.get("AIDJ_RUNTIME_ROOT") or os.path.abspath("runtime")
VR = os.path.join(ROOT, "dj_transition_planner", "eval", "variants")
import os, json, argparse
from collections import Counter

OUT = f"{VR}/chords_crema"
os.makedirs(OUT, exist_ok=True)

MAN = json.load(open(f"{VR}/chords_manifest.json", encoding="utf-8"))


def grid_vote(segs, downs):
    grid = []
    for i in range(len(downs) - 1):
        d0, d1 = downs[i], downs[i + 1]
        grid += [(d0, (d0 + d1) / 2), ((d0 + d1) / 2, d1)]
    out = []
    for g0, g1 in grid:
        votes = Counter()
        for s, e, lab in segs:
            ov = max(0.0, min(e, g1) - max(s, g0))
            if ov > 0:
                votes[lab] += ov
        out.append({"t0": round(g0, 2), "chord": votes.most_common(1)[0][0] if votes else "N"})
    return out


def main_loop(cells):
    seq = [c["chord"] for c in cells]
    best = (None, [], 0.0, 0)
    for n in (4, 8):
        grams = Counter()
        for i in range(len(seq) - n + 1):
            g = tuple(seq[i:i + n])
            if "N" not in g:
                grams[g] += 1
        ranked = [(g, c) for g, c in grams.most_common(12) if len(set(g)) >= 2]
        if not ranked:
            continue
        g, _ = ranked[0]
        occ = [i for i in range(len(seq) - n + 1) if tuple(seq[i:i + n]) == g]
        kept, last = [], -n
        for i in occ:
            if i >= last + n:
                kept.append(i); last = i
        cover = len(kept) * n / max(len(seq), 1)
        if cover > best[2]:
            best = (list(g), [{"cell": i, "t0": cells[i]["t0"]} for i in kept], round(cover, 3), n)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    sh, nsh = map(int, a.shard.split("/"))
    from crema.models.chord import ChordModel
    model = ChordModel()
    tids = [t for i, t in enumerate(sorted(MAN)) if i % nsh == sh]
    for i, tid in enumerate(tids):
        outp = f"{OUT}/{tid}.json"
        if os.path.isfile(outp):
            continue
        try:
            ann = model.predict(filename=MAN[tid]["audio_path"])
            segs = [(float(o.time), float(o.time) + float(o.duration), str(o.value)) for o in ann.data]
            cells = grid_vote(segs, MAN[tid]["downbeats"])
            loop, occ, cover, n = main_loop(cells)
            json.dump({"tid": tid,
                       "segments": [{"t0": round(s, 2), "t1": round(e, 2), "chord": c} for s, e, c in segs],
                       "grid_chords": cells, "main_loop": loop, "n": n,
                       "occurrences": occ, "coverage": cover},
                      open(outp, "w"), ensure_ascii=False)
            print(f"[{i+1}/{len(tids)}] {tid.split(' [')[0][:22]:22s} "
                  f"loop={'-'.join(loop) if loop else 'none'} (n={n}) x{len(occ)} coverage {cover:.0%}", flush=True)
        except Exception as e:
            print(f"✗ {tid[:30]}: {str(e)[:70]}", flush=True)
    print("=== CREMA CHORDS DONE ===", flush=True)


if __name__ == "__main__":
    main()
