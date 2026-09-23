# -*- coding: utf-8 -*-
"""Stage 7: sound events for the pool -> outputs/cue_events.json
Per track: kick_entries / bass_entries / chord_resets / chord_changes (vocals and sections already live in the registry / voiced file)."""
import os, sys, json
import os, sys
ROOT = os.environ.get("AIDJ_RUNTIME_ROOT") or os.path.abspath("runtime")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "djustify"))
import numpy as np, librosa
import song_library as A
STEM4 = f"{ROOT}/dataset/normalized_stems"
OUTP = f"{ROOT}/dj_transition_planner/outputs/cue_events.json"
out = json.load(open(OUTP)) if os.path.isfile(OUTP) else {}

def stem_entries(path, min_quiet=1.2, min_loud=1.0):
    try:
        y, sr = librosa.load(path, sr=11025, mono=True)
    except Exception:
        return []
    hop = int(0.1 * sr)
    e = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    if not (e > 1e-5).any():
        return []
    act = e > max(np.percentile(e[e > 1e-5], 75) * 0.25, 1e-4)
    res, i, n = [], 0, len(act)
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]: j += 1
            quiet = 0; k = i - 1
            while k >= 0 and not act[k]: quiet += 0.1; k -= 1
            if (j - i) * 0.1 >= min_loud and quiet >= min_quiet:
                res.append(round(i * 0.1, 2))
            i = j
        else:
            i += 1
    return res

def chord_ev(tid):
    f = f"variants/chords_crema/{tid}.json"
    if not os.path.isfile(f): return [], []
    d = json.load(open(f))
    resets = [round(o["t0"], 2) for o in d.get("occurrences", [])]
    segs = d.get("segments", []); nov = []
    for i in range(1, len(segs)):
        p, c = segs[i-1], segs[i]
        if c["chord"] != "N" and p["chord"] != c["chord"] and (p["t1"] - p["t0"]) >= 6.0:
            nov.append(round(c["t0"], 2))
    return resets, nov

todo = [str(t) for t in A.REG if A.REG[t].get("bpm") and A.REG[t].get("segments") and str(t) not in out]
print("todo:", len(todo), flush=True)
for k, t in enumerate(todo):
    r = {"kick": stem_entries(f"{STEM4}/{t}/drums.mp3"),
         "bass": stem_entries(f"{STEM4}/{t}/bass.mp3")}
    r["chord_reset"], r["chord_change"] = chord_ev(t)
    out[t] = r
    if k % 25 == 0:
        json.dump(out, open(OUTP, "w"), ensure_ascii=False)
        print(f"{k}/{len(todo)}", t[:30], flush=True)
json.dump(out, open(OUTP, "w"), ensure_ascii=False)
print("=== CUE EVENTS ALL DONE ===", len(out), flush=True)
