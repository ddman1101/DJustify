"""VAD-first cue candidates from the vocal stem (acoustic, lyrics-independent, symmetric).
  voiced/rest segmentation -> exit = voiced END before a real rest; entry = voiced START after a rest.
Every cue snapped to the nearest downbeat. Used by the 3-variant experiment. env: dj

Voiced segmentation backend (set env VAD_BACKEND):
  "svd"   (default) = trained CRNN singing-voice detector (svd/svdnet.pt, Jamendo F1~0.88)
  "energy"          = legacy RMS>thr threshold (for A/B comparison / fallback)"""
import os, sys, numpy as np, librosa, warnings
warnings.filterwarnings("ignore")

STEMS = os.path.join(os.environ.get("AIDJ_RUNTIME_ROOT", "runtime"), "dataset", "normalized_stems")
SR = 22050
HOP = int(0.02 * SR)            # 20 ms frames
BACKEND = os.environ.get("VAD_BACKEND", "svd")

_SVD = None
def _svd():
    global _SVD
    if _SVD is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "svd"))
        from infer import SVD
        _SVD = SVD()
    return _SVD


_VOICED_JSON = None
def _voiced_json():
    global _VOICED_JSON
    if _VOICED_JSON is None:
        import json
        jp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "svd", "voiced_segments.json")
        _VOICED_JSON = json.load(open(jp)) if os.path.isfile(jp) else {}
    return _VOICED_JSON


_SEG_CACHE = {}
def _segments(tid, dur):
    """voiced spans [(s,e),...] via the selected backend (memoized per tid — songs recur across pairs)."""
    if (BACKEND, tid) in _SEG_CACHE:
        return _SEG_CACHE[(BACKEND, tid)]
    # prefer the precomputed JSON (GPU-free, identical to what stage-2 uses)
    if BACKEND == "svd" and tid in _voiced_json():
        segs = [tuple(se) for se in _voiced_json()[tid]]
        _SEG_CACHE[(BACKEND, tid)] = segs
        return segs
    p = os.path.join(STEMS, tid, "vocals.mp3")
    if not os.path.isfile(p):
        return None
    segs = None
    if BACKEND == "svd":
        try:
            segs = _svd().voiced_segments(path=p)
        except Exception as ex:
            warnings.warn(f"SVD backend failed ({ex}); falling back to energy")
    if segs is None:
        env, t = _vocal_env(tid)
        segs = voiced_segments(env, t) if env is not None else None
    _SEG_CACHE[(BACKEND, tid)] = segs
    return segs


def _vocal_env(tid):
    p = os.path.join(STEMS, tid, "vocals.mp3")
    if not os.path.isfile(p):
        return None, None
    y, _ = librosa.load(p, sr=SR, mono=True)
    n = len(y) // HOP
    env = np.sqrt((y[:n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-9)
    t = np.arange(n) * HOP / SR
    return env, t


def voiced_segments(env, t, thr=0.04, min_voiced=0.4, min_rest=0.30):
    """Return list of (start,end) voiced spans; merge over tiny gaps (< min_rest = breath, not a real
    phrase boundary), drop too-short blips."""
    voiced = env > thr
    segs = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            segs.append([float(t[i]), float(t[min(j, len(t) - 1)])])
            i = j
        else:
            i += 1
    # merge spans separated by < min_rest (breaths)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_rest:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [s for s in merged if s[1] - s[0] >= min_voiced]


def snap(t, downbeats):
    if downbeats is None or len(downbeats) == 0:
        return t
    db = np.asarray(downbeats)
    return float(db[int(np.argmin(np.abs(db - t)))])


def snap_rest(t, downbeats, lo, hi):
    """Snap to the nearest downbeat that lies in the rest window [lo, hi] — NEVER into the voiced
    span. If no downbeat falls in the rest, keep the raw acoustic point t (don't risk cutting vocal)."""
    if downbeats is None or len(downbeats) == 0:
        return t
    db = np.asarray(downbeats)
    win = db[(db >= lo - 1e-6) & (db <= hi + 1e-6)]
    if len(win) == 0:
        return t
    return float(win[int(np.argmin(np.abs(win - t)))])


def cue_candidates(tid, downbeats, dur, real_rest=1.5, entry_rest=0.8):
    """Returns (exits, entries): lists of dicts {t, t_raw, rest_s} snapped to downbeat.
    exit  = a voiced span ends AND the gap before the next voiced span >= real_rest (phrase truly ends).
    entry = a voiced span starts AND the gap before it (since previous voiced) >= entry_rest (clean start)."""
    segs = _segments(tid, dur)
    if not segs:
        return [], []
    exits, entries = [], []
    for k, (s, e) in enumerate(segs):
        nxt = segs[k + 1][0] if k + 1 < len(segs) else dur
        gap_after = nxt - e
        prev_end = segs[k - 1][1] if k > 0 else 0.0
        gap_before = s - prev_end
        if gap_after >= real_rest and 6 < e < dur - 3:
            # exit cut lives in the rest AFTER the voiced span ends: [e, nxt]
            exits.append({"t": round(snap_rest(e, downbeats, e, nxt), 3), "t_raw": round(e, 3), "rest_s": round(gap_after, 2)})
        if gap_before >= entry_rest and 3 < s < dur - 5:
            # entry cut lives in the rest BEFORE the voiced span starts: [prev_end, s]
            entries.append({"t": round(snap_rest(s, downbeats, prev_end, s), 3), "t_raw": round(s, 3), "rest_s": round(gap_before, 2)})
    # dedup by snapped time
    def dd(lst):
        seen = set(); out = []
        for c in lst:
            k = round(c["t"], 1)
            if k not in seen:
                seen.add(k); out.append(c)
        return out
    return dd(exits), dd(entries)


if __name__ == "__main__":
    import json, sys
    A = json.load(open("outputs/pool632_analysis_struct.json", encoding="utf-8"))
    for tid in [k for k in A if any(s in k for s in ["漂向北方", "晴天", "三年二班"])][:3]:
        ex, en = cue_candidates(tid, A[tid].get("downbeat_times", []), A[tid].get("duration_sec", 240))
        print(tid.split(" [")[0], "| exits:", [(c["t"], c["rest_s"]) for c in ex[:5]],
              "| entries:", [(c["t"], c["rest_s"]) for c in en[:5]])
