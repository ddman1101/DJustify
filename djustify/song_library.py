# -*- coding: utf-8 -*-
"""Transition Agent v0 — LLM proposal × deterministic verifier loop (LLM-Modulo instance)."""
import os, io, json, re, argparse, random, tempfile, subprocess, time
import numpy as np
import librosa
import soundfile as sf
import requests
import scipy.signal as ss

ROOT = os.environ.get("AIDJ_RUNTIME_ROOT") or os.path.abspath("runtime")
EV = f"{ROOT}/dj_transition_planner/eval"
VR = f"{EV}/variants"
RUNS = f"{VR}/agent_runs"
os.makedirs(RUNS, exist_ok=True)
SR = 44100
MAX_ITERS = 5
SEVERITY = {"D1_exit_vocal": 3.0, "D2_entry_phrase": 3.0, "D4_chorus": 2.0, "D9_bpm": 2.0,
            "D10_genre_tool": 3.0,
            "D7_min_play": 2.0, "D3_energy": 1.0, "D5_key": 1.0, "D6_deadair": 2.0}

REG = json.load(open(os.environ.get("AIDJ_POOL_ANALYSIS") or f"{ROOT}/dj_transition_planner/outputs/pool_analysis.json", encoding="utf-8"))
_VAMP = f"{ROOT}/dj_transition_planner/outputs/va_micl.json"
if os.path.isfile(_VAMP):
    for _t, _m in json.load(open(_VAMP)).items():
        if _t in REG and "arousal_q" in _m:
            REG[_t]["valence"] = _m["valence_q"]
            REG[_t]["arousal"] = _m["arousal_q"]
VOICED = json.load(open(f"{ROOT}/dj_transition_planner/svd/voiced_segments.json"))
_QP = f"{ROOT}/dj_transition_planner/eval/variants/quarantine.json"
_Q = json.load(open(_QP)) if os.path.isfile(_QP) else {}
QUAR = set(_Q.get("quarantined", []))
SUSPECT = set(_Q.get("identity_suspect", []))
MAN = json.load(open(f"{VR}/chords_manifest.json", encoding="utf-8"))
KEYS = json.load(open(f"{VR}/keys.json", encoding="utf-8"))
_CP = os.environ.get("CAND_POOL")
POOL400 = set(json.load(open(_CP if os.path.isabs(_CP) else f"{VR}/{_CP}"))) if _CP else None
_AUDIO = {}
_MJ = None
def audio(tid):
    if tid not in _AUDIO:
        _AUDIO[tid] = librosa.load(MAN[tid]["audio_path"], sr=SR, mono=True)[0]
    return _AUDIO[tid]


def beat_of(tid):
    d = REG[tid]["downbeat_times"]
    return (d[1] - d[0]) / 4


def chorus_groups(tid):
    segs = REG[tid]["segments"]
    g = []
    for s in segs:
        if s["label"] == "chorus":
            if g and abs(g[-1]["end"] - s["start"]) < 0.5:
                g[-1]["end"] = s["end"]; g[-1]["energy"] = max(g[-1]["energy"], s["energy"])
            else:
                g.append({"start": s["start"], "end": s["end"], "energy": s["energy"]})
    return g


def phrase_onsets(tid):
    ons, prev = [], -9
    for s0, e in VOICED.get(tid, []):
        if s0 - prev >= 0.5:
            ons.append(s0)
        prev = e
    return ons


def seg_at(tid, t):
    for s in REG[tid]["segments"]:
        if s["start"] <= t < s["end"]:
            return s
    return {"label": "?", "energy": 0.0}


def exit_base(tid, which=1):
    """v3.1 mixed anchor for the which-th chorus group (0-based)."""
    gs = chorus_groups(tid)
    if not gs:
        return None
    g = gs[min(which, len(gs) - 1)]
    ge = g["end"]
    spans = VOICED.get(tid, [])
    v_end = max((e for s0, e in spans if e <= ge + 1.0), default=ge)
    s_next = min((s0 for s0, e in spans if s0 > v_end + 0.05), default=1e9)
    downs = REG[tid]["downbeat_times"]
    cand = [d for d in downs if v_end - 0.1 <= d <= s_next - 0.3]
    if cand:
        return cand[0]
    beat = beat_of(tid)
    beats = sorted(d + k * beat for d in downs for k in range(4))
    cand = [b for b in beats if v_end - 0.1 <= b <= s_next - 0.2]
    return cand[0] if cand else min(downs, key=lambda d: abs(d - v_end))


def entry_options(tid):
    a = REG[tid]; segs = a["segments"]; dur = a["duration_sec"]
    out = {}
    v1 = next((s for s in segs if s["label"] == "verse"), None)
    if v1: out["v_first"] = v1["start"]
    cho = [s for s in segs if s["label"] == "chorus"]
    if cho:
        out["c_mid"] = min(cho, key=lambda s: abs(s["start"] - dur * 0.4))["start"]
        after = [s for s in segs if s["label"] == "verse" and s["start"] > cho[0]["end"]]
        if after: out["v_after"] = after[0]["start"]
    try:
        cj = json.load(open(f"{VR}/chords_crema/{tid}.json", encoding="utf-8"))
        out["loop"] = min((o["t0"] for o in cj["occurrences_triad"]),
                          key=lambda t: abs(t - dur * 0.45))
    except Exception:
        pass
    ons = phrase_onsets(tid)
    spans = VOICED.get(tid, [])
    beat = beat_of(tid)
    ref = {}
    for k, t0 in out.items():
        cand = [o for o in ons if abs(o - t0) <= 15]
        o = min(cand, key=lambda x: abs(x - t0)) if cand else t0
        prev_end = max((e for s0, e in spans if e <= o - 0.05), default=o - 4)
        grid = [d for d in REG[tid]["downbeat_times"] if prev_end + 0.05 <= d <= o - 0.02]
        if not grid:
            beats = sorted(d + j * beat for d in REG[tid]["downbeat_times"] for j in range(4))
            grid = [b for b in beats if prev_end + 0.05 <= b <= o - 0.02]
        ref[k] = grid[-1] if grid else max(o - 0.05, prev_end + 0.05)
    return ref


# ── Verifier(D1-D5)─────────────────────────────────────────────────────────
def _rb_pitch(y, semis):
    if semis == 0:
        return y
    fi = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); fi.close()
    fo = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); fo.close()
    sf.write(fi.name, y, SR)
    subprocess.run(["rubberband", "-p", str(semis), fi.name, fo.name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    o, _ = librosa.load(fo.name, sr=SR, mono=True)
    os.unlink(fi.name); os.unlink(fo.name)
    return o

def _rb_stretch(y, rate):
    """rubberband time stretch (rate>1 = faster/shorter, same semantics as librosa.effects.time_stretch)."""
    if abs(rate - 1.0) < 1e-4 or len(y) < SR // 4:
        return y
    fi = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); fi.close()
    fo = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); fo.close()
    sf.write(fi.name, y.astype("float32"), SR)
    r = subprocess.run(["rubberband", "-T", f"{float(rate):.6f}", "-c", "5", fi.name, fo.name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not os.path.isfile(fo.name):
        os.unlink(fi.name)
        return librosa.effects.time_stretch(y, rate=float(rate))
    o, _ = librosa.load(fo.name, sr=SR, mono=True)
    os.unlink(fi.name); os.unlink(fo.name)
    return o.astype(y.dtype, copy=False)





def _vstem_rms(tid, t0, t1):
    pth = f"{ROOT}/dataset/normalized_stems/{tid}/vocals.mp3"
    if not os.path.isfile(pth) or t1 <= t0:
        return None
    y, _ = librosa.load(pth, sr=22050, mono=True, offset=max(0, t0), duration=t1 - t0)
    return float(np.sqrt((y ** 2).mean() + 1e-12)) if len(y) else None


POOL_TIDS = None
if os.environ.get("DJ_POOL", "") == "exp":
    try:
        _pc = json.load(open(f"{ROOT}/dj_transition_planner/eval/variants/pool_curated.json"))
        POOL_TIDS = {r["tid"] for r in _pc.values() if r.get("tid") and r.get("status") in ("existing", "ingested")}
        print(f"[pool] DJ_POOL=exp:{len(POOL_TIDS)} tracks", flush=True)
    except Exception as _e:
        print(f"[pool] exp pool load failed: {_e}", flush=True)

def in_pool(t):
    return POOL_TIDS is None or str(t) in POOL_TIDS
