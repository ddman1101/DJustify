# -*- coding: utf-8 -*-
"""plan_sim: pre-render plan simulator (inner loop). C1/C2 use same checker formula (stem RMS, no render) for precise prediction; also provides B vocal onset, A phrase-end silence, and c1 repair advice (scan forward for nearest safe cut). usage: from plan_sim import sim_plan; sim_plan(ta, tb, plan_dict)"""
import sys, os
EV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EV)
import song_library as A

C1_SKIP = {"move1", "move2", "move3", "move4", "blend16", "blendecho", "introstack"}

def _v(tid, t0, t1):
    return A._vstem_rms(tid, t0, t1) or 0.0

def _voiced(tid, t0, t1):
    return any(s < t1 and e > t0 for s, e in A.VOICED.get(tid, []))

def sim_c1(ta, cut, tool=None):
    """Returns (ratio, fail). Uses the same formula as regression_check."""
    if tool in C1_SKIP:
        return 0.0, False
    ref = _v(ta, cut - 3.0, cut) or 1e-9
    after = _v(ta, cut, cut + 0.8)
    _inside = any(s0 + 0.15 < cut < e0 - 0.15 for s0, e0 in A.VOICED.get(ta, []))
    _next_onset = any(cut + 0.1 <= s0 <= cut + 0.8 for s0, e0 in A.VOICED.get(ta, []))
    _gap_ok = (not _inside) and _next_onset
    fail = after > 0.4 * ref and after > 0.04 and _voiced(ta, cut - 0.2, cut + 0.8) and not _gap_ok
    return after / ref, fail

def effect_window_beats(tool, exit_len_beats):
    """exit effect actual processing window length for A (beats), consistent with renderer limits. echo/reverb excluded (capturing tail/wet tail is by design)."""
    t = str(tool or "")
    L = int(exit_len_beats or 0)
    if t.startswith("filter_"): return float(min(16, max(2, L or 8)))
    if t == "brake":            return float(max(4, L or 4))
    if t == "backspin":         return float(min(2, max(1, L or 2))) * 2
    return 0.0

def sim_c1b(ta, cut, tool=None, exit_len_beats=8):
    """effect window articulation: exit effect processing window [cut−L, cut−0.3] vocal coverage ratio; fail = ≥0.5."""
    Lb = effect_window_beats(tool, exit_len_beats)
    if Lb <= 0:
        return 0.0, False
    beat = A.beat_of(ta)
    t0, t1 = cut - Lb * beat, cut - 0.3
    if t1 <= t0: return 0.0, False
    cov = sum(max(0.0, min(t1, e) - max(t0, s)) for s, e in A.VOICED.get(ta, []))
    frac = cov / (t1 - t0)
    return frac, frac >= 0.5

def sim_c2(tb, entry):
    if entry <= 1.0:
        return 0.0, False
    eb_before = _v(tb, entry - 0.5, entry)
    eb_ref = _v(tb, entry, entry + 3.0) or 1e-9
    fail = eb_before > 0.5 * eb_ref and eb_before > 0.04 and _voiced(tb, entry - 0.5, entry)
    return eb_before / eb_ref, fail

def sim_c2b(ta, cut, tb, entry, tool=None, overlap_beats=0.0):
    """Ride dual vocals: Seconds where vocal A and vocal B are simultaneously active within the ride/parallel window."""
    t=str(tool or "")
    if t in ("blend16","blendecho","introstack"): ov_beats=16.0
    elif t.startswith("filter_") and (overlap_beats or 0)>0: ov_beats=float(overlap_beats)
    else: return 0.0, False
    ov=ov_beats*A.beat_of(ta)
    va=[(max(s,cut-ov)-(cut-ov), min(e,cut)-(cut-ov)) for s,e in A.VOICED.get(ta,[]) if e>cut-ov and s<cut]
    if t in ("blend16","blendecho","introstack"):
        b0=entry-ov
        vb=[(max(s,b0)-b0, min(e,entry)-b0) for s,e in A.VOICED.get(tb,[]) if e>b0 and s<entry]
    else:
        vb=[(max(s,entry)-entry, min(e,entry+ov)-entry) for s,e in A.VOICED.get(tb,[]) if e>entry and s<entry+ov]
    inter=sum(max(0.0, min(e1,e2)-max(s1,s2)) for s1,e1 in va for s2,e2 in vb)
    return round(inter,2), inter>1.0

def _grid_cands(tid, t0, span, prefer_downbeat=False):
    """Sort beat grid candidates by |t−t0|."""
    r = A.REG[tid]
    bts = [b for b in (r.get("beat_times") or []) if abs(b - t0) <= span]
    dbs = set(round(b, 2) for b in (r.get("downbeat_times") or []))
    bts.sort(key=lambda b: abs(b - t0))
    if prefer_downbeat:
        return [b for b in bts if round(b, 2) in dbs] + [b for b in bts if round(b, 2) not in dbs]
    return bts

def safe_cut_near(ta, cut, tool=None, span=16.0, step=None):
    """Find nearest C1-PASS cut point on A beat grid (old v0.25s scan deprecated: cut point must be beat grid point)."""
    for t in _grid_cands(ta, cut, span):
        if not (5 < t < A.REG[ta]["duration_sec"] - 1):
            continue
        r, fail = sim_c1(ta, t, tool)
        if not fail:
            return round(t, 2), round(r, 2)
    return None

def safe_entry_near(tb, entry, span=16.0, step=None):
    """Find nearest C2-PASS entry point on B beat grid, bar lines prioritized."""
    for t in _grid_cands(tb, entry, span, prefer_downbeat=True):
        if not (0 <= t < A.REG[tb]["duration_sec"] - 10):
            continue
        r, fail = sim_c2(tb, t)
        if not fail:
            return round(t, 2), round(r, 2)
    return None

def snap_downbeat(tid, t, tol_beats=2.0):
    """Snap to the nearest bar line (within tolerance); if no grid, keep original value. Used for section head candidates."""
    dbs = A.REG[tid].get("downbeat_times") or []
    if not dbs: return t
    c = min(dbs, key=lambda b: abs(b - t))
    return round(float(c), 2) if abs(c - t) <= tol_beats * A.beat_of(tid) else t

def sim_plan(ta, tb, plan):
    """Plan must include cut/entry/exit_tool. Return simulation health check dict (all quantities available before rendering)."""
    cut, entry, tool = float(plan["cut"]), float(plan["entry"]), plan.get("exit_tool")
    c1r, c1f = sim_c1(ta, cut, tool)
    c2r, c2f = sim_c2(tb, entry)
    von = next((s for s, e in A.VOICED.get(tb, []) if e > entry + 0.2 and s >= entry - 0.5), None)
    onset = None if von is None else round(max(0.0, von - entry), 1)
    vend = max([e for s, e in A.VOICED.get(ta, []) if e <= cut + 0.2], default=None)
    nxt = min([s for s, e in A.VOICED.get(ta, []) if s > cut - 0.2], default=None)
    gap = None if vend is None else (round(nxt - vend, 1) if nxt is not None else 99.0)
    out = {"C1_prediction": f"{c1r:.2f}x " + ("FAIL" if c1f else "PASS"),
           "C2_prediction": f"{c2r:.2f}x " + ("FAIL" if c2f else "PASS"),
           "B entry to vocal seconds": onset if onset is not None else ">30",
           "A: pause duration at the end of the phrase (seconds)": gap,
           "_c1_fail": bool(c1f), "_c2_fail": bool(c2f)}
    if c1f:
        fix = safe_cut_near(ta, cut, tool)
        if fix: out["C1 fix advice: nearest safe cut"] = fix[0]
    if c2f:
        fix = safe_entry_near(tb, entry)
        if fix: out["C2 Fix advice_nearest safe entry"] = fix[0]
    return out


def b_lead_need_beats(tool, entry_tool=None, echo_delay_beats=1.0, overlap_beats=0.0):
    """Returns (need_beats, why). need = min instrumental gap from B entry point to B first vocal."""
    t, et = str(tool or ""), str(entry_tool or "")
    dly = max(0.5, min(1.5, float(echo_delay_beats or 1.0)))
    need, why = 0.0, ""
    if t == "echo":
        need = 4.8 * dly
        if (overlap_beats or 0) == 0: need = max(0.0, need - 4.0)
        why = "echo tail still lingering"
    elif t == "blendecho":
        need = 4.8 * dly; why = "Echo at the end of phrase A overlaps onto B after the transition"
    elif t == "reverb":
        need = 6.0
        if (overlap_beats or 0) == 0: need = max(0.0, need - 4.0)
        why = "Reverb tail is still ringing"
    elif t.startswith("filter_"):
        need = 2.0; why = "2-beat frequency handshake tail (B enters with extreme frequency then releases)"
    elif t in ("blend16", "introstack"):
        need = 16.0; why = "Parallel window 16 beats B must be instrumental (lead16)"
    if et == "spinup":
        need = max(need, 2.0); why = (why + ";" if why else "") + "Spinup grabs 2 beats before, causing vocal speed change"
    return need, why

def _vocal_gap(tb, t):
    """Seconds from B entry point t to first vocal onset; 0 if within vocal region."""
    for s, e in A.VOICED.get(tb, []):
        if s <= t < e: return 0.0
        if s >= t: return max(0.0, s - t)
    return 99.0

def sim_c2c(ta, cut, tb, entry, tool=None, entry_tool=None, echo_delay_beats=1.0, overlap_beats=0.0):
    """effect tail/footprint suppresses B vocals."""
    need_b, _ = b_lead_need_beats(tool, entry_tool, echo_delay_beats, overlap_beats)
    if need_b <= 0: return 99.0, 0.0, False
    beat = A.beat_of(ta)
    t = str(tool or "")
    if t in ("echo", "blendecho", "reverb"):
        src = 2.0 * beat if t != "reverb" else 8.0 * beat
        if not any(s < cut and e > cut - src for s, e in A.VOICED.get(ta, [])):
            return 99.0, 0.0, False
    need_s = need_b * beat
    gap = _vocal_gap(tb, float(entry))
    return round(gap, 2), round(need_s, 2), gap < need_s - 0.1


import json as _json, os as _os
_EVD = _os.path.join(_os.path.dirname(EV), "outputs")
try:
    _CUEEV = _json.load(open(f"{_EVD}/cue_events.json"))
    _CONS = _json.load(open(f"{_EVD}/cue_consensus.json"))
except Exception:
    _CUEEV, _CONS = {}, {}

def anchors_of(tid):
    """C-lite anchor set (all beat-matched): event anchors + section heads (snapping to bar lines) + consensus beat points. Returns sorted second list."""
    out = set()
    ce = _CUEEV.get(str(tid)) or {}
    for k in ("kick", "bass", "chord_reset", "chord_change"):
        out.update(ce.get(k, []))
    for sg in A.REG[tid].get("segments", []):
        out.add(snap_downbeat(tid, sg["start"]))
    for c in ((_CONS.get(str(tid)) or {}).get("card") or []):
        out.add(c.get("Seconds"))
    return sorted(round(float(x), 2) for x in out if x is not None)

def validate_grid(tid, t, tol=0.08, anchor_span_beats=32):
    """Check that t sits on the beat grid of tid and within span beats of an anchor.
    Return dict: ok/nearest beat/deviation seconds/nearest anchor/anchor offset beats (for feedback and LLM regeneration)."""
    t = float(t)
    beat = A.beat_of(tid)
    bts = A.REG[tid].get("beat_times") or []
    if not bts:
        return {"ok": True, "note": "no grid"}
    import bisect
    j = min(range(max(0, bisect.bisect(bts, t) - 2), min(len(bts), bisect.bisect(bts, t) + 2)),
            key=lambda i: abs(bts[i] - t))
    near = float(bts[j]); off = t - near
    ancs = anchors_of(tid)
    na = min(ancs, key=lambda a: abs(a - t)) if ancs else None
    ka = round((t - na) / beat) if na is not None else None
    ok = abs(off) <= tol and (na is None or abs(t - na) <= anchor_span_beats * beat)
    return {"ok": bool(ok), "nearest beat": round(near, 2), "off by s": round(off, 3),
            "nearest anchor": na, "anchor offset beats": ka}


import numpy as _np, librosa as _lr
def _mix_rms(tid, t0, t1):
    """Full mix RMS—must use same audio source as renderer (A.audio=loudness normalization pool; REG['audio_path'] is
    original file; different loudness skews ratios: DA NOW case 0.34 vs actual 0.17)."""
    try:
        y = A.audio(tid)
        a, b = int(max(0.0, t0) * A.SR), int(max(0.1, t1) * A.SR)
        if b <= a: return None
        seg = y[a:b]
        return float(_np.sqrt((_np.asarray(seg, dtype=float) ** 2).mean() + 1e-12))
    except Exception:
        return None

def material_window_beats(tool, exit_len_beats):
    """effect source window is long (beats): effects need material to process. echo only trails the final syllable ≤2; brake ≥4; backspin = 2×len."""
    t = str(tool or ""); L = int(exit_len_beats or 0)
    if t == "brake": return float(max(4, L or 4))
    if t == "backspin": return float(min(2, max(1, L or 2))) * 2
    if t == "echo": return float(min(L or 2, 2))
    return 0.0

def c1c_min_ratio():
    import os
    try:
        return float(os.environ.get("AIDJ_C1C_MIN_RATIO", 0.35))
    except (TypeError, ValueError):
        return 0.35


def sim_c1c(ta, cut, tool=None, exit_len_beats=4):
    """Effect material gap: sample window RMS < 0.35× previous 20s reference = effectively silent (mirror of C1b: that track
    Returns (ratio, fail): C1b covers vocals that should not be in the window,
    this one covers the energy that should be. """
    W = material_window_beats(tool, exit_len_beats) * A.beat_of(ta)
    if W <= 0: return 1.0, False
    ref = _mix_rms(ta, cut - 20, cut - W)
    w = _mix_rms(ta, cut - W, cut)
    if not ref or ref < 0.01 or w is None: return 1.0, False
    return round(w / ref, 2), (w / ref) < c1c_min_ratio()

def sim_c6b(ta, cut, tb, entry, entry_tool=None, overlap_beats=0.0, exit_tool=None, exit_len_beats=4):
    """Entry energy (direct only): B head 4 beats RMS < 0.25 × A exit reference = seam energy cliff.
    Reference = A normal state energy, must subtract effect sampling window (otherwise quiet materials lower the denominator, DA NOW case misses).
    filter_in/spinup exempt (surfacing from bottom/accelerating entry is by design). Return (ratio, fail). """
    et = str(entry_tool or "").strip("_")
    if et not in ("", "direct") or (overlap_beats or 0) > 0: return 1.0, False
    W = max(material_window_beats(exit_tool, exit_len_beats), 4.0) * A.beat_of(ta)
    ref = _mix_rms(ta, cut - 20, cut - W)
    bh = _mix_rms(tb, entry, entry + 4 * A.beat_of(tb))
    if not ref or ref < 0.01 or bh is None: return 1.0, False
    return round(bh / ref, 2), (bh / ref) < 0.25


_ATAB = {}
def anchor_table(tid):
    """Numbered anchor table (all on beat), including personality tags: exit anchors (phrase end/consensus)/entry anchors (section start/drums bass)/harmonic anchors."""
    k = str(tid)
    if k in _ATAB: return _ATAB[k]
    items = []
    ce = _CUEEV.get(k) or {}
    for t in ce.get("kick", []): items.append((t, "entry anchors", "drums enter"))
    for t in ce.get("bass", []): items.append((t, "entry anchors", "bass enters"))
    for t in ce.get("chord_reset", []): items.append((t, "harmonic anchors", "harmonic loop restart"))
    for t in ce.get("chord_change", []): items.append((t, "harmonic anchors", "harmonic change"))
    for sg in A.REG[tid].get("segments", []):
        if sg["label"] in ("start", "end", "silence"): continue
        items.append((snap_downbeat(tid, sg["start"]), "entry anchors", sg["label"] + "Head"))
    for c in ((_CONS.get(k) or {}).get("card") or []):
        if c.get("Seconds") is not None:
            items.append((c["Seconds"], "exit anchors", "Consensus point(" + "+".join(c.get("Agreement layer", [])) + ")"))
    for sg in A.REG[tid].get("segments", []):
        if sg["label"] in ("start", "end", "silence"): continue
        items.append((snap_downbeat(tid, sg["end"]), "exit anchors", sg["label"] + "End"))
    best = {}
    pri = {"exit anchors": 0, "entry anchors": 1, "harmonic anchors": 2}
    for t, typ, desc in items:
        t = round(float(t), 2)
        if t not in best or pri[typ] < pri[best[t][0]]:
            best[t] = (typ, desc)
    segs = A.REG[tid]["segments"]; voiced = A.VOICED.get(tid, []); beat = A.beat_of(tid); win = 16 * beat
    def _vc(a, b): return sum(max(0, min(b, e) - max(a, s)) for s, e in voiced if e > a and s < b) / max(1e-6, b - a)
    real = [s for s in segs if s["label"] not in ("start", "end", "silence")]
    rows = []
    for i, (t, v) in enumerate(sorted(best.items())):
        typ, ev = v
        seg = next((s for s in segs if s["start"] <= t < s["end"]), None)
        lab = seg["label"] if seg else "?"
        ap = []
        es = next((s for s in real if abs(s["end"] - t) <= beat), None)
        if es:
            ap.append("cut entrance")
            if es["label"] == "chorus": ap.append("blend exit")
        hs = next((s for s in real if abs(s["start"] - t) <= beat), None)
        if hs:
            ap.append("cut entry")
            if hs["label"] in ("verse", "chorus", "hook", "drop") and t - win >= 0 and _vc(t - win, t) < 0.15:
                ap.append("blend entry")
        if ("Drums" in str(ev) or "bass" in str(ev)) and "cut entry" not in ap:
            ap.append("cut entry")
        rows.append({"id": f"a{i+1}", "Seconds": t, "Type": typ, "events": ev, "Segment": lab, "applies to": ap})
    _ATAB[k] = rows
    return rows

def resolve_cue(tid, v, max_off=32):
    """C-full analysis: v is {"anchor":"aN","offset_beats":k} → traverse k steps along real beat grid;
    v is number → return as-is (legacy C-lite path, subsequent grid validation). Return (seconds or None, err). """
    if isinstance(v, (int, float)): return float(v), None
    if isinstance(v, str):
        try: return float(v), None
        except ValueError: pass
        v = {"anchor": v, "offset_beats": 0}
    if not isinstance(v, dict): return None, f"Unknown cue format:{v!r}"
    aid = str(v.get("anchor", "")).strip()
    try: k = int(round(float(v.get("offset_beats", 0) or 0)))
    except Exception: return None, f"offset_beats must be integer:{v.get('offset_beats')!r}"
    if abs(k) > max_off: return None, f"offset_beats={k} exceeds ±{max_off}"
    tab = anchor_table(tid)
    import re as _re
    m = _re.search(r"(\d+)", aid)
    hit = next((r for r in tab if r["id"] == aid), None)
    if hit is None and m:
        hit = next((r for r in tab if r["id"] == f"a{m.group(1)}"), None)
    if hit is None:
        return None, f"Anchor {aid!r} does not exist (use actual ID from anchor table, e.g., a7; valid: a1~a{len(tab)})"
    bts = A.REG[tid].get("beat_times") or []
    if not bts:
        return round(hit["Seconds"] + k * A.beat_of(tid), 2), None
    import bisect
    j = min(range(max(0, bisect.bisect(bts, hit["Seconds"]) - 2),
                  min(len(bts), bisect.bisect(bts, hit["Seconds"]) + 2)),
            key=lambda i: abs(bts[i] - hit["Seconds"]))
    j2 = max(0, min(len(bts) - 1, j + k))
    return round(float(bts[j2]), 2), None


def _lead_clean_beats(tb, e, beats=16):
    """Number of clean instrumental beats before B entry point."""
    beat = A.beat_of(tb)
    t0 = e - beats * beat
    if t0 < 0: return 0.0
    occ = sum(max(0.0, min(e - 0.3, e2) - max(t0, s2)) for s2, e2 in A.VOICED.get(tb, []))
    return max(0.0, beats - occ / beat)

def blend_table(ta, tb, overlaps=(8, 16, 32), max_rows=8):
    """Enumerate feasible blend pairs from (Aexit anchors × Bentry anchors × overlap duration), only entering table if passing all sim gates.
    An empty table means no blend is possible for this pair, so the planner
    falls back to the cut family. Each row names the moves it allows. """
    rows = []
    beat_a = A.beat_of(ta)
    ta_dur, tb_dur = A.REG[ta]["duration_sec"], A.REG[tb]["duration_sec"]
    a_anchors = [r for r in anchor_table(ta) if r["Seconds"] > ta_dur * 0.35]
    b_anchors = [r for r in anchor_table(tb) if r["Seconds"] < tb_dur * 0.55 and tb_dur - r["Seconds"] >= 45]
    for ov in overlaps:
        for ra in a_anchors:
            cut = ra["Seconds"]
            w0 = cut - ov * beat_a
            a_vocal = sum(max(0.0, min(cut - 0.3, e) - max(w0, s)) for s, e in A.VOICED.get(ta, []))
            a_clean = a_vocal < 0.15 * ov * beat_a
            for rb in b_anchors:
                entry = rb["Seconds"]
                lead = _lead_clean_beats(tb, entry, ov)
                if lead < ov * 0.75: continue
                tool = "blend16" if a_clean else "blendecho"
                sec, f2b = sim_c2b(ta, cut, tb, entry, tool, ov)
                if f2b: continue
                _, _, f2c = sim_c2c(ta, cut, tb, entry, tool, None, 1.0, ov)
                if f2c: continue
                r2, f2 = sim_c2(tb, entry)
                if f2: continue
                seg_lab = next((sg["label"] for sg in A.REG[ta].get("segments", [])
                                if sg["start"] <= cut <= sg["end"]), "?")
                a_pri = {"bridge": 0, "inst": 0, "interlude": 0, "break": 0,
                         "outro": 1}.get(seg_lab, 2 if not a_clean else 3)
                rows.append({"a_anchor": ra["id"], "a_seconds": cut, "a_events": ra["events"], "a_section": seg_lab,
                             "b_anchor": rb["id"], "b_seconds": entry, "b_events": rb["events"],
                             "overlap beats": ov, "Type": tool, "clean beats on B": round(lead, 1),
                             "Side A": "Instrumental window" if a_clean else "Vocal riding (blendecho)"})
                rows[-1]["_pri"] = a_pri
    rows.sort(key=lambda r: (r.pop("_pri", 3), abs(r["overlap beats"] - 16), -r["clean beats on B"]))
    return rows[:max_rows]


def sim_c9(ta, cut):
    """A entry runway."""
    dur = A.REG[ta]["duration_sec"]
    lim = max(45.0, 0.2 * dur)
    if cut < lim:
        return f"A hands off after only {cut:.0f}s (requires ≥{lim:.0f}s)", True
    for sg in A.REG[ta].get("segments", []):
        if sg["label"] in ("intro", "start") and sg["start"] <= cut <= sg["end"]:
            return f"cue_out lands in the {sg['label']} section of A", True
    return None, False


try:
    _SEGENV2 = _json.load(open(f"{_EVD}/seg_env.json"))
except Exception:
    _SEGENV2 = {}

def blend_window_card(ta, cut, tb, entry, ov_beats=16):
    """For the given intent blend (A cut × B entry × overlap long), return 4D details: harmony/energy/rhythm conversion/low frequency."""
    beat = A.beat_of(ta)
    w0, w1 = cut - ov_beats * beat, cut
    b0, b1 = entry - ov_beats * beat, entry
    card = {"Overlap Window": f"A[{w0:.1f}~{cut:.1f}] × B[{max(0,b0):.1f}~{entry:.1f}], {ov_beats} beats"}
    ceA = _CUEEV.get(str(ta)) or {}; ceB = _CUEEV.get(str(tb)) or {}
    chgA = [t for t in ceA.get("chord_change", []) if w0 < t < w1]
    resB = [t for t in ceB.get("chord_reset", []) if b0 - 8*beat < t < entry]
    chgB = [t for t in ceB.get("chord_change", []) if b0 < t < b1]
    card["Harmony"] = {"A: harmonic change within the window": chgA or "none (stable)",
                  "Is B in harmonic loop section": ("Yes (loop restart@" + ",".join(f"{t:.1f}" for t in resB[:2]) + ")") if resB else "No loop marker found",
                  "B window has harmonic change": chgB or "none (stable)"}
    def seg_env_of(tid, t):
        se = (_SEGENV2.get(str(tid)) or {}).get("segs", [])
        segs = A.REG[tid].get("segments", [])
        for i, sg in enumerate(segs):
            if sg["start"] <= t <= sg["end"]:
                ev = next((x for x in se if x.get("i") == i), None)
                return {"Segment": sg["label"], "envelope": ev.get("env") if ev else None, "trend": ev.get("trend") if ev else None}
        return None
    card["Energy"] = {"A: overlap window segment": seg_env_of(ta, (w0+w1)/2), "B overlap window section": seg_env_of(tb, max(0,(b0+b1)/2)),
                  "advice": "A: decreasing + B: increasing = smooth; A: still high energy while B: weak = risk of a gap"}
    ra, rb = A.REG[ta]["bpm"], A.REG[tb]["bpm"]
    r = rb / ra
    fold = min((abs(r-1), "1:1"), (abs(r/2-1), "B is double speed (converted)"), (abs(r*2-1), "B is half speed (converted)"), key=lambda x: x[0])
    card["Rhythm"] = {"BPM": f"A {ra:.0f} / B {rb:.0f}", "Relationship": fold[1],
                  "advice": "When not 1:1, extend long section by 8 beats or change introstack (LPF to filter rhythmic conflicts)" if fold[1] != "1:1" else "Same tempo, 16 beats safe"}
    bassB = sorted(ceB.get("bass", []))
    nb = next((t for t in bassB if t >= b0), None)
    card["Low frequency"] = {"B_bass enters at": (f"{nb:.1f}s(" + ("Overlap Window ⚠ Simultaneous dual bass; rendering has no bass-swap" if nb < entry else "After cue_in, ideal") + ")") if nb is not None else "No bass marker found",
                  "advice": "Prioritize overlapping windows where the B bass has not yet entered"}
    sec, f2b = sim_c2b(ta, cut, tb, entry, "blend16", ov_beats)
    card["Vocal re-verification"] = f"Overlap Window Dual Vocal Intersection {sec}s" + ("(⚠ exceeds limit)" if f2b else "(OK)")
    return card
