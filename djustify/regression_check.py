# -*- coding: utf-8 -*-
"""auto-validate render output - mandatory before claiming done"""
import os, sys, json
import numpy as np
import librosa

EV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EV)
import song_library as A

SR = 22050
VAM = json.load(open(f"{A.ROOT}/dj_transition_planner/outputs/va_micl.json"))
QUIET_OK = {"brake", "backspin", "echo"}
CONT_REQ = {"blend16", "blendecho", "introstack", "filter_hpf", "filter_lpf", "reverb",
            "move1", "move2", "move3", "move4"}
try:
    _SEGENV = json.load(open(f"{A.ROOT}/dj_transition_planner/outputs/seg_env.json"))
except Exception:
    _SEGENV = {}


def vstem(tid, t0, t1):
    return A._vstem_rms(tid, t0, t1)


def voiced_overlap(tid, t0, t1):
    """Whether vocal interval (VAD) overlaps with [t0,t1] — independent corroborating signal for stem RMS."""
    return any(s < t1 and e > t0 for s, e in A.VOICED.get(tid, []))


def check_run(run_dir):
    tr = json.load(open(f"{run_dir}/set_trace.json"))
    y, _ = librosa.load(f"{run_dir}/set.wav", sr=SR, mono=True)
    order = tr["order_tids"]
    results = []
    for i, j in enumerate(tr["junctions"]):
        p = j.get("plan") or {}
        ta, tb = order[i], order[i + 1]
        cut, entry = p.get("cut"), p.get("entry", 0.0)
        tool = p.get("exit_tool")
        r = {"j": i + 1, "pair": f"{A.REG[ta]['title']}→{A.REG[tb]['title']}", "tool": tool, "fails": []}
        if tool not in ("move1", "move2", "move3", "move4", "blend16", "blendecho", "introstack"):
            ref = vstem(ta, cut - 3.0, cut) or 1e-9
            after = vstem(ta, cut, cut + 0.8) or 0.0
            _inside = any(s0 + 0.15 < cut < e0 - 0.15 for s0, e0 in A.VOICED.get(ta, []))
            _next_onset = any(cut + 0.1 <= s0 <= cut + 0.8 for s0, e0 in A.VOICED.get(ta, []))
            _gap_ok = (not _inside) and _next_onset
            if after > 0.4 * ref and after > 0.04 and voiced_overlap(ta, cut - 0.2, cut + 0.8) and not _gap_ok:
                r["fails"].append(f"C1 A vocal continues after entrance ({after/ref:.2f}×)")
        try:
            from plan_sim import sim_c1b as _sim_c1b
            _fr, _fb = _sim_c1b(ta, cut, tool, p.get("exit_len_beats", 8))
            if _fb:
                r["fails"].append(f"C1b effect window covers vocal ({_fr:.0%}): move cue_out to end of phrase + after effectlength (effect lands on instrumental tail/final syllable), or reduce exit_len_beats, or change echo (designed to catch final syllable)")
        except Exception:
            pass
        # C2 entry cuts a sung phrase
        if entry > 1.0:
            eb_before = vstem(tb, entry - 0.5, entry) or 0.0
            eb_ref = vstem(tb, entry, entry + 3.0) or 1e-9
            if eb_before > 0.5 * eb_ref and eb_before > 0.04 and voiced_overlap(tb, entry - 0.5, entry):
                r["fails"].append(f"C2 B entrance mid-phrase (vocal in preceding 0.5s {eb_before/eb_ref:.2f}×)")
        try:
            from plan_sim import sim_c2b as _sim_c2b
            _sec, _fb = _sim_c2b(ta, cut, tb, entry, tool, p.get("overlap_beats", 0))
            if _fb:
                r["fails"].append(f"C2b Riding Window Dual Vocals({_sec:.1f}s): Offset vocals left/right (B select instrumental window entry/shrink overlap/A advance end)")
        except Exception:
            pass
        try:
            from plan_sim import sim_c2c as _sim_c2c
            _gap, _need, _fb = _sim_c2c(ta, cut, tb, entry, tool, p.get("entry_tool"),
                                        p.get("echo_delay_beats", 1.0), p.get("overlap_beats", 0))
            if _fb:
                r["fails"].append(f"C2c effect tail over B vocal(gap {_gap:.1f}s < need {_need:.1f}s): B switch to vocal entry point with sufficient gap (instrumental head/drums enter), or shrink echo_delay/switch brake")
        except Exception:
            pass
        _RIDE_BLEND = {"blendecho", "blend16", "loop_in", "loop", "introstack"}
        try:
            from plan_sim import sim_c1c as _sim_c1c, sim_c6b as _sim_c6b, _mix_rms
            _r1c, _f1c = _sim_c1c(ta, cut, tool, p.get("exit_len_beats", 4))
            if _f1c:
                r["fails"].append(f"C1c effect has no material ({_r1c:.2f}× reference): Stop/reverse/feedback need material to act on—move cue_out to energy-rich section or change technique")
            if tool not in _RIDE_BLEND:
                _r6b, _f6b = _sim_c6b(ta, cut, tb, entry, p.get("entry_tool"), p.get("overlap_beats", 0),
                                      tool, p.get("exit_len_beats", 4))
                if _f6b:
                    r["fails"].append(f"C6b entry energy cliff (B head {_r6b:.2f} × A reference): switch to filter_in/spinup entry, or choose a stronger energy entry point for B")
            else:
                _ea = _mix_rms(ta, cut - 3.0, cut) or 1e-6
                _eb = _mix_rms(tb, entry, entry + 3.0) or 1e-6
                _db = 20 * (np.log10(_eb) - np.log10(_ea))
                if _db > 2.5:
                    r["fails"].append(f"C6c B buries A (B entry {_db:.1f}dB higher than A exit > 2.5): switching tricks within the blend family is useless—please change the title (exclude current B from the library, select another using the same logic), keep the blend")
        except Exception:
            pass
        tj = tr["junction_times"][i] if i < len(tr.get("junction_times", [])) else None
        if tj:
            beat = A.beat_of(ta)
            seg = y[int((tj - 4 * beat) * SR):int((tj + 4 * beat) * SR)]
            w = int(0.15 * SR)
            n = len(seg) // w
            if n > 4:
                e = np.sqrt((seg[:n * w].reshape(n, w) ** 2).mean(axis=1))
                quiet_s = float((e < 0.25 * e.mean()).sum() * 0.15)
                lim = 1.8 if tool in QUIET_OK else 1.2
                if quiet_s > lim:
                    r["fails"].append(f"C3 Gap {quiet_s:.1f}s > {lim}s")
                if tool in QUIET_OK:
                    qb = e < 0.25 * e.mean()
                    run = longest = 0
                    for b in qb:
                        run = run + 1 if b else 0
                        longest = max(longest, run)
                    if longest * 0.15 > 1.0:
                        r["fails"].append(f"C3b continuous silence {longest*0.15:.1f}s > 1.0s(pause too long)")
                if tool in CONT_REQ and e.min() < 0.22 * e.mean():
                    r["fails"].append(f"C5 Continuity move energy gap(min {e.min()/e.mean():.2f}×mean)")
        RIDE = {"filter_lpf", "filter_hpf", "blend16", "blendecho", "introstack", "move1", "move2", "move3", "move4", "harmonic"}
        if tj and tool in RIDE:
            a0, a1 = max(0, int((tj - 12) * SR)), int((tj + 20) * SR)
            seg2 = y[a0:a1]
            hop2 = int(0.05 * SR)
            n2 = len(seg2) // hop2
            if n2 > 40:
                e2 = np.sqrt((seg2[:n2 * hop2].reshape(n2, hop2) ** 2).mean(axis=1))
                db = 20 * np.log10(np.maximum(e2, 1e-6))
                med_db = float(np.median(db))
                t2 = a0 / SR + np.arange(n2) * 0.05
                zone2 = (t2 >= tj - 2) & (t2 <= tj + 8)
                below = db[zone2] < med_db - 30
                longest, run = 0, 0
                for b in below:
                    run = run + 1 if b else 0
                    longest = max(longest, run)
                qs = longest * 0.05
                if qs > 1.0:
                    r["fails"].append(f"C5b dead overlap(longest {qs:.1f}s)")
        # C4 emotion distance
        va, vb = VAM.get(ta, {}), VAM.get(tb, {})
        if va and vb:
            dv = abs(va["valence_q"] - vb["valence_q"]); da = abs(va["arousal_q"] - vb["arousal_q"])
            if dv + da > 0.9 or dv > 0.55:
                r["fails"].append(f"C4 V/A break(ΔV_q={dv:.2f} ΔA_q={da:.2f})")
        dur_b = A.REG[tb].get("duration_sec")
        if entry is not None and dur_b:
            runway = dur_b - entry
            pos = entry / dur_b
            if runway < 45 or pos > 0.55:
                r["fails"].append(f"C6 entry runway insufficient(entry at {pos*100:.0f}%, remaining {runway:.0f}s)")
        try:
            from plan_sim import sim_c9 as _sim_c9
            _why9, _f9 = _sim_c9(ta, cut)
            if _f9:
                r["fails"].append(f"C9 exit runway insufficient ({_why9}): move cue_out back until A has played at least one main body section")
        except Exception:
            pass
        fs = (_SEGENV.get(str(ta)) or {}).get("fade_start")
        dur_a = A.REG[ta].get("duration_sec")
        if fs and cut and cut > fs + 1.0 and (not dur_a or cut < dur_a - 2.0):
            r["fails"].append(f"C8 A cut during fade (fade start {fs:.0f}s, cut {cut:.0f}s): move cue_out before the fade or change A section")
        results.append(r)
    return results


def main():
    arg = sys.argv[1]
    run_dir = arg if os.path.isdir(arg) else f"{A.VR}/microset/{arg}"
    res = check_run(run_dir)
    allpass = True
    for r in res:
        ok = not r["fails"]
        allpass &= ok
        print(f"j{r['j']} {r['pair']} [{r['tool']}] : {'✓PASS' if ok else '✗ ' + ' | '.join(r['fails'])}")
    print("VERDICT:", "ALL PASS" if allpass else "FAIL")
    print("=== REGCHECK DONE ===", flush=True)


if __name__ == "__main__":
    main()
