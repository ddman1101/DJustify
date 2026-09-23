# -*- coding: utf-8 -*-
"""Micro-set v1: four songs linked as one. Sort (energy grammar + V/A, LLM selects with reasoning) → run agent at each of the three seams"""
import os, json, re, argparse, time, itertools
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as ss
import requests

import os as _os
BLEND_B_RIDE = float(_os.environ.get("BLEND_B_RIDE", "0.7"))
BLEND_SNAP_BEATS = float(_os.environ.get("BLEND_SNAP_BEATS", "2"))
BLEND_XF_BEATS = float(_os.environ.get("BLEND_XF_BEATS", "1"))
def blend_bed_env(L, beat, SR, a_seg, b_seg, snap_beats=None):
    """A stays full; B is compressed within the blend window to BLEND_B_RIDE, maxed out ~2 beats before handoff (to avoid click)."""
    import numpy as _np
    a_env = _np.ones(L)
    b_env = _np.full(L, BLEND_B_RIDE)
    ne = int(min(L, max(1, 2 * beat * SR)))
    if ne > 0:
        b_env[-ne:] = _np.linspace(BLEND_B_RIDE, 1.0, ne)
    return a_env, b_env, 1.0

import song_library as A

SR = A.SR
FREE_MODE = False
_VST = {}
OUT_ROOT = f"{A.VR}/microset"
os.makedirs(OUT_ROOT, exist_ok=True)
DEFAULT = ["Sun Never Sets", "Prodigal son returns", "Cultivate love", "What I Miss"]


def score_order(feats):
    s = 0.0
    for a, b in zip(feats, feats[1:]):
        s += abs(np.log2(max(b["chorus_E"], 1e-2) / max(a["chorus_E"], 1e-2)))
        s += 0.6 * (abs((b["V"] or .5) - (a["V"] or .5)) + abs((b["A"] or .5) - (a["A"] or .5)))
        rb = (b.get("bpm") or 120) / (a.get("bpm") or 120)
        while rb > 1.5: rb /= 2
        while rb < 0.75: rb *= 2
        gap = abs(np.log2(rb))
        s += 3.0 if gap > 0.12 else gap * 2.0
    return s


def _vst_fx(tool):
    """Load and cache VST plugins (requires DISPLAY; automatically starts Xvfb :99 when no X is available)."""
    os.environ.setdefault("DISPLAY", ":99")
    import subprocess
    if subprocess.run(["pgrep", "-x", "Xvfb"], capture_output=True).returncode != 0:
        subprocess.Popen(["Xvfb", ":99", "-screen", "0", "640x480x24"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
    if tool in _VST: return _VST[tool]
    from pedalboard import load_plugin
    VD = os.path.join(os.environ.get("AIDJ_RUNTIME_ROOT", "runtime"), "vst")
    if tool == "echo_vst":
        p = load_plugin(f"{VD}/zam-plugins-4.5/ZamDelay.vst3")
    else:
        p = load_plugin(f"{VD}/dragonfly-reverb-3.2.10/DragonflyPlateReverb.vst3")
    _VST[tool] = p
    return p


def apply_exit_tail(yA, cut, tool, beat, len_beats, echo_delay_beats=1.0,
                    echo_feedback=0.62, echo_lo_hz=250, echo_hi_hz=3800, echo_reps=12):
    """Returns (last len sec processed before A entry, wet tail after cut)."""
    if tool == "reverb":
        try:
            _vst_fx("reverb_vst"); tool = "reverb_vst"
        except Exception:
            pass
    if tool == "brake":
        len_beats = max(4, len_beats)
    if tool == "echo":
        len_beats = min(len_beats, 2)
    nb = int(max(1, len_beats) * beat * SR)
    i_cut = int(cut * SR)
    head = yA[i_cut - nb:i_cut].copy()
    tail = np.zeros(int(8 * beat * SR))
    if tool == "echo":
        _lo = float(min(max(echo_lo_hz, 80), 1000)); _hi = float(min(max(echo_hi_hz, 1500), 8000))
        bh, ah = ss.butter(2, _lo/(SR/2), "high"); bl, al = ss.butter(2, _hi/(SR/2), "low")
        _fb = float(min(max(echo_feedback, 0.2), 0.8)); _reps = int(min(max(echo_reps, 3), 12))
        D = int(0.5 * beat * SR)
        N2 = len(head) + len(tail)
        send = np.concatenate([head, np.zeros(len(tail))])
        wet = np.zeros(N2 + D)
        for _ in range(_reps):
            d2 = np.zeros(N2 + D)
            d2[D:D+N2] = ss.lfilter(bl, al, ss.lfilter(bh, ah, send + _fb * wet[:N2]))
            wet = d2
        head = head + wet[:len(head)]
        tail = wet[len(head):len(head)+len(tail)]
    elif tool == "brake":
        seg = yA[i_cut - nb:i_cut + int(0.5*SR)]
        sp = np.linspace(1, 0, nb) ** 1.8; ph = np.cumsum(sp)
        head = np.interp(ph, np.arange(len(seg)), seg) * np.linspace(1, .5, nb)
    elif tool == "backspin":
        src = yA[i_cut - 2*nb:i_cut][::-1]
        sp = np.linspace(1, 3.5, nb); ph = np.cumsum(sp); ph = ph/ph[-1]*(len(src)-1)
        bl, al = ss.butter(2, 5000/(SR/2), "low")
        head = ss.lfilter(bl, al, np.interp(ph, np.arange(len(src)), src)) * np.linspace(1, .45, nb)
    elif tool == "reverb":
        from pedalboard import Pedalboard, Reverb
        rv = Pedalboard([Reverb(room_size=0.92, damping=0.35, wet_level=1.0, dry_level=0.0)])
        send = np.concatenate([head, np.zeros(len(tail))]).astype("float32")
        wet = rv(send[None, :], SR)[0]
        ntail = len(tail)
        env = np.linspace(1, 0, ntail) ** 0.7
        head = head + 1.45 * wet[:len(head)]
        tail = 1.45 * wet[len(head):len(head) + ntail] * env
    elif tool in ("echo_vst", "reverb_vst"):
        p = _vst_fx(tool)
        try:
            if tool == "echo_vst":
                D = 0.5 * beat * 1000.0
                for k, v in [("sync_bpm", False), ("delaytime_ms", D), ("time_ms", D),
                             ("lpf_hz", 3800.0), ("feedback", 0.62), ("dry_wet", 1.0),
                             ("invert", False), ("output_gain_db", 0.0)]:
                    try: setattr(p, k, v)
                    except Exception: pass
            else:
                for k, v in [("dry_level", 0.0), ("wet_level", 100.0), ("decay_s", 2.6),
                             ("dampen_hz", 6000.0), ("low_cut_hz", 200.0), ("predelay_ms", 10.0)]:
                    try: setattr(p, k, v)
                    except Exception: pass
        except Exception:
            pass
        send = np.concatenate([head, np.zeros(len(tail))]).astype("float32")
        sig = send[None, :] if tool == "echo_vst" else np.stack([send, send])
        w2 = p(sig, SR)
        wet = w2[0] if w2.shape[0] == 1 else w2.mean(axis=0)
        g = 1.0 if tool == "echo_vst" else 1.2
        env = np.linspace(1, 0, len(tail)) ** 0.7
        head = head + g * wet[:len(head)]
        tail = g * wet[len(head):len(head) + len(tail)] * env
    elif tool in ("filter_lpf", "filter_hpf"):
        from pedalboard import Pedalboard, LadderFilter
        fl = int(min(16, max(2, len_beats or 8)))
        w = int(fl * beat * SR)
        seg = yA[i_cut - w:i_cut].copy()
        if tool == "filter_lpf":
            lf = LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=16000, resonance=0.35)
            f0, f1 = 16000, 120
        else:
            lf = LadderFilter(mode=LadderFilter.Mode.HPF24, cutoff_hz=30, resonance=0.35)
            f0, f1 = 30, 8000
        board = Pedalboard([lf]); nbk = 24
        for kk in range(nbk):
            s2, e2 = kk*len(seg)//nbk, (kk+1)*len(seg)//nbk
            lf.cutoff_hz = float(f0 * (f1/f0) ** (kk/(nbk-1)))
            seg[s2:e2] = board(seg[None, s2:e2].astype("float32"), SR, reset=False)[0]
        ov = int(min(2, fl / 2) * beat * SR)
        head2 = seg[:-ov]
        tail = seg[-ov:] * (np.cos(np.linspace(0, np.pi/2, ov)) ** 2)
        return head2, tail, w
    return head, tail, nb


def _match_gain_db(ta, cut, tb, entry):
    """Loudness match at the handover, in dB."""
    if os.environ.get("LOUD_MATCH", "1") == "0":
        return 0.0
    try:
        w = int(0.5 * SR)
        def body_rms(y):
            n = len(y) // w
            if n < 4:
                return float(np.sqrt((y ** 2).mean())) + 1e-9
            e = np.sqrt((y[:n * w].reshape(n, w) ** 2).mean(axis=1))
            e = e[e > e.max() * 0.05]
            return (float(np.percentile(e, 75)) if len(e) else 1e-9) + 1e-9
        ra = body_rms(A.audio(ta)); rb = body_rms(A.audio(tb))
        g = 20 * np.log10(ra / rb)
        return 0.0 if abs(g) <= 1.5 else float(np.clip(g, -2.0, 2.0))
    except Exception:
        return 0.0

def make_entry_stream(tidB, entry, tool, keysync, beatA, dur, need_sync=True):
    """Returns (seg, sync_delta): B entry first beatmatch aligns to A beat, 8 bars gradually returns to original speed (DJ loosens pitch fader)."""
    yB = A.audio(tidB)
    seg = yB[int(entry * SR):int((entry + dur) * SR)].copy()
    sync_delta = 0.0
    beatB = A.beat_of(tidB)
    rt = beatB / beatA
    while rt > 1.5: rt /= 2
    while rt < 0.75: rt *= 2
    if need_sync and tool != "spinup" and 0.005 < abs(np.log2(rt)) <= 0.05:
        ramp_out = 8 * 4 * beatA
        src_take = int(ramp_out * rt * SR)
        head_out = A._rb_stretch(seg[:src_take], float(rt))
        sync_delta = src_take / SR - len(head_out) / SR
        seg = np.concatenate([head_out, seg[src_take:]])
    if keysync:
        seg = A._rb_pitch(seg, keysync)
    if tool == "spinup":
        beatB = A.beat_of(tidB)
        nb = int(4 * beatB * SR)
        sp = 0.03 + 0.97 * np.linspace(0, 1, nb) ** 1.5
        ph = np.cumsum(sp); consumed = int(ph[-1])
        src0 = max(0, int(entry * SR) - consumed)
        seg2 = yB[src0:int((entry + dur) * SR)]
        if keysync: seg2 = A._rb_pitch(seg2, keysync)
        o = np.interp(ph, np.arange(len(seg2)), seg2) * np.linspace(0.35, 1.0, nb)
        seg = np.concatenate([o, seg2[consumed:]])
    elif tool == "filter_in_hpf":
        _has_v = (not FREE_MODE) and any(s0 < entry + 4 * beatA and e > entry for s0, e in A.VOICED.get(tidB, []))
        if _has_v:
            nop = min(int(2 * beatA * SR), len(seg))
            seg[:nop] *= (np.sin(np.linspace(0, np.pi / 2, nop)) ** 2)
            return seg, sync_delta
        from pedalboard import Pedalboard, LadderFilter
        nop = min(int(8 * beatA * SR), len(seg))
        lf = LadderFilter(mode=LadderFilter.Mode.HPF24, cutoff_hz=8000, resonance=0.3)
        board = Pedalboard([lf]); nbk = 24
        head = seg[:nop].copy()
        for kk in range(nbk):
            s2, e2 = kk*nop//nbk, (kk+1)*nop//nbk
            head[s2:e2] = board(head[None, s2:e2].astype("float32"), SR, reset=False)[0]
            lf.cutoff_hz = float(8000 * (30/8000) ** (kk/(nbk-1)))
        seg[:nop] = head * (0.4 + 0.6 * np.sin(np.linspace(0, np.pi/2, nop)) ** 2)
        return seg, sync_delta
    elif tool == "filter_in":
        _has_v = (not FREE_MODE) and any(s0 < entry + 4 * beatA and e > entry for s0, e in A.VOICED.get(tidB, []))
        if _has_v:
            nop = min(int(2 * beatA * SR), len(seg))
            seg[:nop] *= (np.sin(np.linspace(0, np.pi / 2, nop)) ** 2)
            return seg, sync_delta
        from pedalboard import Pedalboard, LadderFilter
        nop = min(int(8 * beatA * SR), len(seg))
        lf = LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=120, resonance=0.3)
        board = Pedalboard([lf]); nbk = 24
        head = seg[:nop].copy()
        for kk in range(nbk):
            s, e = kk*nop//nbk, (kk+1)*nop//nbk
            head[s:e] = board(head[None, s:e].astype("float32"), SR, reset=False)[0]
            lf.cutoff_hz = float(120 * (16000/120) ** (kk/(nbk-1)))
        seg[:nop] = head * (0.4 + 0.6 * np.sin(np.linspace(0, np.pi/2, nop)) ** 2)
    n_dc = min(int(0.005 * SR), len(seg))
    seg[:n_dc] *= np.sin(np.linspace(0, np.pi / 2, n_dc)) ** 2
    if tool in ("__direct__", "direct"):
        head_rms = float(np.sqrt(np.mean(seg[:int(0.5 * SR)] ** 2)))
        if head_rms > 0.1:
            nr = min(int(1 * beatA * SR), len(seg))
            seg[:nr] *= np.linspace(0.6, 1.0, nr)
    return seg, sync_delta


def render_set(order, plans, out_dir, entry0_override=None, final_cut=None):
    """Render the whole set on one timeline; plans[i] is the i-th junction."""
    def _in_voiced(tid, t, m=0.08):
        return any(s0 + m < t < e0 - m for s0, e0 in A.VOICED.get(tid, []))
    for i in range(min(len(order) - 1, len(plans))):
        pl = plans[i]
        if pl.get("_snapped") or str(pl.get("exit_tool", "")).startswith("move"):
            continue
        try:
            ta, tb = order[i], order[i + 1]
            for key, tid in (("cut", ta), ("entry", tb)):
                v0 = float(pl.get(key, 0.0) or 0.0)
                bts = A.REG[tid].get("beat_times") or []
                if v0 <= 0 or not bts:
                    continue
                bts = np.asarray(bts, dtype=float)
                j = int(np.argmin(np.abs(bts - v0)))
                cands = [bts[j]] + ([bts[j - 1]] if j > 0 else []) + ([bts[j + 1]] if j + 1 < len(bts) else [])
                cands = [float(c) for c in cands if abs(c - v0) <= 0.6 * A.beat_of(tid)]
                cands.sort(key=lambda c: abs(c - v0))
                orig_bad = _in_voiced(tid, v0)
                chosen = next((c for c in cands if not _in_voiced(tid, c)), None)
                if chosen is None and orig_bad and cands:
                    chosen = cands[0]
                if chosen is not None and abs(chosen - v0) > 1e-3:
                    pl.setdefault("_snap_log", {})[key] = [round(v0, 3), round(chosen, 3)]
                    pl[key] = round(chosen, 3)
            pl["_snapped"] = True
        except Exception:
            pass
    out, tmarks = [], []
    t_acc = 0.0
    cur_tail = None
    cur_sync_delta = 0.0
    for i, tid in enumerate(order):
        beat = A.beat_of(tid)
        ks_here = int(plans[i - 1].get("keysync", 0)) if i > 0 else 0
        if i == 0:
            cur_sync_delta = 0.0
            entry0 = (entry0_override if entry0_override is not None
                      else A.entry_options(tid).get("v_first") or 0.0)
            stream = A.audio(tid)[int(entry0 * SR):]
            base_entry = entry0
        else:
            pl_prev = plans[i - 1]
            if "_b_cont" in pl_prev:
                stream = A.audio(tid)[int(pl_prev["_b_cont"] * SR):]
                if pl_prev.get("_gain_db"):
                    stream = stream * (10 ** (float(pl_prev["_gain_db"]) / 20))
                base_entry = pl_prev["_b_cont"]
                cur_sync_delta = 0.0
                cur_tail = None
                ks_here = 0
                pl_prev = dict(pl_prev, entry=base_entry)
            else:
                _need_sync = float(pl_prev.get("overlap_beats", 0) or 0) > 0
                stream, _sd = make_entry_stream(tid, pl_prev["entry"], pl_prev.get("entry_tool", "cut"),
                                                ks_here, A.beat_of(order[i - 1]), 600, need_sync=_need_sync)
                _g_db = _match_gain_db(order[i - 1], float(pl_prev["cut"]), tid, float(pl_prev["entry"]))
                if _g_db:
                    stream = stream * (10 ** (_g_db / 20))
                pl_prev["_gain_db"] = round(_g_db, 2)
                _etp = str(pl_prev.get("exit_tool", ""))
                if _etp == "echo" or (_etp in ("filter_lpf", "filter_hpf")
                                      and not float(pl_prev.get("overlap_beats", 0) or 0)):
                    _w = int(4 * A.beat_of(order[i - 1]) * SR)
                    stream = np.concatenate([np.zeros(_w, dtype=stream.dtype), stream])
                    _sd -= _w / SR
                elif _etp in ("brake", "backspin"):
                    _w = int(2 * A.beat_of(order[i - 1]) * SR)
                    stream = np.concatenate([np.zeros(_w, dtype=stream.dtype), stream])
                    _sd -= _w / SR
                cur_sync_delta = _sd
            if cur_tail is not None and len(cur_tail):
                L = min(len(cur_tail), len(stream))
                stream[:L] = stream[:L] + cur_tail[:L]
            base_entry = pl_prev["entry"]
        if i < len(order) - 1 and plans[i].get("exit_tool") == "blend16":
            pl = plans[i]
            nxt_tid = order[i + 1]
            beatB = A.beat_of(nxt_tid)
            rt = beatB / beat
            while rt > 1.5: rt /= 2
            while rt < 0.75: rt *= 2
            ov = 16 * beat
            cut = pl["cut"]; entryB = pl["entry"]
            b_end = cut - base_entry - cur_sync_delta
            body = stream[:max(0, int((b_end - ov) * SR))]
            a_tail = stream[max(0, int((b_end - ov) * SR)):int(b_end * SR)].copy()
            src0 = max(0.0, entryB - ov * rt)
            bh = A.audio(nxt_tid)[int(src0 * SR):int(entryB * SR)].copy()
            _g_db = _match_gain_db(tid, cut, nxt_tid, entryB); pl["_gain_db"] = round(_g_db, 2)
            if _g_db: bh = bh * (10 ** (_g_db / 20))
            if abs(rt - 1) > 0.005:
                bh = A._rb_stretch(bh, float(rt))
            L = min(len(a_tail), len(bh), int(ov * SR))
            ne = min(int(0.008 * SR), L // 2)
            a_env = np.ones(L); a_env[-ne:] = np.linspace(1, 0, ne)
            b_env = np.ones(L); b_env[:ne] = np.linspace(0, 1, ne)
            mix = a_tail[:L] * a_env + bh[-L:] * (BLEND_B_RIDE * b_env)
            out.append(body); out.append(mix)
            t_acc += (len(body) + len(mix)) / SR
            tmarks.append(t_acc)
            plans[i]["_b_cont"] = entryB
            cur_tail = None
            continue
        if i < len(order) - 1 and plans[i].get("exit_tool") == "blendecho":
            pl = plans[i]
            nxt_tid = order[i + 1]
            beatB = A.beat_of(nxt_tid)
            rt = beatB / beat
            while rt > 1.5: rt /= 2
            while rt < 0.75: rt *= 2
            ov = 16 * beat
            cut = pl["cut"]; entryB = pl["entry"]
            b_end = cut - base_entry - cur_sync_delta
            body = stream[:max(0, int((b_end - ov) * SR))]
            a_tail = stream[max(0, int((b_end - ov) * SR)):int(b_end * SR)].copy()
            src0 = max(0.0, entryB - ov * rt)
            bi = A.audio(nxt_tid)[int(src0 * SR):int(entryB * SR)].copy()
            _g_db = _match_gain_db(tid, cut, nxt_tid, entryB); pl["_gain_db"] = round(_g_db, 2)
            if _g_db: bi = bi * (10 ** (_g_db / 20))
            if abs(rt - 1) > 0.005:
                bi = A._rb_stretch(bi, float(rt))
            L = min(len(a_tail), len(bi), int(ov * SR))
            ramp = np.linspace(0, 1, L)
            bl3, al3 = ss.butter(2, 2000 / (SR / 2), "low")
            bi_f = ss.lfilter(bl3, al3, bi[:L]) * (1 - ramp) + bi[:L] * ramp
            _ae, _be, _Lb = blend_bed_env(L, beat, SR, a_tail[:L], bi_f)
            pl["_bed_level"] = _Lb; pl["_mode"] = "realdj" if _Lb == 1.0 else "crossfade"
            mix = a_tail[:L] * _ae + bi_f * _be
            lenb = min(max(int(pl.get("exit_len_beats", 2) or 2), 1), 2)
            _, wet, _ = apply_exit_tail(A.audio(tid), cut, "echo", beat, lenb,
                                        float(pl.get("echo_delay_beats", 1.0)))
            if ks_here and len(wet):
                wet = A._rb_pitch(wet, ks_here)
            b_after = A.audio(nxt_tid)[int(entryB * SR):int(entryB * SR) + len(wet)].copy()
            if _g_db: b_after = b_after * (10 ** (_g_db / 20))
            n2 = min(len(wet), len(b_after))
            out.append(body); out.append(mix)
            t_acc += (len(body) + len(mix)) / SR
            tmarks.append(t_acc)
            if n2 > 0:
                out.append(b_after[:n2] + wet[:n2] * 0.9)
                t_acc += n2 / SR
            plans[i]["_b_cont"] = entryB + n2 / SR
            cur_tail = None
            continue
        if i < len(order) - 1 and plans[i].get("exit_tool") == "introstack":
            pl = plans[i]
            nxt_tid = order[i + 1]
            beatB = A.beat_of(nxt_tid)
            ov_beats = int(pl.get("overlap_beats", 16))
            rt = beatB / beat
            while rt > 1.5: rt /= 2
            while rt < 0.75: rt *= 2
            cut = pl["cut"]
            b_end = cut - base_entry
            body = stream[:max(0, int((b_end - ov_beats * beat) * SR))]
            a_tail = stream[max(0, int((b_end - ov_beats * beat) * SR)):int(b_end * SR)].copy()
            intro_src = ov_beats * beat * rt
            bi = A.audio(nxt_tid)[:int(intro_src * SR)].copy()
            _g_db = _match_gain_db(tid, cut, nxt_tid, intro_src); pl["_gain_db"] = round(_g_db, 2)
            if _g_db: bi = bi * (10 ** (_g_db / 20))
            if abs(rt - 1) > 0.005:
                bi = A._rb_stretch(bi, float(rt))
            L = min(len(a_tail), len(bi), int(ov_beats * beat * SR))
            ramp = np.linspace(0, 1, L)
            bl2, al2 = ss.butter(2, 2000 / (SR / 2), "low")
            bi_f = ss.lfilter(bl2, al2, bi[:L]) * (1 - ramp) + bi[:L] * ramp
            mix = a_tail[:L] * (np.cos(ramp * np.pi / 2) ** 0.7) + bi_f * (0.3 + (BLEND_B_RIDE - 0.3) * ramp)
            out.append(body); out.append(mix)
            t_acc += (len(body) + len(mix)) / SR
            tmarks.append(t_acc)
            plans[i]["_b_cont"] = intro_src
            cur_tail = None
            continue
        if i < len(order) - 1 and str(plans[i].get("exit_tool", "")).startswith("move"):
            pl = plans[i]
            mp = pl["mj_plan"]; clip = np.load(f"{pl['clip_npy']}")
            LEAD = 12.0
            b_end = pl["mj_exit_t"] - LEAD - base_entry
            body = stream[:max(0, int(b_end * SR))]
            c0 = int((mp["exit_clip"] - LEAD) * SR)
            c1 = min(len(clip), int((mp["handover"] + 16 * A.beat_of(tid)) * SR))
            seg = clip[c0:c1]
            out.append(body); out.append(seg)
            t_acc += (len(body) + len(seg)) / SR
            tmarks.append(t_acc - (c1/SR - mp["handover"]))
            f = mp.get("stretch", 1.0)
            beatX = A.beat_of(tid)
            xf = int(2 * beatX * SR)
            clip_after_handover = (c1 / SR) - mp["handover"]
            src_fade0 = pl["mj_entry_v0"] + max(0.0, clip_after_handover - xf / SR) * f
            nxt = A.audio(order[i + 1])
            if abs(f - 1.0) > 0.01:
                ramp_len = int(8 * beatX * SR)
                chunks, src = [], src_fade0
                for k in range(4):
                    fk = f + (1.0 - f) * (k + 0.5) / 4
                    take = int(ramp_len / 4 * fk)
                    seg_r = nxt[int(src * SR):int(src * SR) + take]
                    chunks.append(A._rb_stretch(seg_r, float(fk))
                                  if abs(fk - 1) > 0.004 else seg_r)
                    src += take / SR
                cont = np.concatenate(chunks + [nxt[int(src * SR):]])
                b_cont_src = src
            else:
                cont = nxt[int(src_fade0 * SR):]
                b_cont_src = src_fade0
            tail_seg = out[-1]
            L = min(xf, len(tail_seg), len(cont))
            p = np.linspace(0, 1, L)
            tail_seg[-L:] = tail_seg[-L:] * (np.cos(p*np.pi/2)**2) + cont[:L] * (np.sin(p*np.pi/2)**2)
            ramp_rest = cont[L:max(L, len(cont) - len(nxt) + int(b_cont_src * SR))]
            if abs(f - 1.0) > 0.01 and len(ramp_rest):
                out.append(ramp_rest)
                t_acc += len(ramp_rest) / SR
            plans[i]["_b_cont"] = b_cont_src if abs(f - 1.0) > 0.01 else src_fade0 + L / SR
            cur_tail = None
            continue
        if i < len(order) - 1:
            pl = plans[i]
            lenb = int(pl.get("exit_len_beats", 2))
            dur_here = pl["cut"] - base_entry - cur_sync_delta
            head, tail, consumed = apply_exit_tail(A.audio(tid), pl["cut"], pl["exit_tool"],
                                                   beat, lenb,
                                                   float(pl.get("echo_delay_beats", 1.0)),
                                                   float(pl.get("echo_feedback", 0.5)),
                                                   float(pl.get("echo_lo_hz", 300)),
                                                   float(pl.get("echo_hi_hz", 3000)),
                                                   int(pl.get("echo_reps", 8)))
            if ks_here:
                head = A._rb_pitch(head, ks_here)
                if len(tail): tail = A._rb_pitch(tail, ks_here)
            adv = float(pl.get("entry_advance_beats", 0) or 0)
            if adv > 0 and pl["exit_tool"] in ("brake", "backspin"):
                na = min(int(adv * beat * SR), len(head) - 1)
                if na > 0:
                    tail = head[-na:].copy() if not len(tail) else np.concatenate([head[-na:], tail])
                    head = head[:-na]
            ov_extra = 0
            ovb = float(pl.get("overlap_beats", 0) or 0)
            if ovb > 0:
                ov_s = int(ovb * beat * SR)
                pre_len = max(0, ov_s - consumed - (len(tail) if adv > 0 else 0))
                i_cut2 = int(pl["cut"] * SR)
                a_pre = A.audio(tid)[i_cut2 - consumed - pre_len:i_cut2 - consumed].copy()
                if ks_here and len(a_pre):
                    a_pre = A._rb_pitch(a_pre, ks_here)
                overlay = np.concatenate([a_pre, head]) if len(a_pre) else head
                tail = overlay if not len(tail) else np.concatenate([overlay, tail])
                head = np.zeros(0, dtype=np.float32)
                ov_extra = pre_len
            b1 = int(dur_here * SR) - consumed - ov_extra
            body = stream[:max(0, b1)]
            out.append(body); out.append(head)
            t_acc += (len(body) + len(head)) / SR
            tmarks.append(t_acc)
            cur_tail = tail
        else:
            gs = A.chorus_groups(tid)
            end_abs = final_cut if final_cut is not None else (gs[-1]["end"] if gs else base_entry + 60)
            end_rel = end_abs - base_entry - cur_sync_delta
            seg = stream[:int(np.clip(end_rel + 2.0, 20.0, 180.0) * SR)].copy()
            fade = int(8 * A.beat_of(tid) * SR)
            if len(seg) > fade:
                seg[-fade:] *= np.linspace(1, 0, fade)
            out.append(seg)
            t_acc += len(seg) / SR
    y = np.concatenate(out)
    _w = int(0.1 * SR); _n = len(y) // _w
    if _n >= 4:
        _fr = np.sqrt((y[:_n * _w].reshape(_n, _w) ** 2).mean(1))
        _act = _fr[_fr > _fr.max() * 0.1]
        _cur = float(np.median(_act)) if len(_act) else float(np.sqrt((y ** 2).mean()) + 1e-9)
        _tgt = 10 ** (-14.5 / 20)
        if _cur > 1e-6:
            y = y * float(np.clip(_tgt / _cur, 0.25, 4.0))
    m = np.abs(y).max()
    if m > 0.97:
        y = y / m * 0.97
    sf.write(f"{out_dir}/set.wav", y, SR)
    for j, tj in enumerate(tmarks):
        _pl = plans[j] if j < len(plans) else {}
        _tool = str(_pl.get("exit_tool", ""))
        PRE, POST = 30.0, 30.0
        need = int((PRE + POST) * SR)
        s0 = int((tj - PRE) * SR); s1 = int((tj + POST) * SR)
        if s0 < 0: s1 += -s0; s0 = 0
        if s1 > len(y): s0 = max(0, s0 - (s1 - len(y))); s1 = len(y)
        clip = y[s0:s0 + need]
        if len(clip) < need:
            clip = np.concatenate([clip, np.zeros(need - len(clip), dtype=clip.dtype)])
        sf.write(f"{out_dir}/trans{j+1}.wav", clip, SR)
    return tmarks, len(y) / SR


