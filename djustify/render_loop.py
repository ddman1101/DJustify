# -*- coding: utf-8 -*-
"""loop_in engine (= b25_loop method verified in this session, moved from scratchpad/loopin.py).
Loops B's intro or bridge as a bed, muffled and quiet at first, and holds it until A's phrase ends and B's verse takes over.
render_loopin(ta,tb) → dict(clip, junction, cutA, entryH) or None (infeasible → caller degrades).
Extend context to ≥30s, allowing caller to crop to centered 60s."""
import os
import numpy as np, librosa, scipy.signal as ss
import song_library as A
EV = os.path.join(os.environ.get("AIDJ_RUNTIME_ROOT", "runtime"), "dj_transition_planner", "eval")
NP = os.path.join(os.environ.get("AIDJ_RUNTIME_ROOT", "runtime"), "dataset", "normalized_pool")
SR = 44100

def rms(x): return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)
import os as _os
def _bed_env(L2, beat, a_seg, b_seg, snap_beats=None):
    """A full + B full, do not move fader (rely on cue matching, not compensation)."""
    return np.ones(L2), np.ones(L2), 1.0
def limit(x, ceil=0.97):
    from scipy.ndimage import maximum_filter1d
    raw = np.maximum(1.0, np.abs(x) / ceil)
    peak = maximum_filter1d(raw, size=max(1, int(0.005 * SR)))
    b, a = ss.butter(2, 60 / (SR / 2)); sm = ss.filtfilt(b, a, peak)
    env = np.maximum(sm, raw)
    return x / env

def B_intro_loop(tb):
    y = librosa.load(f"{NP}/{tb}.mp3", sr=22050, mono=True)[0]
    bpm = A.REG[tb]["bpm"]; beat = 60.0 / bpm; dbs = A.REG[tb].get("downbeat_times") or []
    if len(dbs) < 4: return None
    hop = 512; ch = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=hop)
    mf = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13, hop_length=hop)[1:]
    rms_f = librosa.feature.rms(y=y, hop_length=hop)[0]; fps = 22050 / hop
    def at(tt):
        f = int(tt * fps); w = max(1, int(0.23 * fps)); sl = slice(max(0, f - w), f + w)
        return ch[:, sl].mean(1), mf[:, sl].mean(1), float(rms_f[sl].mean())
    def chunk(t_): cs = [at(t_ + k * beat) for k in range(4)]; return (np.mean([c[0] for c in cs], 0), np.mean([c[1] for c in cs], 0), np.mean([c[2] for c in cs]))
    def L(a, c):
        dc = 1 - float(np.dot(a[0], c[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(c[0]) + 1e-9))
        dm = float(np.linalg.norm(a[1] - c[1]) / (np.linalg.norm(a[1]) + np.linalg.norm(c[1]) + 1e-9)); dr = abs(a[2] - c[2]) / (max(a[2], c[2]) + 1e-9)
        return 1.0 * dc + 0.6 * dm + 0.2 * dr
    dur = A.REG[tb]["duration_sec"]; voiced = sorted(A.VOICED.get(tb, []))
    def vcov(t0, t1): return sum(max(0, min(t1, e) - max(t0, s)) for s, e in voiced if e > t0 and s < t1)
    med_rms = float(np.median(rms_f)); best = None
    for sg in A.REG[tb]["segments"]:
        if sg["label"] not in ("verse", "chorus"): continue
        H = sg["start"]
        if H > dur * 0.6: continue
        j = int(np.argmin([abs(d - H) for d in dbs]))
        if abs(dbs[j] - H) > 0.35 or j < 2: continue
        H = dbs[j]; b0 = dbs[j - 2]; loop_len = H - b0
        if not (6 * beat < loop_len < 10 * beat): continue
        if vcov(b0, H - 0.1) > 0.15: continue
        if vcov(H - beat, H + 0.05) > 0.05: continue
        if at((b0 + H) / 2)[2] < med_rms * 0.6: continue
        seam = L(chunk(b0), chunk(H))
        score = seam + 0.02 * H
        if best is None or score < best[0]: best = (score, float(H), float(b0), float(loop_len), float(seam))
    if best is None or best[4] > 0.35: return None
    return best

def onset_env_ta(ta):
    y = librosa.load(f"{NP}/{ta}.mp3", sr=22050, mono=True)[0]
    return librosa.onset.onset_strength(y=y, sr=22050, hop_length=512), 22050 / 512

def A_cueout(ta, oe, fps):
    dbs = A.REG[ta].get("downbeat_times") or []; dur = A.REG[ta]["duration_sec"]; beat = 60.0 / A.REG[ta]["bpm"]
    voiced = sorted(A.VOICED.get(ta, [])); onmax = float(oe.max()) + 1e-9
    def straddle(tt): return any(s + 0.25 < tt < e - 0.25 for s, e in voiced)
    def onset_at(tt): f = int(tt * fps); w = max(1, int(0.06 * fps)); return float(oe[max(0, f - w):f + w].max()) / onmax
    best = None
    for db in dbs:
        if not (0.45 * dur < db < 0.75 * dur): continue
        if straddle(db): continue
        prev = max([e for s, e in voiced if e <= db + 0.2], default=-99)
        recent = 0 < (db - prev) < 4 * beat
        sc = -onset_at(db) - (0.5 if recent else 0)
        if best is None or sc < best[0]: best = (sc, float(db))
    return best[1] if best else None

def render_loopin(ta, tb, echo=False, ctx=32):
    """Returns dict(clip, junction seconds, cutA, entryH) or None. ctx = how many seconds to leave before and after (for centered 60s trim)."""
    bi = B_intro_loop(tb)
    if bi is None: return None
    _, H, b0, loop_len, seam = bi
    oe, fps = onset_env_ta(ta); t_cutA = A_cueout(ta, oe, fps)
    if t_cutA is None: return None
    beatA = 60.0 / A.REG[ta]["bpm"]
    yA = librosa.load(f"{NP}/{ta}.mp3", sr=SR, mono=True)[0]; yB = librosa.load(f"{NP}/{tb}.mp3", sr=SR, mono=True)[0]
    rate = A.REG[ta]["bpm"] / A.REG[tb]["bpm"]
    rate = min([rate, rate * 2, rate / 2, rate * 4, rate / 4], key=lambda x: abs(np.log2(x)))
    yBs = A._rb_stretch(yB, rate) if abs(rate - 1) > 0.02 else yB
    scale = len(yBs) / len(yB); Hs = H * scale; b0s = b0 * scale; loop_s = (loop_len * scale)
    chunk = yBs[int(b0s * SR):int(Hs * SR)]
    if len(chunk) < 0.5 * SR: return None
    loop_len_beats = max(1, round(loop_s / beatA)); n_loops = max(1, round(16 / loop_len_beats))
    overlap = np.tile(chunk, n_loops)
    s_cut = int(round(t_cutA * SR))
    if s_cut - len(overlap) < int(8 * SR): return None
    aov = yA[s_cut - len(overlap):s_cut]
    if abs(20 * np.log10(rms(overlap) / rms(aov))) > 6.0: return None
    oa = librosa.onset.onset_strength(y=aov, sr=SR, hop_length=512)
    ob = librosa.onset.onset_strength(y=overlap[:len(aov)], sr=SR, hop_length=512)
    n = min(len(oa), len(ob)); oa = oa[:n] - oa[:n].mean(); ob = ob[:n] - ob[:n].mean()
    maxlag = max(1, int((beatA * 0.5) * SR / 512)); corr = ss.correlate(ob, oa, mode="full"); mid = n - 1
    seg = corr[max(0, mid - maxlag):mid + maxlag + 1]; lag = int(np.argmax(seg)) - min(maxlag, mid); shift = int(lag * 512)
    if shift != 0:
        st = int(b0s * SR) + shift
        overlap = np.tile(yBs[max(0, st):max(0, st) + len(chunk)], n_loops)[:len(aov)]
    L2 = min(len(aov), len(overlap)); ov = overlap[:L2]; a2 = aov[:L2]
    open_ = np.clip(np.linspace(0, 1, L2) / 0.25, 0, 1)
    bl, al = ss.butter(2, 900 / (SR / 2), "low"); b_lp = ss.lfilter(bl, al, ov)
    b_open = b_lp * (1 - open_) + ov * open_
    _ae, _be, _Lb = _bed_env(L2, beatA, a2, b_open)
    mix = a2 * _ae + b_open * _be
    a_head = yA[max(0, s_cut - len(overlap) - int(ctx * SR)):s_cut - len(overlap)]
    b_cont = yBs[int(Hs * SR):int(Hs * SR) + int(ctx * SR)].copy()
    if echo:
        ec = a2[-int(beatA * SR):] if len(a2) >= int(beatA * SR) else a2
        bh, ah = ss.butter(2, 300 / (SR / 2), "high"); bl2, al2 = ss.butter(2, 3200 / (SR / 2), "low")
        D = int(beatA * SR); tl = int(5 * beatA * SR); send = np.concatenate([ec, np.zeros(tl)]); wet = np.zeros(len(send) + D)
        for _ in range(6):
            dd = np.zeros(len(send) + D); dd[D:D + len(send)] = ss.lfilter(bl2, al2, ss.lfilter(bh, ah, send + 0.5 * wet[:len(send)])); wet = dd
        n2 = min(len(wet), len(b_cont)); b_cont[:n2] = b_cont[:n2] + wet[:n2] * 0.7
    out = np.concatenate([a_head, mix[:L2], b_cont])
    junction = (len(a_head) + L2) / SR
    return {"clip": limit(out, 0.97), "junction": junction, "cutA": float(t_cutA), "entryH": float(H)}


def loopmaker(t, want_vocal):
    y = librosa.load(f"{NP}/{t}.mp3", sr=22050, mono=True)[0]
    bpm = A.REG[t]["bpm"]; beat = 60.0 / bpm; dbs = A.REG[t].get("downbeat_times") or []
    if len(dbs) < 8: return None
    hop = 512; ch = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=hop)
    mf = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13, hop_length=hop)[1:]; rms_f = librosa.feature.rms(y=y, hop_length=hop)[0]
    ons = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop); onmax = float(ons.max()) + 1e-9; fps = 22050 / hop
    def onset_at(tt):
        f = int(tt * fps); w = max(1, int(0.06 * fps)); return float(ons[max(0, f - w):f + w].max()) / onmax
    def at(tt):
        f = int(tt * fps); w = max(1, int(0.23 * fps)); sl = slice(max(0, f - w), f + w)
        return ch[:, sl].mean(1), mf[:, sl].mean(1), float(rms_f[sl].mean())
    def chunk(t_): cs = [at(t_ + k * beat) for k in range(4)]; return (np.mean([c[0] for c in cs], 0), np.mean([c[1] for c in cs], 0), np.mean([c[2] for c in cs]))
    def L(a, c):
        dc = 1 - float(np.dot(a[0], c[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(c[0]) + 1e-9))
        dm = float(np.linalg.norm(a[1] - c[1]) / (np.linalg.norm(a[1]) + np.linalg.norm(c[1]) + 1e-9)); dr = abs(a[2] - c[2]) / (max(a[2], c[2]) + 1e-9)
        return 1.0 * dc + 0.6 * dm + 0.2 * dr
    dur = A.REG[t]["duration_sec"]; voiced = sorted(A.VOICED.get(t, []))
    def vcov(t0, t1): return sum(max(0, min(t1, e) - max(t0, s)) for s, e in voiced if e > t0 and s < t1)
    def straddle(tt): return any(s + 0.25 < tt < e - 0.25 for s, e in voiced)
    best = None
    for j in range(len(dbs) - 2):
        db = dbs[j]; end = dbs[j + 2]
        if not (0.25 * dur <= db <= 0.72 * dur): continue
        seg = end - db
        if not (6 * beat < seg < 10 * beat): continue
        vc = vcov(db, end) / seg; is_v = vc > 0.5
        if want_vocal != is_v: continue
        d = L(chunk(db), chunk(end))
        if d > 0.25: continue
        end_strength = onset_at(end); vpen = 0.0
        if straddle(db): vpen += 0.30
        if straddle(end): vpen += 0.30
        score = d - 0.25 * end_strength + vpen
        if best is None or score < best[0]: best = (score, float(db), float(end), d, end_strength)
    return best

def B_head(tb, need_inst_s):
    dur = A.REG[tb]["duration_sec"]; voiced = sorted(A.VOICED.get(tb, []))
    for sg in A.REG[tb]["segments"]:
        if sg["label"] not in ("verse", "chorus"): continue
        h = sg["start"]
        if not (8 <= h <= dur * 0.55): continue
        w0 = h - need_inst_s
        if w0 < 0: continue
        vc = sum(max(0, min(h - 0.3, e) - max(w0, s)) for s, e in voiced if e > w0 and s < h - 0.3)
        if vc < 0.5: return h
    return None

def _do_loop_a(ta, tb, t0, t_end, head, echo, ctx=30):
    yA = librosa.load(f"{NP}/{ta}.mp3", sr=SR, mono=True)[0]; yB = librosa.load(f"{NP}/{tb}.mp3", sr=SR, mono=True)[0]
    beat = 60.0 / A.REG[ta]["bpm"]
    s0 = int(round(t0 * SR)); loop = yA[s0:int(round(t_end * SR))]; Ll = len(loop); loop_s = Ll / SR
    loop_rms = rms(loop)
    rate = A.REG[ta]["bpm"] / A.REG[tb]["bpm"]
    rate = min([rate, rate * 2, rate / 2, rate * 4, rate / 4], key=lambda x: abs(np.log2(x)))
    yBs = A._rb_stretch(yB, rate) if abs(rate - 1) > 0.02 else yB
    scale = len(yBs) / len(yB); head_bs = head * scale; b_pull = head_bs - 2 * loop_s
    dbB = [d * scale for d in (A.REG[tb].get("downbeat_times") or [])]
    if dbB: b_pull = min(dbB, key=lambda x: abs(x - b_pull))
    b_pull = max(0, b_pull); plat_s = head_bs - b_pull
    if plat_s < loop_s * 0.5: return None
    nt = int(np.ceil(plat_s / loop_s)) + 1; loops = np.tile(loop, nt)[:int(plat_s * SR)]; Lp = len(loops)
    bp0 = int(b_pull * SR); bo = yBs[bp0:bp0 + Lp]
    if len(bo) < Lp * 0.6: return None
    L2 = min(Lp, len(bo))
    if abs(20 * np.log10(rms(bo[:L2]) / loop_rms)) > 5.0: return None
    oa = librosa.onset.onset_strength(y=loops[:L2], sr=SR, hop_length=512)
    ob = librosa.onset.onset_strength(y=bo[:L2], sr=SR, hop_length=512)
    n = min(len(oa), len(ob)); oa = oa[:n] - oa[:n].mean(); ob = ob[:n] - ob[:n].mean()
    maxlag = max(1, int((beat * 0.5) * SR / 512)); corr = ss.correlate(ob, oa, mode="full"); mid = n - 1
    seg = corr[max(0, mid - maxlag):mid + maxlag + 1]; lag = int(np.argmax(seg)) - min(maxlag, mid); shift = int(lag * 512)
    if shift != 0:
        bp1 = min(max(0, bp0 + shift), len(yBs) - Lp); bo = yBs[bp1:bp1 + Lp]; L2 = min(Lp, len(bo)); b_pull = bp1 / SR
    head_bs2 = b_pull + plat_s
    _ae, _be, _Lb = _bed_env(L2, beat, loops[:L2], bo[:L2])
    mix = loops[:L2] * _ae + bo[:L2] * _be
    bcont = yBs[int(head_bs2 * SR):int(head_bs2 * SR) + int(ctx * SR)]
    hd = yA[max(0, s0 - int(ctx * SR)):s0]
    if echo:
        ec = loop[-int(beat * SR):]; bh, ah = ss.butter(2, 300 / (SR / 2), "high"); bl, al = ss.butter(2, 3200 / (SR / 2), "low")
        D = int(beat * SR); tl = int(5 * beat * SR); send = np.concatenate([ec, np.zeros(tl)]); wet = np.zeros(len(send) + D)
        for _ in range(7):
            dd = np.zeros(len(send) + D); dd[D:D + len(send)] = ss.lfilter(bl, al, ss.lfilter(bh, ah, send + 0.52 * wet[:len(send)])); wet = dd
        n2 = min(len(wet), len(bcont)); bcont = np.concatenate([bcont[:n2] + wet[:n2] * 0.85, bcont[n2:]])
    out = np.concatenate([hd, mix, bcont]); m = np.abs(out).max()
    out = out / m * 0.92 if m > 1e-6 else out
    junction = (len(hd) + L2) / SR
    return {"clip": out, "junction": junction, "cutA": float(t_end), "entryH": float(head)}

def render_loop_a(ta, tb, echo=False, ctx=30):
    for wv in (False, True):
        lm = loopmaker(ta, wv)
        if not lm: continue
        beatA = 60.0 / A.REG[ta]["bpm"]; need_inst = 8 * beatA * 2 * 1.1
        h = B_head(tb, need_inst)
        if h is None: continue
        r = _do_loop_a(ta, tb, lm[1], lm[2], h, echo, ctx)
        if r is not None: return r
    return None
