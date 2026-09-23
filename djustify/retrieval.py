# -*- coding: utf-8 -*-
"""retrieval: tech-first mode 3 — /live/tf"""
import json, os as _os, shutil
import numpy as np
import song_library as A
import render_plan as PF
import regression_check as RC
from plan_sim import sim_c1, sim_c2, safe_cut_near
import transition_core as LD

_PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
       "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
def camelot(k):
    if not k or ":" not in k: return None
    pc, m = k.split(":")
    if pc not in _PC: return None
    n = _PC[pc]
    return ((n * 7) % 12, 0) if m == "maj" else (((n + 3) % 12 * 7) % 12, 1)
def key_dist(ka, kb):
    a, b = camelot(ka), camelot(kb)
    if not a or not b: return 1.5
    d = min((a[0]-b[0]) % 12, (b[0]-a[0]) % 12)
    return d + (0.5 if a[1] != b[1] else 0)
def bpm_fold(ra, rb):
    rt = rb / ra
    while rt > 1.5: rt /= 2
    while rt < 0.75: rt *= 2
    return abs(rt - 1)

def entry_candidates(tb, need):
    reg = A.REG[tb]; beat = A.beat_of(tb); v = A.VOICED.get(tb, [])
    dur = reg["duration_sec"]
    ce = PF.CUEEV.get(str(tb), {})
    out = []
    def lead_clean(e, beats):
        t0 = e - beats * beat
        if t0 < 0: return 0.0
        occ = sum(max(0.0, min(e - 0.3, e2) - max(t0, s2)) for s2, e2 in v)
        return max(0.0, beats - occ / beat)
    def c2_safe(e):
        return not any(s2 < e - 0.3 and e2 > e - 0.3 for s2, e2 in v)
    def vocal_at(e):
        return any(abs(s2 - e) < 2.0 for s2, e2 in v)
    from plan_sim import snap_downbeat as _sdb
    if need == "lead16":
        for sg in reg.get("segments", []):
            if sg["label"] not in ("verse", "chorus"): continue
            e = _sdb(tb, sg["start"])
            if e < 8 or e / dur > 0.55 or dur - e < 45: continue
            lc = lead_clean(e, 16)
            if lc >= 12 and vocal_at(e):
                out.append((round(e, 1), {"Type": sg["label"] + "Head", "Clean intro beat": round(lc, 1)}, lc / 16))
    elif need == "section_head":
        se = (PF.SEGENV.get(str(tb)) or {}).get("segs", [])
        env0 = {x["i"]: x["env"][0] for x in se}
        for si, sg in enumerate(reg.get("segments", [])):
            if sg["label"] in ("start", "end", "outro", "silence"): continue
            e = _sdb(tb, sg["start"])
            if e / dur > 0.55 or dur - e < 45: continue
            if c2_safe(e) and env0.get(si, 0.4) >= 0.3:
                out.append((round(e, 1), {"Type": sg["label"] + "Head", "Head energy": env0.get(si, 0.4)}, env0.get(si, 0.4)))
    elif need == "drum_entry":
        for e in (ce.get("kick", []) + ce.get("bass", [])):
            if e < 2 or e / dur > 0.55 or dur - e < 45: continue
            if c2_safe(e):
                out.append((round(e, 1), {"Type": "Drums/bass enters"}, 0.8))
    from plan_sim import _vocal_gap
    out = [(e, {**ev, "instrumental lead before vocal (s)": round(_vocal_gap(tb, e), 1)}, sp) for e, ev, sp in out]
    return [c for c in out if not sim_c2(tb, c[0])[1]]

def _tune(name, default):
    try:
        return float(_os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def retrieve(ta, need, va_dir, topk=6, min_gap_s=0.0):
    ra = A.REG[ta]["bpm"]; ka = A.KEYS.get(ta, {}).get("key")
    va = RC.VAM.get(ta, {})
    max_bpm = _tune("AIDJ_MAX_BPM_FOLD", 0.08)
    max_key = _tune("AIDJ_MAX_KEY_DIST", 2.5)
    va_near = _tune("AIDJ_VA_NEAR_BAND", 0.45)
    va_side = _tune("AIDJ_VA_SIDE_BAND", 0.4)
    rows = []
    for tb in A.REG:
        if tb == ta or tb in A.QUAR or tb in A.SUSPECT or not A.in_pool(tb): continue
        rb = A.REG[tb].get("bpm"); vb = RC.VAM.get(tb)
        if not rb or not vb or not A.REG[tb].get("segments"): continue
        bd = bpm_fold(ra, rb)
        if bd > max_bpm: continue
        kd = key_dist(ka, A.KEYS.get(tb, {}).get("key"))
        if kd > max_key: continue
        dv = vb["valence_q"] - va.get("valence_q", .5); da = vb["arousal_q"] - va.get("arousal_q", .5)
        if va_dir == "near" and abs(dv) + abs(da) > va_near: continue
        if va_dir == "up" and not (da > 0.03 and abs(dv) < va_side): continue
        if va_dir == "down" and not (da < -0.03 and abs(dv) < va_side): continue
        cands = entry_candidates(tb, need)
        cands = [c for c in cands if c[1].get("instrumental lead before vocal (s)", 99) >= min_gap_s - 0.1]
        if not cands: continue
        e, ev, sp = max(cands, key=lambda x: x[2])
        score = (sp * 2 + (1 - bd / max_bpm) + (1 - kd / max_key)
                 + (1 - (abs(dv) + abs(da)) / max(0.6, va_near)))
        rows.append({"b_tid": str(tb), "title": A.REG[tb]["title"], "suggested cue_in": e, **ev,
                     "bpm": rb, "bpm conversion difference %": round(bd * 100, 1), "key": A.KEYS.get(tb, {}).get("key"),
                     "key distance": round(kd, 1), "ΔV_q": round(dv, 2), "ΔA_q": round(da, 2), "_s": round(score, 2)})
    rows.sort(key=lambda x: -x["_s"])
    rows = rows[:topk]
    for _i, _r in enumerate(rows): _r["b_no"] = _i + 1
    return rows

NEED_OF = {"blend16": "lead16", "blendecho": "lead16", "introstack": "lead16", "spinup": "drum_entry"}
S1_TASK = ("task: Only check A cards design 2-3 candidate entrance options (plan path first then search connectors)"
           "Each option: cue_out (prioritize vocal phrase endings + rests, fade prohibited), exit_tool,"
           "b_entry_need ∈ [lead16 (B needs 16-beat instrumental intro leading directly into vocal section, used by blend16/blendecho/introstack), "
           "section_head (enter B section head directly, sequence calls), drum_entry (B drums/bass entry point, with spinup)]、"
           "va_dir ∈ [near(emotional continuation), up(push higher), down(cool down)]."
           'Output JSON only:{"option": [{"cue_out": {"anchor":"a7","offset_beats":0}, "exit_tool": "...", "b_entry_need": "...",'
           '"va_dir": "...", "why": "one sentence"}]}'
           "Reminder (highest priority): Follow 17 execute blend if feasible overlap window exists (check A sections bridge/instrumental/outro vs B intro instrumental lead)cut type only allowed after why explains which windows checked and why failed")
S3_TASK = (" task: Below are your previous entry options and the machine-retrieved B candidates (with evidence). Compare across options and select one"
           "(option swap candidates not create new B) output full plan. cue_in must take selected candidate suggested cue_in (adjustable ≤2 beats,"
           "Fine-tuning requires a reason). Fill in the candidate b_tid."
           'Output JSON only:{"b_no": number, "b_tid": "...", "cue_out": {"anchor":"a7","offset_beats":0}, "cue_in": {"anchor":"b3","offset_beats":0}, "exit_tool": "...",'
           '"entry_tool": "...", "exit_len_beats": n, "echo_delay_beats": x, "entry_advance_beats": x,'
           '"overlap_beats": n, "keysync": -2..2, "why": "two sentences"}'
           "Reminder (highest priority): Follow 17 execute blend if feasible overlap window exists (check A sections bridge/instrumental/outro vs B intro instrumental lead)cut type only allowed after why explains which windows checked and why failed")

def pipeline_tf(rid, seed_tid):
    emit, llm = LD.emit, LD.llm_stream
    try:
        ta = next(k for k in A.REG if str(k) == seed_tid)
        ca = PF.song_card(ta)
        GUIDE = PF.GUIDE
        emit(rid, "stage", {"msg": f"Seed:{A.REG[ta]['title']}|【Move First】stage1: Only look at A, think how to move…"})
        s1_msgs = [{"role": "system", "content": GUIDE + S1_TASK},
                   {"role": "user", "content": json.dumps({"A (outgoing)": ca, "technique card": PF.FX_CARD}, ensure_ascii=False)}]
        emit
        s1, _ = llm(rid, "option", s1_msgs, temperature=0.6)
        s1 = s1 or {}
        plans1 = (s1.get("option") or [])[:3]
        for attempt in range(2):
            okp, rej = [], []
            for pr in plans1:
                from plan_sim import resolve_cue
                c, _rerr = resolve_cue(ta, pr.get("cue_out"))
                et = str(pr.get("exit_tool", "echo"))
                if _rerr or c is None:
                    rej.append({"option": pr, "rejected because": f"cue_out anchor parsing failed:{_rerr}", "instruction": "Re-fill using valid IDs from the A card anchor table"})
                    continue
                pr["cue_out"] = c
                from plan_sim import sim_c9 as _s9
                _w9, _f9 = _s9(ta, c)
                if _f9:
                    rej.append({"option": pr, "rejected because": f"C9 {_w9}",
                        "instruction": "Select exit anchors in latter half of A (must pass 20% song length and not be intro)"})
                    continue
                r1, f1 = sim_c1(ta, c, et)
                if f1:
                    alt = safe_cut_near(ta, c, et)
                    rej.append({"option": pr, "rejected because": f"cue_out={c} cuts within A vocal line (articulation ratio {r1:.2f}x)",
                                "Nearest safe cut point": alt[0] if alt else None})
                    continue
                from plan_sim import validate_grid
                _gv = validate_grid(ta, c)
                if not _gv["ok"]:
                    rej.append({"option": pr, "rejected because": f"cue_out={c} is off the beat grid by {_gv['off by s']}s; copy the nearest beat below instead of computing it",
                                "instruction": f"Use the nearest anchor {_gv['nearest anchor']}s plus or minus whole beats (one beat = {round(A.beat_of(ta),3)}s); nearest beat={_gv['nearest beat']}s"})
                    continue
                okp.append(pr)
            emit(rid, "tfgate", {"attempt": attempt, "ok": okp, "rejected": rej})
            if okp or attempt == 1 or not rej:
                plans1 = okp; break
            emit(rid, "stage", {"msg": "all options rejected by sim_c1 gate → attaching safe cut point, asking model to retry..."})
            s1_msgs2 = s1_msgs + [{"role": "assistant", "content": json.dumps(s1, ensure_ascii=False)},
                                  {"role": "user", "content": json.dumps({"All your options were rejected": rej,
                                      "instruction": "cue_out must be after A phrase ends + rest (use nearest safe cut point directly); re-output complete option JSON"},
                                      ensure_ascii=False)}]
            s1, _ = llm(rid, "option retry", s1_msgs2, temperature=0.6); s1 = s1 or {}
            plans1 = (s1.get("option") or [])[:3]
        emit(rid, "stage", {"msg": "stage2: Deterministic retrieval B (BPM conversion/key circle/V/A direction/entry type/sim_c2 gate)…"})
        cand_map = {}
        for pi, pr in enumerate(plans1):
            need = pr.get("b_entry_need") or NEED_OF.get(str(pr.get("exit_tool")), "section_head")
            if need not in ("lead16", "section_head", "drum_entry"): need = "section_head"
            from plan_sim import b_lead_need_beats
            _nb, _ = b_lead_need_beats(pr.get("exit_tool"), None, pr.get("echo_delay_beats", 1.0),
                                       pr.get("overlap_beats", 0))
            cands = retrieve(ta, need, pr.get("va_dir", "near"), min_gap_s=_nb * A.beat_of(ta))
            if cands: cand_map[f"option{pi+1}"] = {"option": pr, "B candidates": cands}
        _no = 0
        for _v in cand_map.values():
            for _c in _v["B candidates"]:
                _no += 1; _c["b_no"] = _no
        if not cand_map:
            cands = retrieve(ta, "section_head", "near")
            if cands: cand_map["fallback"] = {"option": {"exit_tool": "echo", "b_entry_need": "section_head",
                                                    "why": "No candidates for all options, return to the section head sequence"}, "B candidates": cands}
        emit(rid, "tfcands", {"cand_map": cand_map})
        if not cand_map:
            emit(rid, "error", {"tag": "Retrieving", "msg": "No qualified candidates in pool (BPM/key/vocal window all mismatch)"}); return
        # stage3 final
        emit(rid, "stage", {"msg": "stage3: Cross-option comparison, final complete plan (thinking live broadcast)…"})
        s3_msgs = [{"role": "system", "content": GUIDE + PF.BLEND_FEWSHOT + S3_TASK},
                   {"role": "user", "content": json.dumps({"A (outgoing)": ca, "technique card": PF.FX_CARD,
                                                           "Exit options and B candidates": cand_map,
                                                           "Key advice for each candidate (machine-calculated, fill in keysync at final stage)": {
                                                               k: {c["title"]: PF.key_advice(A.KEYS.get(ta, {}).get("key"), c.get("key"))
                                                                   for c in v["B candidates"]} for k, v in cand_map.items()}},
                                                          ensure_ascii=False)}]
        emit(rid, "prompt", {"tag": "final", "system": "(same as textbook) + finaltask", "user": s3_msgs[1]["content"]})
        s3, _ = llm(rid, "final", s3_msgs, temperature=0.6); s3 = s3 or {}
        tb = next((k for k in A.REG if str(k) == str(s3.get("b_tid"))), None)
        if tb is None:
            tb = None
            try:
                _no = int(s3.get("b_no") or 0)
                _all = [c for v in cand_map.values() for c in v["B candidates"]]
                _hit = next((c for c in _all if c.get("b_no") == _no), None)
                if _hit: tb = next((k for k in A.REG if str(k) == _hit["b_tid"]), None)
            except Exception: pass
            if tb is None:
                num = str(s3.get("b_tid", "")).split(" ", 1)[0]
                tb = next((k for k in A.REG if num and str(k).split(" ", 1)[0] == num), None)
        if tb is None:
            emit(rid, "error", {"tag": "final", "msg": f"b_tid not in candidate list:{s3.get('b_tid')}"}); return
        emit(rid, "sel", {"tb": A.REG[tb]["title"], "why": s3.get("why", ""), "gag": None})
        p2, perr = LD.to_plan(s3, ta, tb)
        if p2 is None:
            emit(rid, "error", {"tag": "final", "msg": f"plan rejected:{perr}"}); return
        if p2["exit_tool"] in ("blend16","blendecho","introstack") and not s3.get("_bw_seen"):
            from plan_sim import blend_window_card
            _ovb = 16 if p2["exit_tool"] != "introstack" else int(p2.get("overlap_beats", 16) or 16)
            _bw = blend_window_card(ta, p2["cut"], tb, p2["entry"], _ovb)
            emit(rid, "stage", {"msg": "You selected blend → details card for overlap (harmony/energy/rhythm conversion/bass) sent, please confirm or correct"})
            _m3 = [{"role": "system", "content": GUIDE + PF.BLEND_FEWSHOT + S3_TASK},
                   {"role": "user", "content": json.dumps({"Your previous plan": s3, "Overlap Window Details Card": _bw,
                        "instruction": "Check harmony⑤, energy⑥, fold⑧, low-frequency⑦: can keep original plan, fine-tune anchors/duration, or adjust moves based on this; output complete plan JSON."},
                        ensure_ascii=False)}]
            _s3c, _ = llm(rid, "Overlap Window Check", _m3, temperature=0.5)
            if _s3c:
                _s3c["_bw_seen"] = True
                _p2c, _pe = LD.to_plan(_s3c, ta, tb)
                if _p2c is not None: s3, p2 = _s3c, _p2c
        r1, f1 = sim_c1(ta, p2["cut"], p2["exit_tool"])
        if f1:
            alt = safe_cut_near(ta, p2["cut"], p2["exit_tool"], span=8.0)
            if alt and abs(alt[0] - p2["cut"]) <= 8.0:
                emit(rid, "stage", {"msg": f"final recheck: cue_out {p2['cut']}s still on syllable → snap to nearest safe point {alt[0]}s"})
                p2["cut"] = alt[0]
        from plan_sim import validate_grid as _vg
        for _key, _tid in (("cut", ta), ("entry", tb)):
            _g = _vg(_tid, p2[_key])
            if not _g["ok"] and _g.get("nearest beat") is not None:
                emit(rid, "stage", {"msg": f"final check: {_key}={p2[_key]}s is off the grid, snapped to {_g['nearest beat']}s (anchor {_g['nearest anchor']}s, {_g['anchor offset beats']:+d} beats)"})
                p2[_key] = _g["nearest beat"]
        cb = PF.song_card(tb)
        msgs = [{"role": "system", "content": LD.GUIDE_ADV},
                {"role": "user", "content": json.dumps({"A (outgoing)": ca, "B (incoming)": cb, "technique card": PF.FX_CARD,
                                                        "key advice (computed)": PF.key_advice(A.KEYS.get(ta, {}).get("key"), A.KEYS.get(tb, {}).get("key"))},
                                                       ensure_ascii=False)}]
        pl = dict(s3)
        for r in range(4):
            if r > 0:
                emit(rid, "stage", {"msg": f"— Revision round {r} (thinking live)—"})
                emit(rid, "prompt", {"tag": f"r{r}", "system": "(same as r0 system)", "user": msgs[-1]["content"]})
                pl, _ = llm(rid, f"r{r}", msgs, temperature=0.6)
                if pl is None:
                    emit(rid, "error", {"tag": f"r{r}", "msg": "Model did not output parseable JSON, stopping"}); break
                p2, perr = LD.to_plan(pl, ta, tb)
                if p2 is None:
                    emit(rid, "error", {"tag": f"r{r}", "msg": f"plan rejected:{perr}"})
                    msgs = msgs[:2] + [{"role": "assistant", "content": json.dumps(pl, ensure_ascii=False)},
                                       {"role": "user", "content": json.dumps({"Your plan was rejected": perr,
                                           "instruction": "After fixing this issue, re-output the complete plan JSON (other fields can be reused)"}, ensure_ascii=False)}]
                    continue
            emit(rid, "plan", {"r": r, "plan": p2, "why": (pl or {}).get("why", "")})
            tag = f"{rid}_r{r}"
            emit(rid, "stage", {"msg": "Rendering..."})
            try:
                PF.render_free(f"{A.VR}/live_tmp", dict(ta=ta, tb=tb, plan=p2), tag)
                shutil.rmtree(f"{LD.OUT}/{tag}", ignore_errors=True)
                shutil.move(f"{PF.OUT}/s/{tag}", f"{LD.OUT}/{tag}")
            except Exception as ex:
                emit(rid, "error", {"tag": f"r{r}", "msg": f"Render failed {repr(ex)[:60]}"}); break
            sc, n_fail = LD.scorecard(f"{LD.OUT}/{tag}", ta, tb, p2)
            if LD.SC_STYLE == "family": n_fail = LD.fam_fail(sc)
            emit(rid, "verdict", {"r": r, "scorecard": sc, "n_fail": n_fail,
                                   "audio": f"/live/audio/{tag}/trans1.wav"})
            if n_fail == 0:
                emit(rid, "stage", {"msg": f"Round {r} scoring card is all green, done."}); break
            if r < 3:
                msgs = msgs[:2] + [{"role": "assistant", "content": json.dumps(pl, ensure_ascii=False)},
                                   {"role": "user", "content": json.dumps(
                                       {"Full-dimension scoring card (four tribes)": (LD.fam_feedback(sc) if LD.SC_STYLE == "family" else sc),
                                        "instruction": "Move-first mode: B is already retrieved as final, no title change this round; cue/moves/parameters can be modified. Articulation-type actions cue; energy-type actions adjust parameters or switch moves. Output complete JSON only."},
                                       ensure_ascii=False)}]
    except Exception as e:
        emit(rid, "error", {"tag": "pipeline_tf", "msg": repr(e)[:150]})
    finally:
        LD.RUNS[rid]["done"] = True
        emit(rid, "done", {})
