#!/usr/bin/env python3
"""Plan, render and check one DJ transition.

usage:
  run_transition.py --a "<track id>" --b "<track id>"            # fixed pair (as in Test1)
  run_transition.py --a "<track id>"                             # planner also picks track B
  run_transition.py --a ... --b ... --variant ours-c-r           # ablations

Variants (Sec. 3.2 of the paper):
  ours      reasoning on,  checker-driven repair on   (default)
  ours-c    reasoning on,  checker off
  ours-c-r  reasoning off, checker off

Track ids are the keys of the song registry (the audio file name without its
extension). `--list` prints them. Output goes to $AIDJ_RUNTIME_ROOT/
dj_transition_planner/eval/variants/live_demo/<run id>/: trans1.wav, plan.json
and the checker scorecard of every round.
"""
import argparse, json, os, shutil, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "djustify"))

import song_library as A            # noqa: E402
import transition_core as LD        # noqa: E402
import render_plan as PF            # noqa: E402
from plan_sim import validate_grid, blend_window_card   # noqa: E402

VARIANTS = {"ours": (True, True), "ours-c": (True, False), "ours-c-r": (False, False)}
ROUNDS = 4          # repair rounds when the checker is on


def resolve(name):
    """Accept a full track id, its numeric prefix, or a unique substring."""
    if name in A.REG:
        return name
    num = name.split(" ", 1)[0]
    hit = [k for k in A.REG if str(k).split(" ", 1)[0] == num]
    if len(hit) == 1:
        return hit[0]
    hit = [k for k in A.REG if name.lower() in str(k).lower()]
    if len(hit) == 1:
        return hit[0]
    raise SystemExit(f"track not found or ambiguous: {name!r} ({len(hit)} matches)")


def blend_ok(ta, tb):
    """B has a verse/chorus head whose preceding 16 beats are mostly instrumental: a clean ride window."""
    try:
        beat = 60.0 / A.REG[tb]["bpm"]; voiced = A.VOICED.get(tb, []); dur = A.REG[tb]["duration_sec"]
        for sg in A.REG[tb].get("segments", []):
            if sg["label"] not in ("verse", "chorus"):
                continue
            H = sg["start"]
            if H / dur > 0.55 or dur - H < 45 or H < 16 * beat:
                continue
            t0 = H - 16 * beat
            cov = sum(max(0.0, min(H - 0.3, e) - max(t0, s)) for s, e in voiced)
            if cov / (16 * beat) < 0.30 and any(abs(s - H) < 2.0 for s, e in voiced):
                return True
    except Exception:
        pass
    return False


def technique_constraint(ta, tb):
    if blend_ok(ta, tb):
        return ("This pair has a clean ride window -> exit_tool must come from the ride family: "
                "blendecho / blend16 / loop / loop_in (riding is the DJ's main text; loop and loop_in "
                "take loop_echo to decide whether to throw an echo). Set entry_tool to direct.")
    return ("This pair has no clean ride window -> choose exit_tool from the cut family: echo / brake / "
            "backspin / filter_lpf / filter_hpf (reverb rarely); decide by DJ aesthetics (memorable phrase "
            "ending, narrative), and choose the entry technique independently.")


def plan_pair(rid, ta, tb, reasoning, checker, log):
    ca, cb = PF.song_card(ta), PF.song_card(tb)
    user = {"A (outgoing)": ca, "B (incoming)": cb, "technique card": PF.FX_CARD,
            "technique choice (hard constraint for this pair)": technique_constraint(ta, tb),
            "key advice (computed, just copy it)": PF.key_advice(A.KEYS.get(ta, {}).get("key"),
                                                                A.KEYS.get(tb, {}).get("key"))}
    msgs = [{"role": "system", "content": LD.GUIDE_ADV + PF.BLEND_FEWSHOT},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    result = {"ta": str(ta), "tb": str(tb), "reasoning": reasoning, "checker": checker, "rounds": []}
    for r in range(ROUNDS if checker else 1):
        pl, think = LD.llm_stream(rid, f"r{r}", msgs, temperature=0.6, reasoning=reasoning)
        p2, perr = LD.to_plan(pl or {}, ta, tb)
        if p2 is None and pl is None:            # no JSON at all: one cooler retry
            pl, think = LD.llm_stream(rid, f"r{r}retry", msgs, temperature=0.4, reasoning=reasoning)
            p2, perr = LD.to_plan(pl or {}, ta, tb)
        if p2 is None:
            log(f"round {r}: plan rejected before rendering: {perr}")
            if not checker:
                result["error"] = f"plan: {perr}"
                return result
            msgs = msgs[:2] + [{"role": "assistant", "content": json.dumps(pl, ensure_ascii=False)},
                               {"role": "user", "content": json.dumps({"your plan was returned": perr,
                                "instruction": "Fix this problem and output the complete plan JSON again (other fields may stay)."}, ensure_ascii=False)}]
            continue
        if checker:
            # cues must sit on the beat grid; ask once, then snap
            errs = []
            for key, tid in (("cut", ta), ("entry", tb)):
                g = validate_grid(tid, p2[key])
                if not g["ok"] and g.get("nearest beat") is not None:
                    errs.append({"field": key, "value": p2[key], "off by (s)": g.get("off by s"), "nearest grid point": g["nearest beat"]})
            if errs and not (pl or {}).get("_grid_retried"):
                m2 = msgs[:2] + [{"role": "assistant", "content": json.dumps(pl, ensure_ascii=False)},
                                 {"role": "user", "content": json.dumps({"your plan was returned": errs,
                                  "instruction": "Output cue as {anchor: aN, offset_beats: integer} or copy the nearest grid point; do not do the arithmetic yourself."}, ensure_ascii=False)}]
                pl2, _ = LD.llm_stream(rid, f"r{r}grid", m2, temperature=0.5, reasoning=reasoning)
                if pl2:
                    pl2["_grid_retried"] = True
                    p2b, _ = LD.to_plan(pl2, ta, tb)
                    if p2b is not None:
                        pl, p2 = pl2, p2b
            for key, tid in (("cut", ta), ("entry", tb)):
                g = validate_grid(tid, p2[key])
                if not g["ok"] and g.get("nearest beat") is not None:
                    p2[key] = g["nearest beat"]
            # blend moves get the overlap-window card once
            if p2["exit_tool"] in ("blend16", "blendecho") and not (pl or {}).get("_bw_seen"):
                bw = blend_window_card(ta, p2["cut"], tb, p2["entry"], 16)
                m3 = msgs[:2] + [{"role": "assistant", "content": json.dumps(pl, ensure_ascii=False)},
                                 {"role": "user", "content": json.dumps({"overlap window card": bw,
                                  "instruction": "Check harmony, energy, folded tempo and low end; keep, adjust or change technique; output the complete plan JSON."}, ensure_ascii=False)}]
                pl2, _ = LD.llm_stream(rid, f"r{r}bw", m3, temperature=0.5, reasoning=reasoning)
                if pl2:
                    pl2["_bw_seen"] = True
                    p2b, _ = LD.to_plan(pl2, ta, tb)
                    if p2b is not None:
                        pl, p2 = pl2, p2b
        tag = f"{rid}_r{r}"
        try:
            PF.render_free(f"{A.VR}/live_tmp", dict(ta=ta, tb=tb, plan=p2), tag)
            shutil.rmtree(f"{LD.OUT}/{tag}", ignore_errors=True)
            shutil.move(f"{PF.OUT}/s/{tag}", f"{LD.OUT}/{tag}")
        except Exception as ex:                  # unrenderable plan
            log(f"round {r}: render failed: {ex!r}")
            result["error"] = f"render: {ex!r}"
            return result
        sc, n_fail = LD.scorecard(f"{LD.OUT}/{tag}", ta, tb, p2)
        if LD.SC_STYLE == "family":
            n_fail = LD.fam_fail(sc)
        result["rounds"].append({"round": r, "plan": p2, "why": (pl or {}).get("why", ""), "n_fail": n_fail,
                                 "scorecard": sc, "wav": f"{LD.OUT}/{tag}/trans1.wav"})
        result.update({"plan": p2, "n_fail": n_fail, "green": n_fail == 0, "tag": tag})
        log(f"round {r}: {p2['exit_tool']} cut@{p2['cut']:.1f}s entry@{p2['entry']:.1f}s  failing gates={n_fail}")
        if not checker or n_fail == 0:
            return result
        if r < ROUNDS - 1:
            msgs = msgs[:2] + [{"role": "assistant", "content": json.dumps(pl, ensure_ascii=False)},
                               {"role": "user", "content": json.dumps({
                                   "scorecard (four families)": LD.fam_feedback(sc),
                                   "instruction": ("Cue, technique and parameters may all change. For vocal failures move the cue. "
                                                   "Swapping techniques inside the blend family changes little, so if a blend fails on "
                                                   "energy or vocals, prefer moving cue_in to a lower-energy instrumental entry; fall back "
                                                   "to a cut technique only if B has no instrumental intro. Output the complete JSON only.")},
                                   ensure_ascii=False)}]
    return result        # still failing after ROUNDS: returned flagged, not resampled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", help="outgoing track id")
    ap.add_argument("--b", help="incoming track id; omit to let the planner choose")
    ap.add_argument("--variant", default="ours", choices=sorted(VARIANTS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--list", action="store_true", help="print the track ids in the registry and exit")
    args = ap.parse_args()
    if args.list:
        for k in A.REG:
            print(k)
        return
    if not args.a:
        ap.error("--a is required")
    ta = resolve(args.a)
    rid = args.run_id or f"tr_{int(time.time())}"
    LD.RUNS[rid] = {"events": [], "cond": threading.Condition(), "done": False, "t0": time.time()}
    out_dir = Path(LD.OUT) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{rid}] {m}", flush=True)
    reasoning, checker = VARIANTS[args.variant]
    if args.b:
        tb = resolve(args.b)
        log(f"A={A.REG[ta]['title']}  B={A.REG[tb]['title']}  variant={args.variant}")
        res = plan_pair(rid, ta, tb, reasoning, checker, log)
    else:
        import retrieval as LT
        log(f"A={A.REG[ta]['title']}  free selection (technique-first pipeline)")
        LT.pipeline_tf(rid, str(ta))
        res = {"ta": str(ta), "events": [e for e in LD.RUNS[rid]["events"] if e.get("kind") not in ("think", "out")]}
    LD.RUNS[rid]["done"] = True
    (out_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if res.get("tag"):
        shutil.copy(f"{LD.OUT}/{res['tag']}/trans1.wav", out_dir / "trans1.wav")
        log(f"wrote {out_dir / 'trans1.wav'}  (green={res['green']}, rounds={len(res['rounds'])})")
    elif res.get("error"):
        log(f"no audio: {res['error']}")
    log(f"result: {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
