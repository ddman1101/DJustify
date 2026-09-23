"""LLM client, transition prompts and the rendered-audio scorecard."""
import os, sys, json, re, time, threading, random
import numpy as np
import requests
import soundfile as sf

EV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EV)
import song_library as A
import render_plan as PF
from plan_sim import sim_c1, sim_c1b, sim_c2, sim_c2b, effect_window_beats

EP = os.environ.get("AIDJ_LLM_ENDPOINT", "http://127.0.0.1:8903/v1/chat/completions")
TOKENIZE_EP = EP.rsplit("/v1/chat/completions", 1)[0] + "/tokenize"
LLM_CONTEXT = int(os.environ.get("AIDJ_LLM_CONTEXT", "28672"))
OUT = f"{A.VR}/live_demo"
os.makedirs(OUT, exist_ok=True)
rng = random.Random()

LEGEND = (
    "[Song card fields] hook evidence = a lyric hook shared with track A (the anchor word, the A and B lines "
    "and their timestamps; the hook score rates lyrics only, acoustic fit is your call, and a hook is optional, "
    "but if you use one, put the cue near that line). "
    "V_q = valence percentile (0 to 1, higher is happier). A_q = energy percentile (0 to 1, higher is more intense). "
    "Emotional distance from A = dV + dA: small joins smoothly, large sounds like changing stations. "
    "Vocal intervals = seconds where someone is singing. Sections = structural labels. "
    "Sound events are the four cue anchors: drums enter and bass enters mark where the low end starts, which is "
    "where a beatmatch, a spinup or a blend lands; harmonic loop restart is where the progression returns to its "
    "beginning, so blend16 and introstack aligned there keep both tracks in phase; harmonic change is a chord "
    "change after a long stretch, which reads as a section switch. Pick the task first, then the anchor that suits it. "
    "Each section carries an envelope of four energy points (0 to 1) and a trend: steady, rising, falling, or fading "
    "to silence. The fade start is where energy slides to silence, and it is a no-go zone for cue_out: an effect on a "
    "dying signal leaves an energy vacuum and C8 fails. Cut before the fade while A still has energy, or use another "
    "section of A. A consensus anchor is a point where several independent events coincide (phrase end, section "
    "boundary, harmonic event, bass entry, vocal boundary); the more of them agree, the more it looks like a real DJ "
    "drop. Phrase end plus section boundary plus harmonic event is the golden cue_out, section boundary plus bass "
    "entry the golden cue_in. They are strong candidates, not the only options.")
GUIDE_ADV = (LEGEND +
    "You are a professional pop DJ. What follows is advice, not law: you may overrule any of it, but say why. "
    "1. A folded BPM difference of 6 percent or less is the safest. "
    "2. Keys close on the Camelot wheel sound more consonant. "
    "3. Cutting a chorus in half hurts. "
    "4. The entry technique is chosen independently of the exit: filter_in works after echo or brake as a surfacing "
    "gesture; only avoid an lpf exit into an hpf entry. "
    "5. A spinup should land where drums and bass come straight back in. "
    "6. Beyond one semitone of keysync the voice sounds artificial. "
    "7. Check whether B sings immediately at cue_in: if it does, do not muffle it with a filter; direct is allowed, "
    "but then the exit on A needs an announcing effect (echo, brake, backspin) in front of it, because a smooth "
    "sweep straight into a vocal is abrupt. "
    "8. Choosing a technique is a score-reading task and every technique deserves the same hearing: the exit moves "
    "(echo, reverb, brake, backspin, blendecho, blend16, loop, loop_in, filter_lpf, filter_hpf) and the four entry "
    "moves (direct, spinup, filter_in, filter_in_hpf) are all candidates. Ask which one this particular score makes "
    "memorable before you ask whether it is feasible, and do not fall back on echo out, direct in out of habit: "
    "direct is a non-gesture, and it is only the best choice when A has already made an announcing move or B opens "
    "on a strong hook. "
    "9. When a clean window exists, the ride family (blendecho, blend16, loop, loop_in) beats the sequential moves; "
    "that is a working DJ's main ride. blendecho rides A's chorus over B's intro and throws an echo on the last "
    "syllable. blend16 crosses A over B's intro for sixteen beats with no echo. loop holds a section of A while B "
    "fades up. loop_in loops B's intro as a bed and holds it until A's phrase ends, which is the longest build-up. "
    "loop and loop_in take loop_echo (true or false) to decide whether to throw an echo on release, and the engine "
    "sets their cues, so the cues in the plan are advisory. "
    "10. blend16, blendecho and introstack are structural parallel moves that handle B's entry themselves, so "
    "entry_tool is direct. The standard blend16 score is A's bridge or instrumental section over B's intro, with "
    "cue_in at the head of B's verse or chorus, so that B arrives exactly as the intro ends. blendecho rides A's "
    "chorus over B's intro for sixteen beats with A sung in full, throws the echo on the last line of that chorus, "
    "and hands over to B's verse, so cue_out is the end of A's last chorus line. The echo may land on the vocal "
    "itself rather than waiting for silence, which is usually the stronger gesture, but it must be the last word of "
    "a phrase or section: never cut mid-sentence or mid-word. cue_in is the head of B's verse and the sixteen beats "
    "before it must be instrumental."
    + PF.RENDER_CONTRACT +
    " task: Design a complete transition plan for this pair of titles, cue is fully flexible. Output only JSON:"
    '{"cue_out":{"anchor":"a7","offset_beats":0},"cue_in":{"anchor":"b3","offset_beats":0},"exit_tool":"...","entry_tool":"...","exit_len_beats":n,'
    '"echo_delay_beats":x,"entry_advance_beats":x,"overlap_beats":n,"keysync":-1..1,'
    'loop_echo: true/false (loop/loop_in only, whether to drop echo), bpm_align: native|micro, why: two sentences'
    "Reminder (highest priority): Follow 17 execute blend if feasible overlap window exists (check A sections bridge/instrumental/outro vs B intro instrumental lead)"
    "cut type only allowed after why explains which windows checked and why failed")
MEAN = {"C1": ("cut while A is still singing (cut mid-sentence)", "Move cue_out to after the vocal sentence end — only moving the cut point is valid; adjusting effectparameters is invalid"),
        "C1d": ("blend cue_out cuts mid-phrase A (vocals span cut) - A cuts off abruptly, sounds like A vanishes (blend echo fades A, C1 does not see this issue)", "Move cue_out to the last character of phrase A (★blendexit anchors), letting A finish singing before handing over"),
        "C2d": ("cue_in cuts mid-B phrase (B already singing before entry) - not entering from clean phrase start, sounds like inserting B title mid-way", "Move cue_in to the start of phrase B (the beat where the section begins, preceded by instrumental/rest; ★blendentry anchors)"),
        "C2": ("at entry moment B is halfway through singing (mid-phrase entry)", "Move cue_in to before the sentence start or a non-vocal section"),
        "C2b": ("During riding/parallel window, A and B vocals play simultaneously (duet)", "Enter B on an instrumental window, shorten the overlap, or end A's vocal earlier; for a blend, the sixteen beats before B must be instrumental"),
    "C1c": ("effect source window is nearly silent (abrupt stop/reverse/feedback have no material to process)", "Move cue_out to energetic part of material (end of chorus/while drums still playing), or switch echo→direct cut"),
    "C6b": ("direct entry but B head energy only fraction of A seam energy cliff creates hollow feel", "Switch to filter_in/spinup (rising from underwater is designed), or B chooses a stronger entry point"),
    "C6c": ("riding blend B verse main section energy overwhelms A entrance too strong A drowned", "blend family internal swap invalid (effects similar) -- use [swap title to keep blend]: replace library with B (exclude current, same title logic), or move cue_in to B lower energy instrumental intro"),
    "C6d": ("The blend leaves cue_in inside B's intro, so B's verse or chorus never takes over and B sounds like an intro only", "Move cue_in to the first beat of B's verse or chorus so the body of B takes over; if there is no clean verse head, change track"),
    "C9": ("A hands off after a few seconds/still in intro (entry track too short, total failure)", "Shift cue_out later: must play at least one main body segment (≥45s and past 20% of track length), not within intro section"),
    "C2c": ("B starts speaking while the tail of the exit effect (echo feedback/reverb/handover window) is still ringing", "Side B: change to an entry point before vocals ≥ B needs (instrumental start/drums enter), or reduce echo_delay, switch to brake/backspin"),
        "C1b": ("The processing window of the exit effect (filter sweep/brake stop/backspin) covers the section where A is still singing—effect timing is correct, length is wrong: sweep muffles vocals, stop slows the sentence, reverse plays the sentence backwards", "Move cue_out to the end of the phrase + after effectlength (so the effect lands on the instrumental tail or last word), or reduce exit_len_beats, or switch echo (tail word is its design)."),
        "C3": ("Too quiet near the seam", "Reduce exit_len_beats / B enters earlier / add overlap"),
        "C5": ("There is an energy dip at the seam of consecutive moves", "Add overlap to allow energy handover between the two tracks, or switch to sequence-style techniques"),
        "C3b": ("Sequence-style moves: seam continuous dead zones are too long (between effect tail and B becoming audible)", "shrink exit_len_beats / move cue_in of B to start position / entry_advance_beats advance start"),
        "C4": ("Two songs V/A emotional distance too far (like changing stations)", "Switch to a B with closer V_q/A_q (this track can only be fixed by changing the title)"),
        "C5b": ("Ride-type transition silence exceeds 1 second", "Overlap or move cue to dense phrase sections"),
        "C6": ("cue_in lands too late in B, so B ends before it has told its part of the story", "Move cue_in into the first part of B, at a section head before the chorus, or change track"),
        "C7": ("Selected title deviates from the V/A target range for this story segment", "Switch to a song marked true within the target range (this track can only be fixed by changing the title)"),
        "C8": ("cut falls in A fade-out zone exit effect on dying signal leaves energy vacuum", "Move cue_out before fade start (handover while A still has energy), or switch to A section")}
FAMS = [("V vocals intact", ["C1 exit cuts a sung phrase", "C1d blend exit mid-phrase", "C1b effect window over vocal", "C2 entry cuts a sung phrase", "C2d entry mid-phrase", "C2b both vocals in overlap", "C2c effect tail over B vocal"], "per-rule"),
        ("E energy present", ["C1c effect has no material", "C6b entry energy", "C6c B buries A"], "per-rule"),
        ("E seam energy", ["C3 seam gap", "C3b continuous silence", "C5 energy hole", "C5b dead overlap", "C8 cut inside a fade"], "family"),
        ("P pairing", ["C4 emotion distance", "C7 arc direction"], "family"),
        ("S structure", ["C6 entry runway", "C6d B never reaches its verse", "C9 exit runway"], "family")]
SC_STYLE = os.environ.get("SC_STYLE", "family")

def fam_nest(sc):
    out = {}
    for name, rules, _ in FAMS:
        grp = {r: sc[r] for r in rules if r in sc}
        if grp:
            out[name] = grp
    return out

def fam_feedback(sc):
    """Slimmed version for the model: keep only verdict per item; for FAIL, additionally provide fix (no measured/threshold numbers)."""
    out = {}
    for name, rules, _ in FAMS:
        grp = {}
        for r in rules:
            if r not in sc: continue
            v = sc[r]
            if v.get("verdict") == "FAIL":
                grp[r] = {"verdict": "FAIL", "fix": v.get("fix", "")}
            else:
                grp[r] = v.get("verdict", "N/A")
        if grp:
            out[name] = grp
    return out

def fam_fail(sc):
    n = 0
    for name, rules, cnt in FAMS:
        fails = [r for r in rules if sc.get(r, {}).get("verdict") == "FAIL"]
        n += len(fails) if cnt == "per-rule" else (1 if fails else 0)
    return n

QUIET_OK={"brake","backspin","echo"}; CONT_REQ={"blend16","introstack","filter_hpf","filter_lpf","reverb"}
RIDE={"filter_lpf","filter_hpf","blend16","introstack","harmonic"}

RUNS = {}   # run_id -> {"events":[...], "cond":Condition, "done":bool}
def emit(rid, kind, data):
    r = RUNS[rid]
    with r["cond"]:
        r["events"].append({"kind": kind, "data": data, "t": round(time.time() - r["t0"], 1)})
        r["cond"].notify_all()

def llm_stream(rid, tag, messages, max_tokens=12000, temperature=0.6,
               reasoning=True, json_only=False, json_schema=None):
    """Streaming LLM: thinking token emitted in real-time; returns (parsed_json, full_think). reasoning=False disables thinking (V0 used)."""
    try:
        est = len(requests.post(TOKENIZE_EP,
                                json={"content": "".join(m["content"] for m in messages)}, timeout=30).json()["tokens"])
    except Exception:
        est = sum(int(len(m["content"]) * 0.8) for m in messages)
    room = LLM_CONTEXT - est - 640
    max_tokens = max(768, min(max_tokens, room))
    if room < 2048:
        emit(rid, "stage", {"msg": f"⚠ prompt {est}tok approaching ctx, generation quota only {max_tokens}——card too large"})
    think, body = [], []
    try:
        pp = 1.5 if temperature >= 0.9 else 0.0
        _body = {"messages": messages, "max_tokens": max_tokens,
                 "temperature": temperature, "top_p": 0.95, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": pp, "stream": True}
        if not reasoning: _body["chat_template_kwargs"] = {"enable_thinking": False}
        if json_schema is not None:
            _body["json_schema"] = json_schema
        elif json_only:
            _body["json_schema"] = {"type": "object"}
        _read_timeout = float(os.environ.get("AIDJ_LLM_READ_TIMEOUT", "1800"))
        with requests.post(EP, json=_body, stream=True, timeout=_read_timeout) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "): continue
                s = line[6:].decode()
                if s.strip() == "[DONE]": break
                try: d = json.loads(s)["choices"][0]["delta"]
                except Exception: continue
                rc = d.get("reasoning_content") or ""
                ct = d.get("content") or ""
                if rc:
                    think.append(rc); emit(rid, "think", {"tag": tag, "tok": rc})
                if ct:
                    body.append(ct); emit(rid, "out", {"tag": tag, "tok": ct})
    except Exception as e:
        emit(rid, "error", {"tag": tag, "msg": repr(e)[:100]}); return None, ""
    bt = "".join(body)
    if not think and "</think>" in bt:
        th = bt.split("</think>")[0].replace("<think>", ""); bt = bt.split("</think>")[-1]
        think = [th]
    try:
        parsed = json.loads(bt.strip())
        if isinstance(parsed, dict):
            return parsed, "".join(think)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    parsed_objects = []
    for pos, char in enumerate(bt):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(bt, pos)
        except Exception:
            continue
        if isinstance(obj, dict):
            parsed_objects.append((end - pos, obj))
    if parsed_objects:
        return max(parsed_objects, key=lambda item: item[0])[1], "".join(think)
    _tl, _bl = sum(len(x) for x in think), len(bt)
    emit(rid, "stage", {"msg": f"{tag} returned no JSON: {_tl} thinking chars, {_bl} body chars; body tail: {bt[-160:] if bt else '(empty: the thinking may have used the whole budget)'}"})
    print(f"NOJSON[{tag}] think={_tl} body={_bl} tail={bt[-200:]!r}", flush=True)
    return None, "".join(think)

W_RMS, W_VA, W_GAG, W_TITLE, W_BLEND, W_STAB = 0.40, 0.35, 0.45, 0.15, 0.50, 0.30

_BLENDFAM = ("blendecho", "blend16", "loop_in", "loop", "introstack")

def _a_last_word_end(ta, s):
    """A: last vocal offset in a segment (= end of last word); if no vocals, return to segment end."""
    voiced = A.VOICED.get(ta, [])
    ends = [min(e, s["end"]) for st, e in voiced if e > s["start"] and st < s["end"]]
    return max(ends) if ends else s["end"]

def _enforce_exit(ta, cut, tool):
    """Hard constraint on cue_out: it must sit past A's first chorus.

    For a blend it is the last word of a later chorus, snapped to A's grid, so
    the echo throws that word and B sings on. Otherwise it is a later phrase
    end. Returns (cut, moved)."""
    segs = A.REG[ta].get("segments") or []
    dur = A.REG[ta].get("duration_sec") or (segs[-1]["end"] if segs else 0)
    downs = A.REG[ta].get("downbeat_times") or []
    ch = next((s for s in segs if s["label"] == "chorus"), None)
    min_t = ch["end"] if ch else dur * 0.45
    ok = ("verse", "chorus", "bridge", "inst", "solo")
    is_blend = str(tool) in _BLENDFAM
    if is_blend:
        chs = [s for s in segs if s["label"] == "chorus" and s["start"] >= min_t - 1 and s["end"] < dur - 1]
        if not chs:
            chs = [s for s in segs if s["label"] in ok and s["start"] >= min_t - 1 and s["end"] < dur - 1]
        if not chs:
            return round(min(cut, dur - 1), 2), False
        tgt = min(chs, key=lambda s: abs(s["end"] - cut))
        ne = _a_last_word_end(ta, tgt)
        if downs:
            ne = min(downs, key=lambda d: abs(d - ne))
        return round(ne, 2), abs(ne - cut) > 0.15
    seg = next((s for s in segs if s["start"] <= cut < s["end"]), None)
    if cut >= min_t and seg and seg["label"] in ok:
        return round(cut, 2), False
    cands = [s["end"] for s in segs if s["end"] >= min_t and s["label"] in ok and s["end"] < dur - 1]
    if not cands:
        cands = [min(min_t, dur - 1)]
    ne = min(cands, key=lambda t: abs(t - cut))
    if downs:
        ne = min(downs, key=lambda d: abs(d - ne))
    return round(ne, 2), True

def _enforce_blend_entry(tb, entry):
    """blend cue_in=handoff point=B verse/chorus start, and [entry-16beat, entry] must be an instrumental intro without vocals.
    Return (entry, moved, feasible). feasible=False means this B has no clean entry point (switch B)."""
    segs = A.REG[tb].get("segments") or []
    voiced = A.VOICED.get(tb, [])
    beat = 60.0 / (A.REG[tb].get("bpm") or 120)
    win = 16 * beat
    def vcov(t0, t1):
        ov = sum(max(0, min(t1, e) - max(t0, s)) for s, e in voiced if e > t0 and s < t1)
        return ov / max(1e-6, t1 - t0)
    heads = [s["start"] for s in segs if s["label"] in ("verse", "chorus", "hook", "drop")]
    valid = [h for h in heads if h - win >= 0 and vcov(h - win, h) < 0.15]
    if not valid:
        return round(entry, 2), False, False
    near = min(valid, key=lambda h: abs(h - entry))
    return round(near, 2), (abs(near - entry) > 0.15), True

def to_plan(pl, ta, tb, validate=True):
    from plan_sim import resolve_cue
    _co, _e1 = resolve_cue(ta, pl.get("cue_out"))
    _ci, _e2 = resolve_cue(tb, pl.get("cue_in"))
    if _e1 or _e2:
        return None, "cue anchor parsing failed:" + ";".join(x for x in (_e1, _e2) if x) + "(re-fill using valid IDs from A/B anchor table)"
    pl = dict(pl); pl["cue_out"], pl["cue_in"] = _co, _ci
    try:
        p2 = dict(cut=float(pl["cue_out"]), entry=float(pl["cue_in"]),
                  exit_tool=str(pl["exit_tool"]), entry_tool=str(pl.get("entry_tool", "direct")),
                  exit_len_beats=int(pl.get("exit_len_beats", 8) or 8),
                  echo_delay_beats=float(pl.get("echo_delay_beats", 0.75) or 0.75),
                  entry_advance_beats=float(pl.get("entry_advance_beats", 0) or 0),
                  overlap_beats=float(pl.get("overlap_beats", 0) or 0),
                  keysync=int(np.clip(int(pl.get("keysync", 0) or 0), -2, 2)),
                  loop_echo=bool(pl.get("loop_echo", False)),
                  b_ride=float(np.clip(float(pl.get("b_ride", 0.7) or 0.7), 0.3, 1.0)))
        if p2["entry_tool"] == "direct": p2["entry_tool"] = "__direct__"
    except Exception:
        return None, "JSON field missing or type error (cue_out/cue_in/exit_tool required)"
    da, db = A.REG[ta]["duration_sec"], A.REG[tb]["duration_sec"]
    if not (5 < p2["cut"] < da - 1):
        return None, f"cue_out={p2['cut']:.1f}s out of bounds (A length {da:.0f}s, must be 5~{da-1:.0f}s)"
    if not (0 <= p2["entry"] < db - 35):
        return None, (f"cue_in={p2['entry']:.1f}s is too close to the end of B (B is {db:.0f}s and the render needs at least 35s of material, "
                      f"Must < {db-35:.0f}s; advice: connect to B front segment, see C6 entry runway)")
    if p2["exit_tool"] in _BLENDFAM:
        _da = A.REG[ta].get("downbeat_times") or []
        _db = A.REG[tb].get("downbeat_times") or []
        if _da: p2["cut"] = round(min(_da, key=lambda d: abs(d - p2["cut"])), 2)
        if _db: p2["entry"] = round(min(_db, key=lambda d: abs(d - p2["entry"])), 2)
        _beatB = 60.0 / (A.REG[tb].get("bpm") or 120)
        if validate and p2["entry"] - 16 * _beatB < 0:
            return None, (f"blend entry point {p2['entry']:.1f}s too early in B: insufficient 16 beats ahead (need ≥ {16*_beatB:.0f}s),"
                          f"Overlap window is empty and cannot overlap—select a later entry point with 16 beats of instrumentation before it (anchor marked blend entry)")
        _, _, _feas = _enforce_blend_entry(tb, p2["entry"])
        p2["_blend_feasible"] = _feas
    return p2, None

def scorecard(out_dir, ta, tb, plan, mode="pair", skip_c4=False):
    cut, entry, tool = plan["cut"], plan["entry"], plan["exit_tool"]
    sc = {}
    r1, f1 = sim_c1(ta, cut, tool)
    sc["C1 exit cuts a sung phrase"] = {"means": MEAN["C1"][0], "fix": MEAN["C1"][1], "measured": f"{r1:.2f}x",
                      "threshold": "0.40x", "verdict": "FAIL" if f1 else "PASS"}
    if str(tool) in _BLENDFAM:
        _vc = A.VOICED.get(ta, [])
        _after = sum(max(0, min(cut + 1.5, e) - max(cut, s)) for s, e in _vc if e > cut and s < cut + 1.5) / 1.5
        sc["C1d blend exit mid-phrase"] = {"means": MEAN["C1d"][0], "fix": MEAN["C1d"][1],
                                 "measured": f"cut 1.5s after A vocals {_after:.0%}", "threshold": "< 50% (should stop after the phrase end)",
                                 "verdict": "FAIL" if _after > 0.5 else "PASS"}
    _vcb = A.VOICED.get(tb, [])
    _before = sum(max(0, min(entry, e) - max(entry - 1.5, s)) for s, e in _vcb if e > entry - 1.5 and s < entry) / 1.5
    sc["C2d entry mid-phrase"] = {"means": MEAN["C2d"][0], "fix": MEAN["C2d"][1],
                        "measured": f"B vocals {_before:.0%} in the 1.5s before entry", "threshold": "< 50% (should be clean before the section start)",
                        "verdict": "FAIL" if _before > 0.5 else "PASS"}
    if effect_window_beats(tool, plan.get("exit_len_beats", 8)) > 0:
        rb_, fb_ = sim_c1b(ta, cut, tool, plan.get("exit_len_beats", 8))
        sc["C1b effect window over vocal"] = {"means": MEAN["C1b"][0], "fix": MEAN["C1b"][1], "measured": f"Vocals in window {rb_:.0%}",
                            "threshold": "< 50%", "verdict": "FAIL" if fb_ else "PASS"}
    _s2b, _f2b = sim_c2b(ta, cut, tb, entry, tool, plan.get("overlap_beats", 0))
    if str(tool) in ("blend16", "blendecho", "introstack") or (str(tool).startswith("filter_") and (plan.get("overlap_beats") or 0) > 0):
        sc["C2b both vocals in overlap"] = {"means": MEAN["C2b"][0], "fix": MEAN["C2b"][1], "measured": f"Intersection {_s2b:.1f}s",
                             "threshold": "≤ 1.0s", "verdict": "FAIL" if _f2b else "PASS"}
    r2, f2 = sim_c2(tb, entry)
    sc["C2 entry cuts a sung phrase"] = {"means": MEAN["C2"][0], "fix": MEAN["C2"][1], "measured": f"{r2:.2f}x",
                      "threshold": "0.50x", "verdict": "FAIL" if f2 else "PASS"}
    from plan_sim import sim_c2c, b_lead_need_beats, sim_c1c, sim_c6b, sim_c9
    _w9, _f9 = sim_c9(ta, cut)
    sc["C9 exit runway"] = {"means": MEAN["C9"][0], "fix": MEAN["C9"][1],
                      "measured": _w9 or f"cut@{cut:.0f}s OK", "verdict": "FAIL" if _f9 else "PASS"}
    _r1c, _f1c = sim_c1c(ta, cut, tool, plan.get("exit_len_beats", 4))
    if str(tool) in ("brake", "backspin", "echo"):
        sc["C1c effect has no material"] = {"means": MEAN["C1c"][0], "fix": MEAN["C1c"][1],
                              "measured": f"{_r1c:.2f}× reference", "threshold": "≥ 0.35×", "verdict": "FAIL" if _f1c else "PASS"}
    _r6b, _f6b = sim_c6b(ta, cut, tb, entry, plan.get("entry_tool"), plan.get("overlap_beats", 0),
                         tool, plan.get("exit_len_beats", 4))
    if str(plan.get("entry_tool") or "").strip("_") in ("", "direct") and not (plan.get("overlap_beats") or 0):
        sc["C6b entry energy"] = {"means": MEAN["C6b"][0], "fix": MEAN["C6b"][1],
                           "measured": f"B head {_r6b:.2f}× A reference", "threshold": "≥ 0.25×", "verdict": "FAIL" if _f6b else "PASS"}
    if str(tool) in ("blendecho", "blend16", "loop_in", "loop", "introstack"):
        from plan_sim import _mix_rms
        _ea6 = _mix_rms(ta, cut - 3.0, cut) or 1e-6
        _eb6 = _mix_rms(tb, entry, entry + 3.0) or 1e-6
        _db6 = 20.0 * (np.log10(_eb6) - np.log10(_ea6))
        sc["C6c B buries A"] = {"means": MEAN["C6c"][0], "fix": MEAN["C6c"][1],
                               "measured": f"B entry is {_db6:+.1f}dB higher than A exit", "threshold": "≤ +2.5dB",
                               "verdict": "FAIL" if _db6 > 2.5 else "PASS"}
    if str(tool) in ("blendecho", "blend16", "loop_in", "loop"):
        _segB = A.REG.get(tb, {}).get("segments", [])
        _lblB = next((s["label"] for s in _segB if s["start"] <= entry < s["end"]), "?")
        sc["C6d B never reaches its verse"] = {"means": MEAN["C6d"][0], "fix": MEAN["C6d"][1],
                            "measured": f"Handoff point B falls on '{_lblB}'", "threshold": "verse/chorus/hook (verse takes over)",
                            "verdict": "PASS" if _lblB in ("verse", "chorus", "hook", "drop") else "FAIL"}
    _g2c, _n2c, _f2c = sim_c2c(ta, cut, tb, entry, tool, plan.get("entry_tool"),
                               plan.get("echo_delay_beats", 1.0), plan.get("overlap_beats", 0))
    if _n2c > 0:
        _blend_echo = str(tool) in ("blendecho", "blend16", "loop_in", "loop", "introstack")
        sc["C2c effect tail over B vocal"] = {"means": MEAN["C2c"][0], "fix": MEAN["C2c"][1],
                               "measured": f"Gap before vocals {_g2c:.1f}s" + ("(blend: A tail echo overlaps B intro vocals = design)" if _blend_echo else ""),
                               "threshold": f"≥ {_n2c:.1f}s (footprint table)",
                               "verdict": "PASS" if _blend_echo else ("FAIL" if _f2c else "PASS")}
    va, vb = PF.VAM.get(ta, {}), PF.VAM.get(tb, {})
    if skip_c4:
        sc["C4 emotion distance"] = {"verdict": "N/A", "reason": "story mode: pairing taste is left to the C7 curve verdict (the curve can require emotional jumps)"}
    elif va and vb:
        dv = abs(va["valence_q"] - vb["valence_q"]); da = abs(va["arousal_q"] - vb["arousal_q"])
        f4 = dv + da > 0.9 or dv > 0.55
        sc["C4 emotion distance"] = {"means": MEAN["C4"][0], "fix": MEAN["C4"][1],
                          "measured": f"ΔV={dv:.2f} ΔA={da:.2f}", "threshold": "ΔV+ΔA≤0.9 and ΔV≤0.55",
                          "verdict": "FAIL" if f4 else "PASS"}
    y, sr = sf.read(f"{out_dir}/trans1.wav")
    if y.ndim > 1: y = y.mean(1)
    tj = 30.0; beat = A.beat_of(ta)
    seg = y[max(0, int((tj - 4 * beat) * sr)):int((tj + 12 * beat) * sr)]
    w = int(0.15 * sr); n = len(seg) // w
    if n > 4:
        e = np.sqrt((seg[:n * w].reshape(n, w) ** 2).mean(1))
        quiet_s = float((e < 0.25 * e.mean()).sum() * 0.15)
        lim = 1.8 if tool in QUIET_OK else 1.2
        sc["C3 seam gap"] = {"means": MEAN["C3"][0], "fix": MEAN["C3"][1], "measured": f"{quiet_s:.1f}s",
                          "threshold": f"{lim}s", "verdict": "FAIL" if quiet_s > lim else "PASS"}
        if tool in QUIET_OK:
            qb = e < 0.25 * e.mean()
            run = longest = 0
            for b in qb:
                run = run + 1 if b else 0
                longest = max(longest, run)
            ls = longest * 0.15
            sc["C3b continuous silence"] = {"means": MEAN["C3b"][0], "fix": MEAN["C3b"][1], "measured": f"{ls:.1f}s",
                               "threshold": "1.0s", "verdict": "FAIL" if ls > 1.0 else "PASS"}
        if tool in CONT_REQ:
            mn = float(e.min() / e.mean())
            sc["C5 energy hole"] = {"means": MEAN["C5"][0], "fix": MEAN["C5"][1], "measured": f"min {mn:.2f}x",
                              "threshold": "0.22x", "verdict": "FAIL" if mn < 0.22 else "PASS"}
    if tool in RIDE:
        hop = int(0.05 * sr); seg2 = y[:int(32 * sr)]; n2 = len(seg2) // hop
        if n2 > 40:
            e2 = np.sqrt((seg2[:n2 * hop].reshape(n2, hop) ** 2).mean(1))
            db = 20 * np.log10(np.maximum(e2, 1e-6)); med = float(np.median(db))
            t2 = np.arange(n2) * 0.05
            zone = (t2 >= tj - 2) & (t2 <= tj + 8)
            below = db[zone] < med - 30
            longest = run = 0
            for b in below:
                run = run + 1 if b else 0; longest = max(longest, run)
            qs = longest * 0.05
            sc["C5b dead overlap"] = {"means": MEAN["C5b"][0], "fix": MEAN["C5b"][1], "measured": f"{qs:.1f}s",
                               "threshold": "1.0s", "verdict": "FAIL" if qs > 1.0 else "PASS"}
    dur_b = A.REG[tb].get("duration_sec")
    if dur_b:
        runway = dur_b - entry; pos = entry / dur_b
        if mode == "short":
            f6 = runway < 20
            th = "Remaining ≥ 20s (short set mode: allows deep extraction of chorus titles)"
        else:
            f6 = runway < 45 or pos > 0.55
            th = "entry ≤ 55% and remaining ≥ 45s"
        sc["C6 entry runway"] = {"means": MEAN["C6"][0], "fix": MEAN["C6"][1],
                          "measured": f"entry at {pos*100:.0f}% of B, {runway:.0f}s remaining",
                          "threshold": th, "verdict": "FAIL" if f6 else "PASS"}
    _fs = (PF.SEGENV.get(str(ta)) or {}).get("fade_start")
    if _fs:
        _da = A.REG[ta].get("duration_sec") or 1e9
        f8 = cut > _fs + 1.0 and cut < _da - 2.0
        sc["C8 cut inside a fade"] = {"means": MEAN["C8"][0], "fix": MEAN["C8"][1],
                            "measured": f"fade start {_fs:.0f}s / cut {cut:.0f}s",
                            "threshold": "cut ≤ fade start (or play to end)", "verdict": "FAIL" if f8 else "PASS"}
    NA = {"C2b both vocals in overlap": "Sequence-style transitions: A and B do not play simultaneously, no riding window",
          "C2c effect tail over B vocal": "This move has no B-end footprint requirement (brake/backspin stop is silent) or the tail does not contain vocals",
          "C1b effect window over vocal": "echo/reverb/blend series: effects do not process unfinished phrases (catching the tail/wet tail/parallel processing is by design)",
          "C3 seam gap": "Seam window insufficient, measurement skipped",
          "C3b continuous silence": "Non-sequential move (continuous dead zones gated by C5/C5b)",
          "C4 emotion distance": "Missing V/A data",
          "C5 energy hole": "Sequence-style moves: seam gaps are designed semantics, do not check for dead holes",
          "C5b dead overlap": "Non-riding move, no riding phase",
          "C6 entry runway": "Missing songlength data",
          "C8 cut inside a fade": "A: no fade-out tail (natural ending), no fade zone"}
    for k, why in NA.items():
        if k not in sc:
            sc[k] = {"verdict": "N/A", "reason": why}
    ORDER = ["C9 exit runway", "C1 exit cuts a sung phrase", "C1d blend exit mid-phrase", "C1b effect window over vocal", "C2 entry cuts a sung phrase", "C2d entry mid-phrase", "C2b both vocals in overlap", "C2c effect tail over B vocal", "C1c effect has no material", "C6b entry energy", "C6c B buries A", "C3 seam gap", "C3b continuous silence",
             "C4 emotion distance", "C5 energy hole", "C5b dead overlap", "C6 entry runway", "C6d B never reaches its verse", "C8 cut inside a fade"]
    sc = {k: sc[k] for k in ORDER if k in sc}
    return sc, sum(1 for v in sc.values() if v["verdict"] == "FAIL")
