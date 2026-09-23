# -*- coding: utf-8 -*-
"""render_plan:0 rules are clean; planner — all knowledge is in prompt, code performs no taste filtering."""
import os, sys, json, random
import numpy as np

EV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EV)
import song_library as A
import render_dsp as M
import regression_check as RC

OUT = f"{A.VR}/earlab"
CUE = json.load(open(f"{A.ROOT}/dj_transition_planner/outputs/cuedetr.json"))
try:
    CUEEV = json.load(open(f"{A.ROOT}/dj_transition_planner/outputs/cue_events.json"))
except Exception:
    CUEEV = {}
try:
    SEGENV = json.load(open(f"{A.ROOT}/dj_transition_planner/outputs/seg_env.json"))
except Exception:
    SEGENV = {}
try:
    CONS = json.load(open(f"{A.ROOT}/dj_transition_planner/outputs/cue_consensus.json"))
except Exception:
    CONS = {}
VAM = RC.VAM
rng = random.Random()

GUIDE = (
    "You are a professional pop DJ. What follows is advice, not law: overrule any of it on musical grounds, but say why in the reason."
    "① Emotional continuity: Connect tracks with similar V_q/A_q for smoothness; large jumps feel like changing stations. ② BPM: Most stable if difference (after ×2/÷2 conversion) ≤6%,"
    "The greater the difference, the more it tests technique; sequence-style transitions allow B to enter at original tempo, just hit the beat points (changing tempo for the whole track damages sound quality and emotion);"
    "③ Close key circle distance sounds more harmonious. ④ Exit point: select a position after a phrase ends and there is a rest. ⑤ Entry point: select the first beat of a section start;"
    "⑤ Chorus cut hurts;⑥ filter_in and cue_out independent:filter_in(B muffled to full) follows echo/brake/backspin as surfacing gesture,no need pair with filter cue_out;avoid lpf out hpf in spectrum clash;"
    "⑦ Exit points at phrase ends followed by rests are highlight moments for echo/brake/backspin (reverb like shouting into a valley, memorable; using filter here wastes the score);"
    "⑧ B entry followed by drum and bass,spinup cool(landing before quiet intro like power cut);"
    "⑨ keysync:card set includes key advice(machine calculates semitone shift for same key/fifth circle),follow advice;>±1 semitone voice fake;⑪ sequence semantics:echo/reverb/brake/backspin are sequential —"
    "B enters only after effect finishes (exit_len_beats determines duration), the only exception is brake/backspin can use entry_advance_beats"
    "Have B start the transition early before the stop; filter overlap_beats>0 indicates a riding style — overlap_beats determines the parallel width of the two tracks;"
    "⑩ Check B cue_in for immediate vocal start(vocal range):if yes no filter muffle vocal,"
    "Direct is allowed, but the A-side entrance must be preceded by a distinctive special effect (echo/brake/backspin) to set the stage—a smooth sweep directly into the vocal will feel abrupt;"
    "⑫ Selecting moves is like reading sheet music, and moves have two tiers; you must exhaust Tier 1 before falling back to Tier 2:"
    "Tier 1, the ride and structural moves, chosen whenever they are possible, always with entry_tool direct and no entry move of their own: blendecho, blend16, loop_in, loop, introstack. A and B run in parallel and the move handles B's entry itself. This is a working DJ's main ride. "
    "【Tier 2 Punctuation/Rescue Moves (Only provided if all five Tier 1 moves fail on this score】=echo / reverb / brake / backspin / filter_lpf / filter_hpf, can pair entry direct/spinup/filter_in/filter_in_hpf;"
    "Selection process: check if each node fits Tier 1 5 moves (see B needs); if yes, pick the best storytelling move, set entry to direct; if all 5 fail (vocal duet window/no loopable segment/A no clean phrase end/BPM diff too large for full stop), downgrade to Tier 2 and explain why Tier 1 failed per move in why;"
    "Structural Tier 1 is the main ride as long as the window is clean (A tail 16 beats and B intro has no vocal duet), far prioritized over any Tier 2;"
    "filter sweep is safest but least memorable; using it throughout the track feels like an auto-crossfader—only choose it when the score truly has no better option;"
    "blend16/blendecho/introstack are structural parallel moves (they handle B entry themselves, entry_tool set to direct):"
    "blend16 standard chart = A bridge overlaps B intro, cue_in selects B verse or chorus start;"
    "blendecho=A chorus rides B intro, tail echoes out; cue_in=B verse head (first 16 bars must be instrumental);"
    "⑬ Sound events are your transition anchors; use them for move selection: drums enter/bass enters = low-frequency heartbeat begins (beatmatch, spinup landing point, overlay title must align with the heartbeat);"
    "harmonic loop restart=harmony returns to start (blend16/introstack cues align here two songs harmonic phase aligned not muddy also instrumental section phrase end)"
    "harmonic change = chord change after a long pause (strong section feel, good landing spot for switch/impact); vocal interval boundary = vocal cue and hazard."
    "First consider what task this cue needs to accomplish, then select the corresponding type of anchor; the move and anchor must match."
    "⑭ sections envelope (4-point energy within section) and trend form energy trajectory: trend=fade to silence sections, and fade start points marked on cards"
    "Yes, cue_out zone—any exit effect on a fading signal is inaudible, leaving only an energy vacuum."
    "Either cut before fade starts while A still has energy (something to hand over), or switch to a different A section;"
    "The envelope can also judge whether the B entry section holds (rising = build in progress, flat high energy = can layer directly)."
    "⑮ Consensus anchor = the point where five independent events (phrase end/section boundary/harmonic events/low-frequency entry/vocal boundary) coincide at the same moment,"
    "The more layers agree, the more the point looks like one a real DJ would use: phrase end plus section boundary plus harmonic event is the golden cue_out, where the line is finished, the section is over and the progression comes back round. "
    "Section boundary plus bass entry is the golden cue_in for B: the section starts and the low end arrives together. Single-layer events stay in their own fields; a consensus point is a preferred candidate, not the only option. "
    "The final layer of the phrase is transcribed from title sound symbols; occasionally rushes the beat—compare with vocal intervals.")

RENDER_CONTRACT = (
    "16. Footprints. Every move has two physical footprints: the window it occupies on A (the material the effect processes, where a vocal means C1b) and what it needs from B (the B needs field of each move on the technique card: B must not sing during the effect tail, the parallel window or the pre-roll, or C2c fails). Check first whether the footprints fit: does A's phrase end and the rest after it leave room for the occupied window, and is the instrumental gap before B's entry long enough for what the move needs. If they do not fit, move the entry or change the move. The footprints define the feasible set; your taste chooses inside it. A DJ moves in beats, not seconds: every anchor and candidate is on the grid (entries on bar lines), so adjust in whole beats times the beat length. An offset in seconds that is not a beat (+0.3s) has no musical meaning. Always emit cue_out and cue_in as {'anchor': 'aN', 'offset_beats': integer} chosen from the anchor table on the A and B cards, with |offset| at most 32; the machine resolves them to seconds on the real grid, so you choose the event and it does the arithmetic. The applies-to column of an anchor is a hint, not a limit: it says which kind of move the anchor suits structurally (blend exit at a chorus boundary of A, blend entry at a verse head of B with 16 instrumental beats before it, cut exit at a section end, cut entry at a section head or where drums and bass come in). Reason about the pair yourself; following the hint is usually safest, and departing from it is fine if you say why. The checker catches a wrong choice. The a7 and b3 in the examples are placeholders: write an id that actually exists in the card you were given. Exit anchors (phrase ends, consensus points) are for cue_out and entry anchors (section heads, drums and bass) are for cue_in; using an entry anchor as cue_out cuts into someone singing, which is C1. An anchor marked with a star is the point a working DJ would use, so prefer a starred anchor for the blend family, and explain any other choice. "
    "17. Ride first. A ride is the body of a DJ set and the cut moves are punctuation, so ride whenever you can. Default to the blend family (blendecho, blend16, loop_in, loop). To fall back to a cut move (echo, reverb, brake, backspin, filter) you must first show in the reason that every possible cue_in on this B has vocals in the sixteen beats before it, so there is nothing to ride. If you cannot show that, pick another entry point on B, or change B with b_tid and keep the blend. A cut move is the last resort, not the safe default. blendecho keeps A present (an instrumental section or a chorus) over B's intro and throws an echo on A's phrase end as B's verse takes over. blend16 is the same without the echo, a pure equal-power cross. loop_in loops B's intro or bridge as a bed and releases into B's verse when A lets go. loop holds a section of A while B grows underneath. introstack overlaps A's tail with B's intro. For every ride, put cue_in on B's lowest-energy instrumental intro or breakdown: the lower B's entry, the less the two full-level tracks bury each other. The B card lists section energy, so take the quietest instrumental section. If B is loud all the way through, a ride still works but the engine compensates with a crossfade, which is worse than a real ride, so prefer another entry or another B. "
    "Always prioritize the vocal gate first (entry = after phrase ends + rest; exit = before phrase starts), then within the safe vocal range"
    "Align beat grid and anchors; vocals take priority in conflicts. Prefer half-beat offset over cutting mid-phrase."
    "a. Breathing measures: echo entrance, and filter sequence usage (overlap_beats=0) auto-empty 4 beats (1-2-3-4,"
    "B enters only when effect tails fade. Select cue_in at the downbeat before the phrase start (section head/drum entry anchors usually satisfy this)."
    "If downbeat conflicts with phrase start, retreat to the nearest beat before the phrase start; avoid crescendo weak starts or long silence intros, leave a bar empty then soft entry like a glitch;"
    "b.filter: exit sweep length=exit_len_beats (2-16 beats), sweep window=[cue_out-exit_len, cue_out]. A is still playing within the window—the window must fall after vocals end (instrumental tail) for a clean sweep; for titles where vocals end at the phrase tail, reduce exit_len to 2-4 beats, or move cue_out to phrase tail + exit_len beats; overlap the last 2 beats with the following section;"
    "c. brake takes at least 4 beats and backspin at most 2 (exit_len_beats outside that range is clamped). Their effect window is [cue_out minus length, cue_out] and A is slowed or reversed inside it, so cue_out is not the phrase end but the phrase end plus the effect length: let the brake or backspin land on the last word or the instrumental after it, never inside a line still being sung, or C1b fails. "
    "d.echo tail echoes for 8 beats, echo_delay_beats between 0.5–1; entry_advance_beats between 0–2;"
    "e.spinup grabs ~2 beats of material before cue_in and stretches it into a 4-beat roll-in: the sound 2 beats before cue_in will be audible"
    "(vocals there will be pulled in); do not choose cue_in too close to the song start;"
    "f.B entry speed unchanged (BPM conversion difference 0.5–3% only triggers 8-bar micro-correction to gradually return to original speed), cue seconds cut by face value, no automatic snap to grid—"
    "So cue_in should be on the downbeat; but the first rule for cue_out is still finishing the musical phrase: first find the phrase-end rest, then take the nearest downbeat after it,"
    "Do not cut into the phrase early just for aesthetically pleasing beat numbers;"
    "g.blend16/blendecho ignore overlap_beats, hard-set to 16 beats; for both methods, cue_in = handover completion point,"
    "Side B: starts sounding 16 beats before cue_in (its intro) — so the window 16 beats before cue_in must be clean.")

GUIDE = GUIDE + RENDER_CONTRACT

SKILL_PRIOR = [("blendecho", .22), ("blend16", .20), ("loop_in", .15), ("loop", .13), ("introstack", .12),
               ("filter_lpf", .05), ("echo", .05), ("brake", .03), ("filter_hpf", .02), ("reverb", .02), ("backspin", .01)]

def draw_skill_hint(r=None):
    r = r or rng
    x = r.random(); acc = 0.0
    for tool, p in SKILL_PRIOR:
        acc += p
        if x <= acc:
            return tool
    return SKILL_PRIOR[0][0]

FX_CARD = {
    "exit effect": {
        "echo":     {"material needed": "The tail window must have energy (feedback silence = nothing is being projected, C1c)", "B needs": "Vocal gap ≥4.8×delay beats (sequential breathing already covers 4 beats) — B stays silent during tail echo; instrument tail feedback unlimited", "effect": "Echo the final character out", "best used when": "End of phrase + subsequent rest, with expressive tail (shout/long note)", "parameters": {"exit_len_beats": "1-4", "echo_delay_beats": "0.5-1.5"}},
        "reverb":   {"B needs": "Vocal gap ≥6 beats (sequential breathing already covers 4 beats) — reverb covers B opening = muddy", "effect": "Reverb tail", "best used when": "Lyrical/atmospheric ending, want A to fade out elegantly.", "parameters": {"exit_len_beats": "2-8"}},
        "brake":    {"material needed": "The exit window (exit_len beats before cue_out) must have energy (>=0.35x previous reference). Dead silence is useless (C1c).", "B needs": "None — stopping to silence, B opening directly is classic", "effect": "Turntable Stop (process exit_len beats of material before cue_out, ≥4 beats)", "best used when": "Sharp drop from energy peak creates surprise, B explodes directly; brake lands on the last word or instrumental tail", "weak point": "Stopping window on a still-singing phrase = dragging the title hand slow (C1b)", "parameters": {"exit_len_beats": "4-8 (stop length; cue_out=end of phrase + this length)"}},
        "backspin": {"material needed": "Reverse window must have energy (C1c)", "B needs": "None", "effect": "Reverse tail (reverse the material for 2×exit_len beats before cue_out, exit_len ≤2)", "best used when": "Hip-hop/strong groove section, old-school feel memory point; reversed on the last word/drum hit", "weak point": "Reverse window on the sentence = reverse the title manually (C1b)", "parameters": {"exit_len_beats": "1-2"}},
        "blend16":  {"B needs": "Instrumental for 16 beats before cue_in (lead16)", "effect": "16-beat equal-power crossfade: A fades out, B fades in, two tracks run in parallel with pure volume handoff (no frequency sweep)", "best used when": "A bridge/instrumental overlaps B intro, cue_in set at B main/chorus start; title emerges exactly when intro overlap ends", "weak point": "Both sides having vocal duets will clash; the 16 beats before cue_out and the 16 beats before cue_in must both be clean", "note": "cue_in = transition completion point, B starts sounding in the 16 beats before cue_in (its intro)"},
        "blendecho": {"B needs": "cue_in 16-beat instrumental + cue_in post-vocal gap ≥4.8×delay beats (A phrase end echo still overlaps B).", "effect": "An extension of blend16: A's chorus rides B's intro for 16 beats with A's vocal kept and B opening up through a low-pass, then the end of A's chorus is thrown with an echo over B's verse", "best used when": "A's chorus end lines up with B's intro-to-verse turn: cue_out at the end of A's last chorus line, cue_in at B's verse head with 16 instrumental beats before it", "weak point": "The 16 beats before B's cue_in must be instrumental while A is still singing, and A's chorus must end on a clean phrase end", "parameters": {"exit_len_beats": "1 to 2, so the echo takes only the last syllable"}},
        "loop": {"B needs": "Side B: has verse/chorus start, with enough preceding instrumental for loop to layer in", "effect": "A ride move: two bars of A are looped to hold the floor while B fades up in parallel from an instrumental point, and B's verse takes over when the loop ends on a strong beat, so A leads and B grows underneath", "best used when": "A has two clean loopable bars, instrumental or an even sung phrase, and B has an instrumental entry; use it when the outgoing track should lead the build", "weak point": "A cannot find 2 self-similar bars, so it is infeasible (engine auto-degrades)", "parameters": {"loop_echo": "true|false — whether to drop echo on the last word of the loop to send off A (true = more memorable)"}, "note": "cue determined by engine; entry_tool set to direct"},
        "loop_in": {"B needs": "B verse/chorus start first 2 bars must be clean instrumental (loopable intro/bridge)", "effect": "A ride move: two bars of B's intro or bridge become a bed, low-passed and quiet at first and then opening up, looped for about 16 beats; when A lets go at its phrase end, B releases from its verse head, so B leads the build, the wait and the release", "best used when": "B has clean loopable intro/bridge, A has clear phrase end; want build-up/release feel, longer buildup than blendecho", "weak point": "Side B: not feasible if no clean loop section (engine auto-downgrades to blendecho)", "parameters": {"loop_echo": "true|false — whether to drop echo when A is released (true=more dramatic)"}, "note": "Cue determined by engine (A phrase end + B verse head); set entry_tool to direct"},
        "introstack": {"B needs": "Overlay disc instrument (lead16)", "effect": "A tail overlaps B intro: B fades in from LPF, A fades out, B resumes from intro end", "best used when": "Side B: intro is instrumental build (drums/riff progressive), like live DJing", "parameters": {"overlap_beats": "8-16 (crossfade width)"}},
        "filter_lpf": {"B needs": "Vocal gap ≥2 beats (frequency handshake with tail, B enters with extreme frequency then releases)", "effect": "Sweep exit_len_beats until only low frequency remains before handing over (sweep window=[cue_out−exit_len, cue_out], A is still playing within the window).", "weak point": "Safest but least memorable, like an auto crossfader; sweeping the filter over vocals = muffled vocals", "parameters": {"exit_len_beats": "2-16: Sweep length—vocals sing until the end of the phrase, then shorten to 2-4 beats (fast sweep once the phrase ends); use 8-16 beats for slow sweep only if there is an instrumental tail after the phrase end", "overlap_beats": "2-16: B enters a few beats early to run in parallel with A (ride width; 2=only overlap handshake tail)"}},
        "filter_hpf": {"B needs": "Vocal gap ≥2 beats (same as lpf)", "effect": "Sweep exit_len_beats until only high frequency remains before handing over (sweep window=[cue_out−exit_len, cue_out]).", "weak point": "Same lpf, but high-frequency residue becomes harsh over time", "parameters": {"exit_len_beats": "2-16: Same lpf principle", "overlap_beats": "2-16: B enters a few beats early to run in parallel with A (ride width; 2=only overlap handshake tail)"}},
    },
    "entry effect": {
        "filter_in": {"effect": "Side B: release to full frequency from 8 beats of only low frequencies (emerging)", "best used when": "B intro is instrumental/groove section; can follow any entry move (after echo/brake release, or after a breath bar where B rises from underwater), no need to pair with filter entry"},
        "filter_in_hpf": {"effect": "Side B: release to full frequency from 8 beats of only high frequencies (grounding the airiness)", "best used when": "B intro has clear hi-hat/synth layer; can stand alone, avoid following lpf entry (spectral clash)"},
        "spinup":    {"B needs": "Gap before vocals ≥ 2 bars (grabbing 2 bars beforehand will change vocal speed)", "effect": "B starts with an acceleration into the entry", "best used when": "B entry with drums+bass is cool", "parameters": {"entry_advance_beats": "0-2 (transition starts beats before A full stop)"}},
        "direct":    {"effect": "Side B: enter directly with original track", "best used when": "B start strong hook; or B entry vocals start immediately (A exit uses brake/backspin declarative FX—echo/reverb tail suppresses B opening=C2c)", "Energy threshold": "B intro 4 beats RMS ≥ 0.25× A entry ref (C6b) — B intro too weak = seam energy cliff, change filter_in/spinup", "weak point": "zero gesture no transition smooth exit filter blend then direct vocal abrupt check B vocal range"},
    },
}
import os as _os2
try:
    BLEND_FEWSHOT = open(_os2.path.join(_os2.path.dirname(__file__), "prompts", "blend_fewshot.txt"), encoding="utf-8").read()
except Exception:
    BLEND_FEWSHOT = ""

_DORMANT_EXIT = {k: FX_CARD["exit effect"].pop(k) for k in ("introstack",) if k in FX_CARD["exit effect"]}


_PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def _camelot(k):
    if not k or ":" not in k: return None
    pc, m = k.split(":")
    if pc not in _PC: return None
    n = _PC[pc]
    return ((n * 7) % 12, 0) if m == "maj" else (((n + 3) % 12 * 7) % 12, 1)
def key_dist(ka, kb):
    a, b = _camelot(ka), _camelot(kb)
    if not a or not b: return None
    d = min((a[0] - b[0]) % 12, (b[0] - a[0]) % 12)
    return d + (0.5 if a[1] != b[1] else 0)
def _shift_key(k, semis):
    pc, m = k.split(":"); return f"{_NAMES[(_PC[pc] + semis) % 12]}:{m}"
def key_advice(ka, kb, max_semis=1):
    """Returns dict: current distance, advice keysync (for B), result after shift. Principle: match keys if possible, otherwise use adjacent keys on the circle of fifths (distance ≤1), otherwise do not change."""
    d0 = key_dist(ka, kb)
    if d0 is None:
        return {"current": "Missing key data", "suggested keysync": 0}
    if d0 <= 1.0:
        return {"current": f"A {ka} / B {kb}, Circle of Fifths distance {d0} (consonant)", "suggested keysync": 0}
    best = None
    for sgn in (1, -1):
        for s_ in range(1, max_semis + 1):
            kb2 = _shift_key(kb, sgn * s_); d = key_dist(ka, kb2)
            if d is not None and (best is None or d < best[0] or (d == best[0] and s_ < abs(best[1]))):
                best = (d, sgn * s_, kb2)
    if best and best[0] <= 1.0:
        tag = "Same key" if best[0] == 0 else ("Relative major/minor key" if best[0] == 0.5 else "Adjacent in the circle of fifths")
        return {"current": f"A {ka} / B {kb}, Circle of Fifths distance {d0} (dissonant)", "suggested keysync": best[1],
                "Move back": f"B→{best[2]}, distance to {best[0]}({tag})", "note": "Vocals are not fake within ±1 semitone; no advice beyond that"}
    return {"current": f"A {ka} / B {kb}, Circle of Fifths distance {d0}", "suggested keysync": 0,
            "note": "±1 semitone cannot save adjacent tracks; avoid harmonic overlap between two tracks by selecting points (sequence moves/rests at phrase ends)"}


def song_card(tid):
    r = A.REG[tid]
    beat = A.beat_of(tid)
    v = VAM.get(tid, {})
    se = SEGENV.get(str(tid), {})
    env_by_i = {x["i"]: x for x in se.get("segs", [])}
    segs = []
    for si, s in enumerate(r.get("segments", [])):
        row = {"label": s["label"], "start": round(s["start"], 1), "end": round(s["end"], 1)}
        ev = env_by_i.get(si)
        if ev:
            row["envelope"] = "→".join(f"{x:.2f}" for x in ev["env"])
            row["trend"] = ev["trend"]
        segs.append(row)
    vsp = [[round(s, 1), round(e, 1)] for s, e in A.VOICED.get(tid, [])][:60]
    ce = CUEEV.get(str(tid), {})
    events = {"drums enter": ce.get("kick", [])[:8], "bass enters": ce.get("bass", [])[:8],
              "harmonic loop restart": ce.get("chord_reset", [])[:10], "harmonic change": ce.get("chord_change", [])[:8]}
    events = {k: v for k, v in events.items() if v}
    card = {"tid": tid, "title": r["title"], "bpm": r["bpm"], "Seconds per beat": round(beat, 3),
            "V_q": round(v.get("valence_q", 0.5), 2), "A_q": round(v.get("arousal_q", 0.5), 2),
            "Seconds per bar": round(beat * 4, 2), "key": A.KEYS.get(tid, {}).get("key"),
            "length": int(r["duration_sec"]), "sections": segs,
            "Vocal interval (singing title period, seconds)": vsp,
            "Sound events (transition anchors, seconds)": events}
    if se.get("fade_start"):
        card["fade start (after this point energy continuously slides to silence, cue_out forbidden zone)"] = se["fade_start"]
    from plan_sim import anchor_table
    _tab = anchor_table(tid)
    def _slim(a): return {"id": a["id"], "Seconds": a["Seconds"], "Segment": a.get("Segment"), "events": a["events"], "applies to": a.get("applies to", [])}
    card["anchor table (cite as {anchor: aN, offset_beats: integer}; the applies-to column is a structural hint, not a limit: blend exit at a chorus boundary of A, cut exit at a section end, blend entry at a verse head with 16 instrumental beats before it, cut entry at a section head or where drums and bass enter)"] = \
        [_slim(a) for a in _tab]
    return card


def _blend_hint(tid, tab):
    """Star the anchors a DJ would use: a blend exit on the last word of a later chorus of A, a blend entry at B's verse head with 16 instrumental beats before it."""
    segs = A.REG.get(tid, {}).get("segments") or []
    dur = A.REG.get(tid, {}).get("duration_sec") or (segs[-1]["end"] if segs else 0)
    voiced = A.VOICED.get(tid, [])
    beat = A.beat_of(tid); win = 16 * beat
    ch = next((s for s in segs if s["label"] == "chorus"), None)
    min_t = ch["end"] if ch else dur * 0.45
    def vcov(t0, t1):
        ov = sum(max(0, min(t1, e) - max(t0, s)) for s, e in voiced if e > t0 and s < t1)
        return ov / max(1e-6, t1 - t0)
    exit_ts = []
    for s in segs:
        if s["label"] == "chorus" and s["start"] >= min_t - 1 and s["end"] < dur - 1:
            ends = [min(e, s["end"]) for st, e in voiced if e > s["start"] and st < s["end"]]
            exit_ts.append(max(ends) if ends else s["end"])
    entry_ts = [s["start"] for s in segs if s["label"] in ("verse", "chorus", "hook", "drop")
                and s["start"] - win >= 0 and vcov(s["start"] - win, s["start"]) < 0.15]
    for r in tab:
        t = r.get("Seconds"); tags = []
        if t is not None and any(abs(t - x) <= 2 * beat for x in exit_ts): tags.append("★ blend exit (A chorus last character)")
        if t is not None and any(abs(t - x) <= 1 * beat for x in entry_ts): tags.append("★ blend entry (verse head · 16 beats instrumental prior)")
        if tags: r["DJadvice"] = " / ".join(tags)
    return tab


def _first_chorus_end(tid):
    """A: end time of the first chorus title (lower bound for exit point); if no chorus, fall back to the end of the second verse or 45% duration."""
    segs = A.REG[tid].get("segments") or []
    dur = A.REG[tid].get("duration_sec") or (segs[-1]["end"] if segs else 0)
    ch = next((s for s in segs if s["label"] == "chorus"), None)
    if ch:
        return ch["end"]
    vz = [s for s in segs if s["label"] == "verse"]
    if len(vz) >= 2:
        return vz[1]["end"]
    return dur * 0.45


def _snap_exit_latter(tid, cut, min_t):
    """Exit point hard constraint: cue_out must be >= min_t (after first chorus) and fall in latter phrase (verse/chorus/bridge/inst/solo),
    not in intro/start/outro. Violation snaps to nearest latter phrase end >= min_t and aligns to A downbeat. Return (cut, snapped?)."""
    segs = A.REG[tid].get("segments") or []
    dur = A.REG[tid].get("duration_sec") or (segs[-1]["end"] if segs else 0)
    downs = A.REG[tid].get("downbeat_times") or []
    ok = ("verse", "chorus", "bridge", "inst", "solo")
    seg = next((s for s in segs if s["start"] <= cut < s["end"]), None)
    if cut >= min_t and seg and seg["label"] in ok:
        return round(cut, 2), False
    cands = [s["end"] for s in segs if s["end"] >= min_t and s["label"] in ok and s["end"] < dur - 1]
    if not cands:
        cands = [s["end"] for s in segs if s["end"] >= min_t and s["end"] < dur - 1]
    if not cands:
        cands = [min(min_t, dur - 1)]
    ne = min(cands, key=lambda t: abs(t - cut))
    if downs:
        ne = min(downs, key=lambda d: abs(d - ne))
    return round(ne, 2), True


def _snap_entry_section_head(tb, entry, beat):
    """cue_out (handoff = B verse/chorus start) must land on the first beat of the section."""
    import song_library as _A
    segs = _A.REG.get(tb, {}).get("segments") or []
    voiced = _A.VOICED.get(tb, [])
    def vcov(t0, t1):
        ov = sum(max(0, min(t1, e) - max(t0, s)) for s, e in voiced if e > t0 and s < t1)
        return ov / max(1e-6, t1 - t0)
    win = 16 * beat
    heads = [s["start"] for s in segs if s["label"] in ("verse", "chorus", "inst", "bridge", "drop", "hook")]
    valid = [h for h in heads if h - win >= 0 and vcov(h - win, h) < 0.35]
    if not valid:
        return entry, False
    near = min(valid, key=lambda h: abs(h - entry))
    if abs(near - entry) <= 24 * beat:
        return round(near, 2), abs(near - entry) > 0.15
    return entry, False

def render_free(d, spec, tag):
    out = f"{OUT}/s/{tag}"
    os.makedirs(out, exist_ok=True)
    plan = spec["plan"]
    _et = str(plan.get("exit_tool"))
    if _et in ("blendecho", "blend16", "loop_in", "loop", "introstack"):
        import song_library as _A
        _bt = 60.0 / (_A.REG.get(spec["tb"], {}).get("bpm") or 120)
        _ne, _moved = _snap_entry_section_head(spec["tb"], float(plan.get("entry", 0) or 0), _bt)
        plan = dict(plan)
        if _moved: plan["entry"] = _ne; plan["_entry_snapped"] = True
        if str(plan.get("entry_tool") or "").strip("_") not in ("", "direct"):
            plan["entry_tool"] = "__direct__"; plan["_entry_forced_direct"] = True
        spec = dict(spec, plan=plan)
    if _et in ("loop_in", "loop"):
        try:
            import render_loop as _LI
            _fn = _LI.render_loopin if _et == "loop_in" else _LI.render_loop_a
            _r = _fn(spec["ta"], spec["tb"], echo=bool(plan.get("loop_echo")))
        except Exception:
            _r = None
        if _r is not None:
            import soundfile as _sf
            _SR = 44100; _clip = _r["clip"]; _js = _r["junction"]
            _need = int(60.0 * _SR)
            _s0 = int((_js - 30.0) * _SR); _s1 = int((_js + 30.0) * _SR)
            if _s0 < 0: _s1 += -_s0; _s0 = 0
            if _s1 > len(_clip): _s0 = max(0, _s0 - (_s1 - len(_clip))); _s1 = len(_clip)
            _c = _clip[_s0:_s0 + _need]
            if len(_c) < _need: _c = np.concatenate([_c, np.zeros(_need - len(_c), dtype=_c.dtype)])
            _sf.write(f"{out}/trans1.wav", _c, _SR)
            _sf.write(f"{out}/set.wav", _c, _SR)
            plan = dict(plan); plan["cut"] = round(_r["cutA"], 2); plan["entry"] = round(_r["entryH"], 2)
            json.dump({"order_tids": [spec["ta"], spec["tb"]], "junctions": [{"plan": plan}],
                       "junction_times": [30.0], "planner": "free", "why": [spec.get("why1"), spec.get("why2")]},
                      open(f"{out}/set_trace.json", "w"), ensure_ascii=False)
            res = RC.check_run(out)
            return [f for r in res for f in r["fails"]]
        plan["exit_tool"] = "blendecho"
        spec = dict(spec, plan=plan)
    M.FREE_MODE = True
    try:
        plan = spec["plan"]
        entry0 = max(0.0, plan["cut"] - 50.0)
        final = plan["entry"] + 30.0
        tmarks, dur = M.render_set([spec["ta"], spec["tb"]], [plan], out,
                                   entry0_override=entry0, final_cut=final)
    finally:
        M.FREE_MODE = False
    json.dump({"order_tids": [spec["ta"], spec["tb"]], "junctions": [{"plan": plan}],
               "junction_times": [round(float(x), 2) for x in tmarks],
               "planner": "free", "why": [spec.get("why1"), spec.get("why2")]},
              open(f"{out}/set_trace.json", "w"), ensure_ascii=False)
    res = RC.check_run(out)
    fails = [f for r in res for f in r["fails"]]
    return fails
