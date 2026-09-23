#!/usr/bin/env python3
"""Independent acceptance check for one rendered story set."""
import json, os, sys
from pathlib import Path

# Resolve the target before anything is imported: the modules below resolve
# their own paths at import time.
_RUN_DIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else ""
sys.path.insert(0, str(Path(__file__).resolve().parent / "../djustify")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import song_library as A
import lyric_gate as LS
import story_set as S
import regression_check as RC

TARGET, SLACK = 180.0, 8.0
MIN_SHARE = 30.0
RIDE = {"blendecho", "blend16", "loop_in", "loop", "introstack"}


def check(run_dir: str) -> tuple[bool, list[str], list[str]]:
    fails: list[str] = []
    notes: list[str] = []
    trace_path = os.path.join(run_dir, "set_trace.json")
    if not os.path.isfile(trace_path):
        return False, ["No set_trace.json"], []
    t = json.load(open(trace_path, encoding="utf-8"))
    catalog = LS.LyricCatalog()

    # 1. length -------------------------------------------------------------
    duration = float(t["set_duration_sec"])
    if not (TARGET - SLACK <= duration <= TARGET + SLACK):
        fails.append(f"Total length {duration:.1f}s is not within {TARGET-SLACK:.0f}–{TARGET+SLACK:.0f}s")
    notes.append(f"Total length {duration:.1f}s")

    # 2. every song gets real airtime ---------------------------------------
    shares = [float(x) for x in t["per_song_share_sec"]]
    if min(shares) < MIN_SHARE:
        fails.append(f"Played for {min(shares):.1f}s with song (below {MIN_SHARE:.0f}s)")
    notes.append("Assign " + str([round(x, 1) for x in shares]))

    # 3. no song may appear twice, under any upload -------------------------
    titles = [s["title"] for s in t["selections"]]
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if S.same_song(titles[i], titles[j]):
                fails.append(f"Duplicate song: {titles[i]} / {titles[j]}")

    # 4/5/6. per-song story checks ------------------------------------------
    for sel in t["selections"]:
        tid, title = sel["tid"], sel["title"]
        act = sel["act"]
        description = str(act.get("description", ""))
        fit = sel.get("story_fit")
        if fit != "direct fit":
            fails.append(f"[{sel['position']}] {title} verdict={fit}")
        evidence = str(sel.get("story_evidence") or "")
        try:
            full = catalog.full_lyrics(tid)
        except Exception as exc:
            fails.append(f"[{sel['position']}] {title} Failed to get complete lyrics: {exc}")
            continue
        # the quote the judge used must really be in the lyric
        if not evidence:
            fails.append(f"[{sel['position']}] {title} No word-by-word evidence")
        elif LS._normalized_evidence(evidence) not in LS._normalized_evidence(full):
            fails.append(f"[{sel['position']}] {title} Word-by-word evidence not in lyrics")
        # What the song is about, judged by the model reading the whole lyric
        # with no act in front of it, must belong to this act.
        conflict = LS.theme_conflicts_with_act(description, tid)
        if conflict:
            fails.append(f"[{sel['position']}] {title} {conflict}")
        else:
            row = LS.song_theme(tid) or {}
            themes = "／".join(row.get("themes", [])) or "(not judged)"
            notes.append(f"[{sel['position']}] {title} Theme interpretation {themes}")
        arousal = (A.REG.get(tid) or {}).get("arousal")
        if LS.arousal_conflicts_with_act(description, arousal):
            fails.append(
                f"[{sel['position']}] {title} This act requires explosive/high energy,"
                f"arousal={float(arousal):.3f} below {LS.ENERGY_FLOOR_UP:.2f}")
        if isinstance(arousal, (int, float)):
            notes.append(
                f"[{sel['position']}] {title} arousal={float(arousal):.3f}"
                f" (act direction {LS.act_energy_direction(description) or 'neutral'})")

    arousals = [(A.REG.get(sel["tid"]) or {}).get("arousal") for sel in t["selections"]]
    acts = [str(sel["act"].get("description", "")) for sel in t["selections"]]
    if all(isinstance(value, (int, float)) for value in arousals) and len(arousals) == 3:
        values = [float(value) for value in arousals]
        tolerance = 0.12
        first_peak = LS.act_demands_peak_energy(acts[0])
        last_peak = LS.act_demands_peak_energy(acts[-1])
        # The same step the planner was told to honour is re-checked here, so a
        # set that merely stays level cannot be handed over as a build-up.
        step = float(os.environ.get("AIDJ_MIN_ARC_STEP", "0"))
        if last_peak and not first_peak and step > 0:
            if not (values[1] >= values[0] + step and values[2] >= values[1] + step):
                fails.append(
                    f"Energy not pushed up level by level: arousal {[round(v, 3) for v in values]}"
                    f"(this story requires each section to rise by at least {step:.2f})")
            else:
                notes.append(f"Energy arc {[round(v, 3) for v in values]} ↗"
                             f"(Each segment at least +{step:.2f})")
        elif last_peak and not first_peak:
            if not (values[0] <= values[1] + tolerance
                    and values[1] <= values[2] + tolerance
                    and values[2] > values[0]):
                fails.append(
                    f"Energy arc not pushing up: arousal {[round(v, 3) for v in values]}"
                    "(story: cumulative energy → build-up → climax)")
            notes.append(f"Energy arc {[round(v, 3) for v in values]} ↗")
        elif first_peak and not last_peak:
            if not (values[0] >= values[1] - tolerance
                    and values[1] >= values[2] - tolerance):
                fails.append(
                    f"Energy arc not pulling down: arousal {[round(v, 3) for v in values]}"
                    "(story: high-energy opening → transition into immersion)")
            notes.append(f"Energy arc {[round(v, 3) for v in values]} ↘")
        else:
            notes.append(f"Energy arc {[round(v, 3) for v in values]} (this story has no explicit direction requirement)")

    # 7. the rendered audio must still pass the team's own checker -----------
    plans = [j["plan"] for j in t["junctions"]]
    try:
        rows = RC.check_run(run_dir)
    except Exception as exc:
        fails.append(f"regression checker failed to run: {exc}")
        rows = []
    for index, row in enumerate(rows):
        bad = S.story_mode_regression_fails(
            row.get("fails", []), plans[index] if index < len(plans) else None)
        if bad:
            fails.append(f"Transition {index+1} checker red light: {bad}")
    notes.append("Interlude " + " / ".join(
        f"{p['exit_tool']}→{(p.get('entry_tool') or '').strip('_') or 'direct'}"
        + ("★RIDE" if p["exit_tool"] in RIDE else "")
        for p in plans))

    # 8. the audio has to exist and be the length the trace claims ----------
    wav = os.path.join(run_dir, "set.wav")
    if not os.path.isfile(wav):
        fails.append("no set.wav")
    else:
        import wave
        with wave.open(wav) as handle:
            real = handle.getnframes() / float(handle.getframerate())
        if abs(real - duration) > 1.0:
            fails.append(f"set.wav actual duration {real:.1f}s does not match trace duration {duration:.1f}s")
    return not fails, fails, notes


def rank_key(run_dir: str) -> tuple[int, int, float]:
    """How close this render is to acceptable, for picking a fallback."""
    _, fails, _ = check(run_dir)
    try:
        trace = json.load(open(os.path.join(run_dir, "set_trace.json"), encoding="utf-8"))
    except Exception:
        return (99, 0, 999.0)
    direct = sum(1 for sel in trace["selections"] if sel.get("story_fit") == "direct fit")
    return (len(fails), -direct, abs(float(trace["set_duration_sec"]) - TARGET))


if __name__ == "__main__":
    ok, fails, notes = check(_RUN_DIR)
    for line in notes:
        print("  · " + line)
    for line in fails:
        print("  ✗ " + line)
    key = rank_key(_RUN_DIR)
    print(f"SCORE {key[0]} {-key[1]} {key[2]:.1f}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
