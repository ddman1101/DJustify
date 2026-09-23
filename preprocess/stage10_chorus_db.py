#!/usr/bin/env python3
"""Stage 10b: chorus database for the story-set planner.

For every track in the registry that has timed lyrics, take the lyric lines
that overlap its first `chorus` section (from stage 1). If the registry has no
chorus label, fall back to the highest-energy section, then to ten lines around
one third of the song. Output:

    {"<tid>": {"title": "...", "chorus_lyrics": "line\\nline\\n..."}}

Adapted from Li-Jie Lin's build_chorus_db.py, reduced to the fields the
selector reads.

usage: python stage10_chorus_db.py [--output chorus_db.json] [--max-lines 14] [--only tids.json]
"""
import argparse, json, os, re
from pathlib import Path

ROOT = Path(os.environ.get("AIDJ_RUNTIME_ROOT") or os.path.abspath("runtime"))
POOL = Path(os.environ.get("AIDJ_LYRICS_POOL_ROOT") or "lyrics")
REGISTRY = Path(os.environ.get("AIDJ_POOL_ANALYSIS") or ROOT / "dj_transition_planner" / "outputs" / "pool_analysis.json")


def timed_lines(tid):
    p = POOL / "whisper_json" / f"{tid}.json"
    if p.is_file():
        segs = json.load(open(p, encoding="utf-8")).get("segments", [])
        return [(float(s["start"]), float(s.get("end", s["start"])), s["text"].strip()) for s in segs if str(s.get("text", "")).strip()]
    p = POOL / "lyrics_lrc" / f"{tid}.txt"
    if p.is_file():                      # [mm:ss.xx]text
        out = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
            if m and m.group(3).strip():
                out.append((int(m.group(1)) * 60 + float(m.group(2)), None, m.group(3).strip()))
        return [(s, (out[i + 1][0] if i + 1 < len(out) else s + 4.0), t) for i, (s, _, t) in enumerate(out)]
    return []


def clean_title(tid, card):
    t = (card or {}).get("title")
    if t:
        return str(t).strip()
    t = re.sub(r"^\d+\s+", "", tid)
    t = re.sub(r"\s*\[[^\]]+\]\s*$", "", t)
    return t.split(" - ", 1)[-1].strip() or tid


def chorus_span(card):
    segs = [s for s in (card or {}).get("segments", []) if s.get("label") == "chorus"]
    if segs:
        return float(segs[0]["start"]), float(segs[0]["end"])
    segs = [s for s in (card or {}).get("segments", []) if s.get("energy") is not None]
    if segs:
        s = max(segs, key=lambda x: x["energy"])
        return float(s["start"]), float(s["end"])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(POOL / "chorus_db.json"))
    ap.add_argument("--max-lines", type=int, default=14)
    ap.add_argument("--fallback-lines", type=int, default=10)
    ap.add_argument("--only", help="JSON list of tids to include (default: every registry track with lyrics)")
    a = ap.parse_args()
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    tids = json.load(open(a.only, encoding="utf-8")) if a.only else list(reg)
    db = {}
    for tid in tids:
        lines = timed_lines(tid)
        if not lines:
            continue
        span = chorus_span(reg.get(tid))
        if span:
            pick = [t for s, e, t in lines if e > span[0] and s < span[1]][: a.max_lines]
        else:
            pick = []
        if not pick:
            dur = float((reg.get(tid) or {}).get("duration_sec") or lines[-1][1])
            i = min(range(len(lines)), key=lambda k: abs(lines[k][0] - dur / 3))
            pick = [t for _, _, t in lines[i: i + a.fallback_lines]]
        db[tid] = {"title": clean_title(tid, reg.get(tid)), "chorus_lyrics": "\n".join(pick)}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(db, open(a.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"=== STAGE10 CHORUS DB DONE === {len(db)} tracks -> {a.output}")


if __name__ == "__main__":
    main()
