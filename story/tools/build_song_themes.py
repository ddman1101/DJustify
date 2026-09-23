#!/usr/bin/env python3
"""Ask the model what each song is about, once, for the whole library.

The question is deliberately blind: the model sees the lyrics and nothing else
-- no act, no story, no hint of a wanted answer -- so it has nothing to
rationalise toward.  Naming a song's subject is a task a small model does
reliably; deciding whether a song "fits act 2 of a five-act breakup narrative"
is not, which is why that call kept rubber-stamping directly compliant for songs that had
nothing to do with the act.

The verdict does not depend on the act, so 400 calls serve every story, every
act and every future run.  Results land in reports/song_themes.json.

usage: build_themes.py [--refresh]
"""
from __future__ import annotations

import json
import os
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "djustify")); sys.path.insert(0, str(ROOT / "story"))

import song_library as A
import lyric_gate as LS
import story_set as S

OUT = LS._theme_store_path()
refresh = "--refresh" in sys.argv

existing: dict[str, dict[str, str]] = {}
if not refresh and os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as handle:
        existing = json.load(handle)

catalog = LS.LyricCatalog()
# lyric_llm_call emits progress events into transition_core's RUNS table, so
# the run id has to exist there before the first call.
import transition_core as LD
import threading, time
LD.RUNS.setdefault("themes", {"events": [], "cond": threading.Condition(),
                              "done": False, "t0": time.time()})
llm = S.lyric_llm_call("themes")

todo = [t for t in A.REG if t not in existing and A.in_pool(t) and t not in A.QUAR]
print(f"Library {len(A.REG)} tracks, parsed {len(existing)}, pending {len(todo)}", flush=True)

done = 0
for tid in todo:
    try:
        lyrics = catalog.full_lyrics(tid)
    except Exception:
        continue
    if not lyrics.strip():
        continue
    system, user = LS.build_theme_prompt(lyrics)
    try:
        row = llm("song theme", [{"role": "system", "content": system},
                                 {"role": "user", "content": user}], 0.0, False)
    except Exception as exc:                       # keep going; one bad song
        print(f"  ✗ {A.REG[tid]['title'][:34]}: {exc}", flush=True)
        continue
    if not isinstance(row, dict) or "themes" not in row:
        continue
    # The evidence has to be a real line, the same rule every other stage obeys.
    evidence = str(row.get("evidence", ""))
    if evidence and LS._normalized_evidence(evidence) not in LS._normalized_evidence(lyrics):
        row["evidence_verbatim"] = False
    else:
        row["evidence_verbatim"] = True
    existing[tid] = {
        "themes": [str(t) for t in (row.get("themes") or ["Other"])],
        "summary": str(row.get("summary", "")),
        "relation": str(row.get("relation", "No explicit target")),
        "evidence": evidence,
        "evidence_verbatim": row["evidence_verbatim"],
    }
    done += 1
    if done % 20 == 0:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=1)
        print(f"  {done}/{len(todo)}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(existing, handle, ensure_ascii=False, indent=1)

counts: dict[str, int] = {}
for row in existing.values():
    for theme in row.get("themes", []):
        counts[theme] = counts.get(theme, 0) + 1
print(f"\nDone, total {len(existing)} songs. Theme distribution:")
for theme, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"   {n:4d}  {theme}")
print(f"Output {OUT}")
