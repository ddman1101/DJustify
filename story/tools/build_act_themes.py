#!/usr/bin/env python3
"""Ask, once per act, what a song has to be about to belong in it."""
from __future__ import annotations

import json
import os
import sys
import threading
import time

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "djustify")); sys.path.insert(0, str(ROOT / "story"))

import song_library as A
import lyric_gate as LS
import story_set as S
import transition_core as LD

POOL_FLOOR = 30
# ...and one that leaves nearly the whole library sets no condition at all.
POOL_CEILING = 260
SONGS = [t for t in A.REG if S._runtime_song_ok(t)]


def pool_size(description: str) -> int:
    return sum(1 for t in SONGS
               if not LS.theme_conflicts_with_act(description, t))

def load_stories() -> dict[str, str]:
    """Every story the act themes are built for."""
    path = os.environ.get("AIDJ_STORY_FILE", str(HERE / "stories.json"))
    rows = json.load(open(path, encoding="utf-8"))
    return {key: (row["story"] if isinstance(row, dict) else row)
            for key, row in rows.items()}


STORIES = load_stories()
refresh = "--refresh" in sys.argv
out = LS.act_theme_store_path()
store: dict[str, dict] = {}
if not refresh and os.path.exists(out):
    with open(out, encoding="utf-8") as handle:
        store = json.load(handle)

LD.RUNS.setdefault("acts", {"events": [], "cond": threading.Condition(),
                            "done": False, "t0": time.time()})
llm = S.lyric_llm_call("acts")

for number, story in sorted(STORIES.items()):
    print(f"\nstory {number}:{story}")
    for act in S.story_acts(story):
        description = str(act["description"])
        key = LS.act_theme_key(description)
        row = store.get(key)
        if row is None:
            system, user = LS.build_act_theme_prompt(description, story)
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
            NUDGES = [
                "The conditions you provided leave only {n} songs available in the library of 400 songs; this act cannot be performed."
                "Step 1: If required_relations is not explicitly stated by this act, clear it."
                "Do not modify the theme. Re-output the JSON.",
                "Still only {n} songs left. Step 2: Add a theme that is genuinely adjacent in emotion"
                "e.g. post breakup resonance acceptable breakup energy boiling acceptable party"
                "Do not add loneliness or other: those two cover most of the library, "
                "Adding this is equivalent to setting no conditions for this act. Re-output the JSON once.",
                "Still only {n} songs left. Please only add one closest theme,"
                "and again without loneliness or other. Output the JSON once more.",
            ]
            attempts: list[tuple[int, dict]] = []
            for nudge in [None] + NUDGES:
                if nudge is not None:
                    messages = messages + [
                        {"role": "assistant",
                         "content": json.dumps(attempts[-1][1], ensure_ascii=False)},
                        {"role": "user",
                         "content": nudge.format(n=attempts[-1][0])}]
                answer = llm("act theme", messages, 0.0, False)
                if not isinstance(answer, dict) or not answer.get("themes"):
                    break
                answer["description"] = description
                answer["story"] = story
                store[key] = answer
                LS._ACT_THEME_STORE = store
                size = pool_size(description)
                answer["pool"] = size
                attempts.append((size, answer))
                store.pop(key, None)
                LS._ACT_THEME_STORE = store
                if POOL_FLOOR <= size <= POOL_CEILING:
                    break
                print(f"      (remaining {size} tracks, available range {POOL_FLOOR}-{POOL_CEILING}, asking again)")
            if not attempts:
                print(f"  Act {act['act']} judgment failed, retain keyword rules")
                continue
            usable = [a for a in attempts if POOL_FLOOR <= a[0] <= POOL_CEILING]
            if usable:
                row = min(usable, key=lambda a: a[0])[1]
            else:
                over = [a for a in attempts if a[0] >= POOL_FLOOR]
                row = (min(over, key=lambda a: a[0]) if over
                       else max(attempts, key=lambda a: a[0]))[1]
            if row.get("required_relations"):
                sys_p, usr_p = LS.build_act_relation_prompt(description)
                check = llm
                if isinstance(check, dict) and not check.get("required_relations"):
                    print(f"      (The text of this act does not specify character relationships,"
                          f"Remove requirement for {'、'.join(row['required_relations'])}")
                    row["required_relations"] = []
                    row["relation_cleared_reason"] = str(check.get("reason", ""))[:200]
            store[key] = row
            LS._ACT_THEME_STORE = store
            row["pool"] = pool_size(description)
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(store, handle, ensure_ascii=False, indent=1)
        themes = "、".join(row.get("themes") or [])
        relations = "、".join(row.get("required_relations") or []) or "Unlimited"
        print(f"  act {act['act']}: {description}")
        print(f"      Needs theme {themes} | character relations {relations} | available {pool_size(description)} tracks")
        print(f"      Reason {row.get('reason','')}")

print(f"\nWrite {out}")
