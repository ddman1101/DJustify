#!/usr/bin/env python3
"""Plan and render one story-driven set.

usage:
  run_story_set.py --story 1
  run_story_set.py --story-text "<three clauses separated by commas>"
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "../djustify")); sys.path.insert(0, str(HERE))

import transition_core as LD          # noqa: E402
import story_set as S              # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--story", help="key of a story in the story file")
    group.add_argument("--story-text", help="the story itself, three clauses")
    ap.add_argument("--story-file", default=str(HERE / "stories.json"),
                    help="where --story looks up its text")
    ap.add_argument("--opening-track", default=None,
                    help="track id of a prescribed opening track; "
                         "without it the planner chooses the opener from act 1")
    ap.add_argument("--run-id", default=None, help="name for this run")
    return ap.parse_args()


def story_text(args) -> str:
    if args.story_text:
        return args.story_text.strip()
    stories = json.load(open(args.story_file, encoding="utf-8"))
    if args.story not in stories:
        raise SystemExit(f"{args.story} is not in {args.story_file}: "
                         f"{', '.join(sorted(stories))}")
    row = stories[args.story]
    return (row["story"] if isinstance(row, dict) else row).strip()


def follow(rid: str, log_path: Path):
    """Print the pipeline's own event stream so a headless run is followable."""
    seen = 0
    while True:
        with LD.RUNS[rid]["cond"]:
            LD.RUNS[rid]["cond"].wait(timeout=2.0)
            events = LD.RUNS[rid]["events"][seen:]
            seen += len(events)
            done = LD.RUNS[rid]["done"]
        with open(log_path, "a", encoding="utf-8") as handle:
            for event in events:
                if event.get("kind") in ("think", "out", "prompt"):
                    continue
                handle.write(json.dumps({"kind": event.get("kind"),
                                         "data": event.get("data")},
                                        ensure_ascii=False) + "\n")
                print(f"[{event.get('kind')}] "
                      f"{json.dumps(event.get('data'), ensure_ascii=False)[:600]}",
                      flush=True)
        if done and not events:
            return


def main():
    args = parse_args()
    story = story_text(args)
    rid = args.run_id or f"story_{int(time.time())}"
    if args.opening_track:
        os.environ["AIDJ_FORCE_ORDER"] = args.opening_track

    LD.RUNS.setdefault(rid, {"events": [], "cond": threading.Condition(),
                             "done": False, "t0": time.time()})
    log_path = Path(LD.OUT) / f"{rid}_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=follow, args=(rid, log_path), daemon=True).start()

    print(f"run {rid} | target {S.SET_TARGET_SEC:.0f}s | story: {story}", flush=True)
    S.pipeline_story_set(rid, story=story)
    time.sleep(3)
    out_dir = Path(LD.OUT) / f"{rid}_story_set"
    if (out_dir / "set.wav").is_file():
        print(f"set: {out_dir / 'set.wav'}", flush=True)
        print(f"trace: {out_dir / 'set_trace.json'}", flush=True)
    else:
        raise SystemExit(f"no set was produced; see {log_path}")


if __name__ == "__main__":
    main()
