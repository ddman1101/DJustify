# Story-driven DJ sets

This folder holds the inference code for the **Story** condition of DJustify
(Section 3.3 of the paper): given a story written as three clauses, the planner
picks one track per act from the song pool, orders them, and renders a
continuous three-minute set. Every junction is planned, rendered and repaired by
the same verify-planning loop used for single transitions, with the pairing
fixed.

The transition system it calls lives one level up in `../djustify/` and the
per-song analysis comes from `../preprocess/`; this folder only adds the layer
that turns a story into a set.

**Prompts.** The prompts in this folder and in `../djustify/` are English
translations of the Chinese prompts that produced the listening-test sets. The
originals are kept verbatim in `../prompts/zh/`. The translation preserves every
rule, but a run with the English prompts will not reproduce the exact plans the
Chinese prompts produced.

## What you need

- Python 3.10 and the packages in `requirements.txt`.
- An OpenAI-compatible chat endpoint serving Qwen3.6-35B-A3B. The paper's runs
  used llama.cpp with a 28k context window.
- Your own audio, plus the per-song analysis and lyrics described in
  [INPUT_DATA.md](INPUT_DATA.md). The pool used in the paper is 400
  pop tracks; the audio is not distributed.

## Layout

| Path | What it is |
|---|---|
| `run_story_set.py` | Entry point: one story in, one rendered set out |
| `stories.json` | The three story prompts used in the listening test |
| `verify_set.py` | Independent acceptance check on a rendered set |
| `tools/build_song_themes.py` | Reads every song's lyrics once and records what it is about |
| `tools/build_act_themes.py` | Reads each act once and records what a song must be about to belong in it |
| `story_set.py` | Story-set planner: acts, opener choice, hop search, set budget |
| `lyric_gate.py` | Per-song lyric judgement, evidence checks, act and arc constraints |
| `../djustify/retrieval.py` | Musically valid candidates for the next track |
| `../djustify/transition_core.py` | LLM client, transition prompts, rendered-audio scorecard |
| `../djustify/plan_sim.py`, `../djustify/regression_check.py` | Pre-render simulation and rendered-audio gates |
| `../djustify/render_plan.py`, `../djustify/render_dsp.py`, `../djustify/render_loop.py` | Rendering |
| `../djustify/song_library.py` | Song registry and cue anchors |

## Running a set

```bash
export AIDJ_RUNTIME_ROOT=/path/to/runtime            # analysis, keys, chords, stems
export AIDJ_LYRICS_POOL_ROOT=/path/to/lyrics         # lyrics_txt, whisper_json
export AIDJ_CHORUS_DB=/path/to/chorus_db.json        # title and chorus per track
export AIDJ_LLM_ENDPOINT=http://127.0.0.1:8903/v1/chat/completions
export AIDJ_LLM_MODEL_ID=qwen3.6-35b-a3b

python tools/build_song_themes.py        # once per pool
python tools/build_act_themes.py         # once per story
python run_story_set.py --story 1 --run-id story1
python verify_set.py "$AIDJ_RUNTIME_ROOT/dj_transition_planner/eval/variants/live_demo/story1_story_set"   # run directory printed by the previous command
```

The run writes `set.wav` and `set_trace.json` into its run directory: the three
tracks, the cue points, the technique and parameters of each junction, and every
gate result behind them. `verify_set.py` re-derives each of those claims from
the trace and the audio and exits non-zero if one of them fails.

To run a story of your own, write it as three clauses and pass it directly:

```bash
python run_story_set.py --story-text "<clause one>,<clause two>,<clause three>"
python tools/build_act_themes.py         # the new acts need their own reading
```

`--opening-track <track id>` starts the set from a prescribed track, which is
how the listening test was run; without it the planner picks the opener itself
from act 1.

## How a set is built

1. The story is split into three acts, one per track.
2. Each act is read once by the model: what a song must be about to belong in
   it, and whether its energy rises, falls or peaks. The answers are cached in
   `reports/act_themes.json`.
3. Every track with lyrics is ranked against act 1 from a compact view of its
   lyrics, and the finalists are re-read in full. Quoted evidence must appear in
   the lyrics verbatim, or the verdict is discarded.
4. Retrieval proposes the musically valid next tracks: folded tempo within 8
   percent, key distance, valence and arousal direction, and a clean entry.
   Those candidates are ranked by their fit to the next act.
5. For the best candidate the planner enumerates every legal transition, drops
   the ones a pre-render simulation rejects, and the model picks one. The result
   is rendered and scored on the audio; a failing gate sends it back for another
   option, and a candidate that cannot pass is abandoned for the next one.
6. The three tracks are rendered as one set, checked again end to end, and
   written out with the trace.

