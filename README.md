# DJustify: Reason-and-Verify DJ Planning for Pop Transitions and Story-Driven Sets

> This repository contains the code for "DJustify: Reason-and-Verify DJ Planning for Pop Transitions and Story-Driven Sets"
> *Submitted to the 2027 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2027)*
> Wei-Han Hsu\*, Li-Jie Lin\*, Po-Hsuan Lai, Po-Hsiang Huang, Jeng-Yue Liu, Li Su, Yi-Hsuan Yang (\* equal contribution)

Demo examples: https://jamie930625.github.io/djustify/

## Overview

DJustify is a pop AI-DJ. An LLM plans a transition (the next track, cue points
and mixing technique), the plan is rendered to audio, and a deterministic
acoustic checker scores the result and sends failures back for repair. The same
loop builds a story-driven set: one track per act of a three-clause story.

```
your songs ──► preprocess/ ──► per-track analysis
                                     │
      track A ──► planner (LLM) ──► render ──► checker ──► transition.wav
                       ▲                         │
                       └──── what failed, how ───┘   (up to 4 rounds)
```

Everything runs locally against an OpenAI-compatible endpoint serving an
open-weight LLM (we used Qwen3.6-35B-A3B, 4-bit, on one 48 GB GPU with
llama.cpp). No music is distributed: bring your own songs.

## Layout

| Path | What it is |
|---|---|
| `run_transition.py` | Plan, render and check one transition (fixed pair, or let the planner pick track B); `--variant` selects the paper's ablations |
| `djustify/` | The shared engine: LLM client and prompts (`transition_core.py`), song cards and technique card (`render_plan.py`), rendering (`render_dsp.py`, `render_loop.py`), pre-render simulation and the rendered-audio gates (`plan_sim.py`, `regression_check.py`), song registry (`song_library.py`), next-track retrieval (`retrieval.py`) |
| `story/` | Story-driven sets: `run_story_set.py`, lyric-based selection, the three listening-test stories, and an independent `verify_set.py` (its own README inside) |
| `preprocess/` | Per-song analysis pipeline: beats, structure, key, chords, vocals, valence/arousal, cue anchors (README inside) |
| `svd/` | The singing-voice detector: model, training code and the weights we used |
| `prompts/zh/` | The original Chinese prompts, verbatim, as used for the listening-test stimuli |

The prompts inside the code are English translations of `prompts/zh/`; the
structure and every rule are kept, but the two are not guaranteed to produce the
same plans.

## Setup

```bash
git clone https://github.com/ddman1101/DJustify.git && cd DJustify
conda create -n dj python=3.10 -y && conda activate dj
pip install -r requirements.txt
```

Serve the LLM (any OpenAI-compatible server works; this is what we ran):

```bash
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --port 8903 -ngl 99 -c 32768 --jinja
export AIDJ_LLM_ENDPOINT=http://127.0.0.1:8903/v1/chat/completions
```

Analyse your songs once (see `preprocess/README.md`; it needs a few extra
environments for the third-party analysers):

```bash
export AIDJ_AUDIO=/path/to/songs
export AIDJ_RUNTIME_ROOT=/path/to/runtime
# ... run the stages in preprocess/README.md
```

## Transitions

```bash
python run_transition.py --list                                   # track ids
python run_transition.py --a "<track A>" --b "<track B>"          # fixed pair (Test1 setting)
python run_transition.py --a "<track A>"                          # planner also chooses B
python run_transition.py --a "<track A>" --b "<track B>" --variant ours-c-r
```

`--variant ours` (default) is reasoning plus checker-driven repair, `ours-c`
reasoning only, `ours-c-r` neither. Each run writes `trans1.wav` and
`result.json` (the plan of every round, the planner's stated reason and the
full scorecard) under
`$AIDJ_RUNTIME_ROOT/dj_transition_planner/eval/variants/live_demo/<run id>/`.
A plan that still fails a gate after four rounds is returned with
`"green": false`; nothing is resampled or filtered.

## Story-driven sets

```bash
cd story
python run_story_set.py --story 1 --run-id story1        # one of the three listening-test stories
python run_story_set.py --story-text "<clause one>,<clause two>,<clause three>"
python verify_set.py <run directory>
```

Sets also need lyrics for the pool (`AIDJ_LYRICS_POOL_ROOT`, `AIDJ_CHORUS_DB`);
`story/INPUT_DATA.md` lists every input field and `story/README.md` walks
through how a set is built.

## Checker

The gates and their thresholds are in `djustify/transition_core.py`
(`scorecard`) and `djustify/regression_check.py`. Four families: vocal, energy,
structure, pairing. A failing gate returns the corrective action the planner
acts on; the listening-test stimuli were the planner's returned output, whether
or not every gate passed.

## Credits

The transition engine grew out of Po-Hsuan Lai's earlier transition-planner
code (retrieval and checker scaffolding); Li-Jie Lin wrote the story-set layer
and the release refactor of the engine; Wei-Han Hsu wrote the planner prompts,
renderer, checker gates and preprocessing.

## License

Code: MIT (`LICENSE`). Singing-voice detector weights: CC BY-NC 4.0 (`svd/LICENSE`).
Third-party models and packages keep their own licenses; see `THIRD_PARTY.md`.
