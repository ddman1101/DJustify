# DJustify: Reason-and-Verify DJ Planning for Pop Transitions and Story-Driven Sets

> This repository contains the code for "DJustify: Reason-and-Verify DJ Planning for Pop Transitions and Story-Driven Sets"
> *Submitted to the 2027 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2027)*
> Wei-Han Hsu\*, Li-Jie Lin\*, Po-Hsuan Lai, Po-Hsiang Huang, Jeng-Yue Liu, Li Su, Yi-Hsuan Yang (\* equal contribution)

Demo examples: https://jamie930625.github.io/djustify/

## Overview

DJustify is a pop AI-DJ. An LLM plans a transition (the next track, cue points, and mixing technique), the plan is rendered to audio, and a deterministic acoustic checker scores the result and sends failures back for repair. The same loop builds a story-driven set. Everything runs locally with an open-weight LLM; no music is distributed.

## Code release

The code is being cleaned up and will appear here in October 2026:

- `djustify/` — transition planner, renderer, and acoustic checker
- `story/` — story-driven track selection for building a set
- `preprocess/` — per-song analysis (beats, structure, key, chords, vocals, valence/arousal, cue anchors)
- `svd/` — singing-voice detector (code and weights)

## License

MIT
