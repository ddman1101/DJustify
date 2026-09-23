# Third-party components

DJustify's own code is MIT (see `LICENSE`); the singing-voice detector weights
are CC BY-NC 4.0 (see `svd/LICENSE`). Everything below is downloaded or
installed separately and keeps its own license. Two of them are
non-commercial, which makes the preprocessing pipeline as a whole
non-commercial: the MuQ-MuLan weights (valence/arousal) and the madmom key
models.

| Component | Used for | License |
|---|---|---|
| Qwen3.6-35B-A3B (weights) | the planner LLM | Apache 2.0 |
| llama.cpp | serving the LLM | MIT |
| all-in-one (`allin1`) | beats, downbeats, section structure | MIT |
| Demucs (`htdemucs`) | vocal stems | MIT |
| MuQ / MuQ-MuLan (weights) | valence / arousal | code: see repository; weights: CC BY-NC 4.0 |
| madmom | key recognition | code: BSD; bundled models: CC BY-NC-SA |
| CREMA | chords | BSD-2-Clause |
| Cue-DETR | cue-point proposals | MIT |
| openai-whisper | lyrics transcription (story sets) | MIT |
| librosa, soundfile, scipy, numpy, pyloudnorm | signal processing | ISC / BSD / MIT |
| Jamendo singing-voice corpus (Ramona et al., 2008) | training data of `svd/svdnet.pt` | per-track Creative Commons |

No music is distributed with this repository. The listening-test pool was
collected by the authors and is not part of the release.
