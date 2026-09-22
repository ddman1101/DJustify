# DJustify

**DJustify: Reason-and-Verify DJ Planning for Pop Transitions and Story-Driven Sets**

Wei-Han Hsu\*, Li-Jie Lin\*, Po-Hsuan Lai, Po-Hsiang Huang, Jeng-Yue Liu, Li Su, Yi-Hsuan Yang
(\* equal contribution)

Submitted to ICASSP 2027.

**Audio examples:** https://jamie930625.github.io/djustify/

## What this is

Automatic DJ systems mostly target electronic dance music, whose regular structure makes rule-based mixing relatively easier. Vocal-rich popular music is harder: which track to play next, where to cue it, and which transition to use are tied together, so fixed rules handle them poorly. DJustify is a pop AI-DJ that puts an LLM in a reason–propose–verify–repair loop and lets it choose the next track, cue points, and mixing technique together. The LLM samples several transition plans; an external acoustic checker then scores the rendered audio and triggers repair when a check fails. The same planner also picks and orders a set along an emotion arc.

## Code release

The code is being cleaned up for release and will appear in this repository in October 2026:

- `djustify/` — the transition planner, renderer, and acoustic checker
- `story/` — story-driven track selection for building a set
- `preprocess/` — per-song analysis (beats, structure, key, chords, vocals, valence/arousal, cue anchors)
- `svd/` — the singing-voice detector (code and weights)

Everything runs locally with an open-weight LLM (Qwen3.6-35B-A3B); no closed-source API is used. No copyrighted music is distributed; the pipeline runs on audio you supply.

## Citation

```bibtex
@inproceedings{hsu2027djustify,
  title     = {DJustify: Reason-and-Verify DJ Planning for Pop Transitions and Story-Driven Sets},
  author    = {Hsu, Wei-Han and Lin, Li-Jie and Lai, Po-Hsuan and Huang, Po-Hsiang and Liu, Jeng-Yue and Su, Li and Yang, Yi-Hsuan},
  booktitle = {Submitted to ICASSP},
  year      = {2027}
}
```

## License

MIT (applies to the code once released).
