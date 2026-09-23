# Input data

The planner never analyses audio itself. It reads one analysis record per
track, plus the lyrics, and opens the audio file only when it renders. Produce
these with `../preprocess/` (see its README), then point the environment
variables below at the results.

```
$AIDJ_RUNTIME_ROOT/
  dj_transition_planner/
    outputs/pool_analysis.json        structure, beats, valence, arousal (override with $AIDJ_POOL_ANALYSIS)
    outputs/cuedetr.json              Cue-DETR cue proposals
    outputs/cue_events.json           drum/bass entries, harmonic restarts and changes
    outputs/seg_env.json              per-section energy envelope and fade start
    outputs/cue_consensus.json        consensus anchors (the anchor table the planner sees)
    outputs/va_micl.json              valence and arousal pool quantiles
    svd/voiced_segments.json          singing-voice activity, seconds
    eval/variants/keys.json           key and confidence
    eval/variants/chords_manifest.json  audio path per track
    eval/variants/quarantine.json     tracks to exclude (may be empty)
    eval/variants/chords_crema/<track>.json  chords and main loop (CREMA)
  dataset/normalized_stems/<track>/vocals.mp3   vocal stem, used by the checker

$AIDJ_LYRICS_POOL_ROOT/
  lyrics_txt/<track>.txt              plain lyrics            (preprocess/stage10_lyrics.py)
  whisper_json/<track>.json           lyrics with timestamps  (preprocess/stage10_lyrics.py)

$AIDJ_CHORUS_DB                       one JSON file: title and chorus per track (preprocess/stage10_chorus_db.py)
```

Every file is keyed by the same track id, which is also the audio file name
without its extension, for example `3004 After Last Night [nlZDv8yuxCc]`.

## pool_analysis.json

One object per track. The planner reads these fields:

| Field | Meaning |
|---|---|
| `title`, `artist` | Shown to the model when it chooses a track |
| `audio_path` | Absolute path to the audio file |
| `duration_sec`, `bpm` | Length and tempo |
| `beat_times`, `downbeat_times` | Beat grid, seconds |
| `segments` | Structure: `index`, `start`, `end`, `label`, `energy` |
| `energy`, `va_sections`, `va_stats` | Per-section energy and emotion |
| `valence`, `arousal` | Track-level emotion, overwritten by `va_micl.json` |
| `has_lyrics` | Whether a lyric file exists |

## va_micl.json

`{"<track>": {"valence_q": 0.68, "arousal_q": 0.61, ...}}`. The quantiles are
what the gates compare: 0 is the least aroused track in the pool, 1 the most.

## voiced_segments.json

`{"<track>": [[20.38, 128.66], [129.43, 138.87], ...]}`, the intervals where a
lead vocal is present. Cue placement and the vocal gates depend on this file,
so a pool analysed with a different detector will cue differently.

## keys.json and chords_manifest.json

`keys.json` holds `{"key": "D#:maj", "conf": 0.887}` per track; the planner uses
it for key compatibility. `chords_manifest.json` maps a track to its audio path.

## Lyrics

`lyrics_txt/<track>.txt` is the plain text the lyric gate reads in full.
`whisper_json/<track>.json` carries the same lines with start times, which is
how the planner checks whether a line is inside the part of the track that is
actually played. A track without lyrics can still be rendered, but it can never
satisfy an act, so it will not appear in a story set.

## chorus_db.json

`{"<track>": {"title": "...", "chorus_lyrics": "..."}}`. The lyric gate uses the
chorus to build the compact view it ranks the pool with, so a track missing from
this file is never considered.

## Caches the tools write

`reports/song_themes.json` holds one blind reading per track: what the song is
about, who it is about, and a quoted line. `reports/act_themes.json` holds one
reading per act. Both are keyed so that a run reuses them; delete a file, or
pass `--refresh`, to rebuild it.
