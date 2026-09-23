# Preprocessing

Turns a folder of songs into the per-track analysis the planner and the checker
read. Nothing here is learned end to end; each stage is an off-the-shelf
estimator or a small script, and the outputs are plain JSON keyed by track id
(the audio file name without its extension).

Set two variables first:

```bash
export AIDJ_AUDIO=/path/to/your/songs          # input audio (mp3/wav/flac)
export AIDJ_RUNTIME_ROOT=/path/to/runtime      # everything below is written here
```

The runtime root uses the layout the planner expects (see
`story/INPUT_DATA.md` for every field):

```
$AIDJ_RUNTIME_ROOT/
  dataset/normalized_pool/<track>.mp3          loudness-normalised copies (stage 0)
  dataset/normalized_stems/<track>/vocals.mp3  Demucs vocal stem (stage 0)
  dj_transition_planner/
    outputs/pool_analysis.json                 stage 1-2 + assemble
    outputs/va_micl.json                       stage 2
    outputs/cuedetr.json                       stage 6
    outputs/cue_events.json                    stage 7
    outputs/seg_env.json                       stage 8
    outputs/cue_consensus.json                 stage 9
    svd/voiced_segments.json                   stage 4
    eval/variants/keys.json                    stage 3
    eval/variants/chords_manifest.json         assemble
    eval/variants/chords_crema/<track>.json    stage 5
    eval/variants/quarantine.json              you write it ({"quarantined": []} is fine)
```

## Stages

| # | Script | Tool | Environment | Output |
|---|---|---|---|---|
| 0 | (shell, below) | ffmpeg loudnorm, Demucs `htdemucs` | `dj` | normalised audio, vocal stems |
| 1 | `stage1_struct.py` | all-in-one (`allin1`) | `allin1` | beats, downbeats, sections |
| 2 | `stage2_va.py` | MuQ-MuLan zero-shot | `dj` | valence / arousal |
| 3 | `stage3_key.py` | madmom CNNKeyRecognition | `allin1` | `keys.json` |
| 4 | `stage4_voiced.sh` → `../svd/precompute_voiced.py` | our CRNN (`../svd/svdnet.pt`) ∪ RMS VAD on the vocal stem | `dj` | `voiced_segments.json` |
| 5 | `stage5_chords.sh` → `stage5_chords.py` | CREMA | `crema` (py3.9, tensorflow-cpu) | `chords_crema/` |
| – | `assemble_reg.py` | librosa | `dj` | `pool_analysis.json`, `chords_manifest.json` |
| 6 | `stage6_cuedetr.py` | Cue-DETR (ISMIR'24) | `aidj` + a clone of cue-detr in `$CUE_DETR_DIR` | `cuedetr.json` |
| 7 | `stage7_cue_events.py` | librosa onsets on stems | `dj` | `cue_events.json` (drum / bass entries, harmonic restarts and changes) |
| 8 | `stage8_seg_env.py` | librosa RMS | `dj` | `seg_env.json` (4-point energy envelope per section, fade start) |
| 9 | `stage9_consensus.py` | rule-based clustering of stages 1, 4, 5, 7 | `dj` | `cue_consensus.json` (the anchor table the planner sees) |

Four Python environments are needed because the third-party analysers pin
incompatible versions. Point the scripts at them with
`ALLIN1_PYTHON`, `DJ_PYTHON`, `CREMA_PYTHON` (defaults: `python`). The `dj`
environment is the one in `../requirements.txt`; the others are:

```bash
conda create -n allin1 python=3.10 -y && conda activate allin1 && pip install allin1 madmom
conda create -n crema  python=3.9  -y && conda activate crema  && pip install crema "tensorflow-cpu<2.16"
```

Stage 0, run once per pool:

```bash
mkdir -p $AIDJ_RUNTIME_ROOT/dataset/normalized_pool $AIDJ_RUNTIME_ROOT/dataset/normalized_stems
for f in "$AIDJ_AUDIO"/*; do
  ffmpeg -loglevel error -i "$f" -af loudnorm=I=-14:TP=-1:LRA=11 -ar 44100 \
    "$AIDJ_RUNTIME_ROOT/dataset/normalized_pool/$(basename "${f%.*}").mp3"
done
python -m demucs --two-stems=vocals -n htdemucs \
  -o $AIDJ_RUNTIME_ROOT/dataset/normalized_stems $AIDJ_RUNTIME_ROOT/dataset/normalized_pool/*.mp3
```

Then:

```bash
bash run_all.sh $AIDJ_RUNTIME_ROOT/dataset/normalized_pool   # stages 1, 2, 3, assemble
bash stage4_voiced.sh
bash stage5_chords.sh
python stage6_cuedetr.py && python stage7_cue_events.py && python stage8_seg_env.py && python stage9_consensus.py
```

## Lyrics (story sets only)

The story-set planner also needs, under `$AIDJ_LYRICS_POOL_ROOT`,
`lyrics_txt/<track>.txt` (plain lyrics) and `whisper_json/<track>.json`
(Whisper word timestamps), plus `$AIDJ_CHORUS_DB` (title and chorus lines per
track). We transcribed with `openai-whisper` (`large-v3`) on the vocal stem and
corrected the text by hand against published lyrics; the chorus database was
built from the corrected lyrics with the same LLM. These files are not
produced by the scripts here because the corrections were manual.

## Weights

`../svd/svdnet.pt` (about 1M parameters) ships with the repository and was
trained on the Jamendo singing-voice corpus with `../svd/train.py`. All other
models are downloaded by their packages on first use.

## Notes

- Section labels follow the Harmonix vocabulary: `start intro verse chorus bridge inst solo outro end`.
- `bpm = 60 / median(diff(beat_times))`.
- Valence/arousal from stage 2 are later replaced by pool quantiles (`va_micl.json`, 0 = lowest in the pool, 1 = highest); the gates compare quantiles, so a different pool shifts the scale.
- The vocal detector decides where cues may go. A pool analysed with a different detector will cue differently.
