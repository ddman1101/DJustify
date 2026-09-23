#!/bin/bash
# Orchestrator: given a folder of audio, run the preprocessing stages and write what the planner reads.
# Spans three environments. Stages 4 (voiced) and 5 (chords) need the stems / manifest first; see their scripts and README.md.
#
# usage: bash run_all.sh <audio_dir> [work_dir]
set -e
AUDIO=$1
WORK=${2:-./pp_out}
[ -z "$AUDIO" ] && { echo "usage: bash run_all.sh <audio_dir> [work_dir]"; exit 1; }
mkdir -p "$WORK"
HERE=$(cd "$(dirname "$0")" && pwd)
ALLIN1=${ALLIN1_PYTHON:-python}
DJ=${DJ_PYTHON:-python}

echo "== Stage 1 · struct (env allin1) =="
$ALLIN1 "$HERE/stage1_struct.py" "$AUDIO" "$WORK/struct.json"

echo "== Stage 3 · key (env allin1) =="
$ALLIN1 "$HERE/stage3_key.py" "$AUDIO" "$WORK/keys.json"

echo "== Stage 2 · valence/arousal (env dj) =="
CUDA_VISIBLE_DEVICES="${VA_GPU:-0}" $DJ "$HERE/stage2_va.py" "$AUDIO" "$WORK/struct.json" "$WORK/va.json"

echo "== Stage 4 - voiced (dj env): needs Demucs vocal stems; see stage4_voiced.sh =="
echo "   (run separately: bash $HERE/stage4_voiced.sh  -> svd/voiced_segments.json)"

echo "== Stage 5 - chords (crema env): needs chords_manifest.json; see stage5_chords.sh =="
echo "   (run separately: bash $HERE/stage5_chords.sh  -> eval/variants/chords_crema/)"

echo "== Assemble → REG (env dj) =="
$DJ "$HERE/assemble_reg.py" "$AUDIO" "$WORK/struct.json" "$WORK/va.json" "$WORK/pool_analysis.json" \
    --keys "$WORK/keys.json" ${VOICED:+--voiced "$VOICED"}

echo "== DONE =="
echo "   REG:  $WORK/pool_analysis.json"
echo "   keys: $WORK/keys.json"
echo "   voiced / chords: see above; they are separate outputs the planner also reads."
