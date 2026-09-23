#!/bin/bash
# Stage 4: singing-voice intervals voiced_segments (SVD CRNN union RMS VAD, on the Demucs vocal stem)
#
# Env:  dj environment (see README.md)
# Model: ../svd/svdnet.pt (our trained CRNN singing-voice detector)
# Needs: Demucs vocal stems (see PRE below); then runs ../svd/precompute_voiced.py
# Output: svd/voiced_segments.json = {tid: [[start,end],...]} (SVD union energy VAD, merge_gap 0.25 s)
#
# This stage only calls the existing builder in the right environment.
#
# usage: bash stage4_voiced.sh
set -e
DJ=${DJ_PYTHON:-python}
ROOT=${AIDJ_RUNTIME_ROOT:-$(pwd)/runtime}
PLANNER=$ROOT/dj_transition_planner

# PRE: make sure the Demucs (htdemucs) vocal stems exist; if not, separate first:
#   $DJ -m demucs --two-stems=vocals -n htdemucs -o <stems_out> <audio...>
# default stem path: $ROOT/dataset/normalized_stems/<tid>/vocals.mp3

echo "[voiced] running ../svd/precompute_voiced.py (dj env)"
cd "$PLANNER"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $DJ svd/precompute_voiced.py
echo "=== STAGE4 VOICED DONE ==="
# output: $ROOT/dj_transition_planner/svd/voiced_segments.json
