#!/bin/bash
# Stage 5: chords (CREMA on the full mix, voted onto the downbeat grid)
#
# Env:  crema environment (see README.md)
# Model: crema.models.chord.ChordModel (pretrained, 602 classes, bundled with the package)
# Needs: eval/variants/chords_manifest.json = {tid: {audio_path: <full mix>, downbeats: [...]}}
#        downbeats come from stage 1; audio_path points at the normalised full mix.
# Output: eval/variants/chords_crema/<tid>.json
#       = {segments:[{t0,t1,chord}], grid_chords, main_loop, n, occurrences, coverage}
#        (the *_triad / *_simple extra fields of the original pool are not reproduced)
#
# CPU only.
#
# usage: bash stage5_chords.sh
set -e
CREMA=${CREMA_PYTHON:-python}
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES= $CREMA stage5_chords.py
echo "=== STAGE5 CHORDS DONE ==="
# output: $ROOT/dj_transition_planner/eval/variants/chords_crema/<tid>.json
