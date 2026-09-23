#!/bin/bash
# Stage 5 — 和弦 chords(CREMA,吃 full mix + downbeat 格)
#
# Env:  crema environment (see README.md)
# 模型: crema.models.chord.ChordModel(套件內建預訓練,602 類)
# 依賴: eval/variants/chords_manifest.json = {tid: {audio_path:<full mix mp3>, downbeats:[...]}}
#       downbeats 來自 stage1;audio_path 指 normalized_pool 的 full mix。
# 輸出: eval/variants/chords_crema/<tid>.json
#       = {segments:[{t0,t1,chord}], grid_chords, main_loop, n, occurrences, coverage}
#       (注:*_triad / *_simple 附加欄位的 builder 已佚失,base 結構可復現)
#
# 這一階段沿用 repo 既有 builder eval/chords_crema.py。CPU-only。
#
# 用法: bash stage5_chords.sh
set -e
CREMA=${CREMA_PYTHON:-python}
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES= $CREMA stage5_chords.py
echo "=== STAGE5 CHORDS DONE ==="
# 產物: $PLANNER/eval/variants/chords_crema/<tid>.json
