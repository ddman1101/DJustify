#!/bin/bash
# Stage 4 — 人聲區間 voiced_segments(SVD CRNN ∪ RMS-VAD,吃 demucs vocals stem)
#
# Env:  dj environment (see README.md)
# 模型: svd/svdnet.pt(訓練好的 CRNN 唱聲偵測;非權重不入 git,見 README「權重」)
# 依賴: 先有 demucs vocals stem(見下 PRE),再跑既有 builder svd/precompute_voiced.py
# 輸出: svd/voiced_segments.json = {tid: [[start,end],...]}(SVD ∪ 能量 VAD,merge_gap 0.25)
#
# 這一階段直接沿用 repo 既有 builder(它包了訓練模型,不重寫)。orchestrator 只負責在對的 env 呼叫它。
#
# 用法: bash stage4_voiced.sh
set -e
DJ=${DJ_PYTHON:-python}
ROOT=${AIDJ_RUNTIME_ROOT:-$(pwd)/runtime}
PLANNER=$ROOT/dj_transition_planner

# PRE:確保 demucs vocals stem 存在(htdemucs)。若缺,先分軌:
#   $DJ -m demucs --two-stems=vocals -n htdemucs -o <stems_out> <audio...>
# 預設 stem 路徑: $ROOT/dataset/mandarin_demucs/htdemucs/<tid>/vocals.mp3

echo "[voiced] 呼叫既有 builder: svd/precompute_voiced.py (env dj)"
cd "$PLANNER"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $DJ svd/precompute_voiced.py
echo "=== STAGE4 VOICED DONE ==="
# 產物: $PLANNER/svd/voiced_segments.json
