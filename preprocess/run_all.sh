#!/bin/bash
# Orchestrator — 給定音檔資料夾,跑完整預處理管線,產出系統需要的產物。
# 跨 3 個 conda env,分階段。voiced/chords 需先備妥 stems/manifest,見對應 stage 腳本與 README。
#
# 用法: bash run_all.sh <audio_dir> [work_dir]
set -e
AUDIO=$1
WORK=${2:-./pp_out}
[ -z "$AUDIO" ] && { echo "用法: bash run_all.sh <audio_dir> [work_dir]"; exit 1; }
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

echo "== Stage 4 · voiced (env dj) — 需 demucs vocals stems;見 stage4_voiced.sh =="
echo "   (獨立跑: bash $HERE/stage4_voiced.sh  → 產 svd/voiced_segments.json)"

echo "== Stage 5 · chords (env venv_crema) — 需 chords_manifest.json;見 stage5_chords.sh =="
echo "   (獨立跑: bash $HERE/stage5_chords.sh  → 產 eval/variants/chords_crema/)"

echo "== Assemble → REG (env dj) =="
$DJ "$HERE/assemble_reg.py" "$AUDIO" "$WORK/struct.json" "$WORK/va.json" "$WORK/pool_analysis.json" \
    --keys "$WORK/keys.json" ${VOICED:+--voiced "$VOICED"}

echo "== DONE =="
echo "   REG:  $WORK/pool_analysis.json"
echo "   keys: $WORK/keys.json"
echo "   voiced / chords 見上,為系統另外消費的獨立產物。"
