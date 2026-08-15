#!/usr/bin/env bash
# Epoch-0 baseline: score the bare base text checkpoint (the model every student inits from)
# on BOTH ARC eval splits, using the SAME scorer/generation path (eval_arc.py, greedy,
# 12288/20480 budget) that eval_epochs.py drives for the trained checkpoints -> apples-to-apples.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs   # split stdout/stderr is redirected here; without this the redirects fail silently

VENV=$(pwd)/.venv-qwen35-train/bin/python
CU=$(cd .venv-qwen35-train/lib/python3.12/site-packages/nvidia/cu13 && pwd)
MODEL=${MODEL:-ckpt/qwen35_9b_text_v2}
TRAIN_EVAL=${TRAIN_EVAL:-data/arc_data/train_eval_data}
VAL_EVAL=${VAL_EVAL:-data/arc_data/val_eval_data}
OUT=${OUT:-ckpt/arc_q35_base_baseline}
TRAIN_GPU=${TRAIN_GPU:-1}
VAL_GPU=${VAL_GPU:-2}

export CUDA_HOME=$CU
export LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

eval_split () {  # $1=gpu $2=data_dir $3=out_subdir
  CUDA_VISIBLE_DEVICES=$1 $VENV eval_arc.py --model_path "$MODEL" \
    --data_dir "$2" --output_dir "$OUT/$3" \
    --max_new_tokens 12288 --max_model_len 20480
}

echo "== base baseline: train split (gpu $TRAIN_GPU) + val split (gpu $VAL_GPU) =="
eval_split "$TRAIN_GPU" "$TRAIN_EVAL" train > logs/base_baseline_train.log 2>&1 &
train_pid=$!
eval_split "$VAL_GPU"   "$VAL_EVAL"   val   > logs/base_baseline_val.log 2>&1 &
val_pid=$!

fail=0
wait "$train_pid" || { echo "FAIL train split (see logs/base_baseline_train.log)"; fail=1; }
wait "$val_pid"   || { echo "FAIL val split (see logs/base_baseline_val.log)"; fail=1; }
if [ "$fail" -ne 0 ]; then
  echo "== base baseline FAILED =="
  exit 1
fi
echo "== base baseline done -> $OUT/{train,val}/eval_results.json =="
