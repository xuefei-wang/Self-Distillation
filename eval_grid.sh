#!/usr/bin/env bash
# Evaluate the 5 Figure-3 checkpoints on both tasks (10 evals) -> results/fig3/eval/<ckpt>/<task>/eval_results.json
# One eval per GPU, launched in waves so two vLLM engines never share a GPU.
# All vLLM runs need VLLM_USE_FLASHINFER_SAMPLER=0 on this venv-only-CUDA box.
set -uo pipefail
cd "$(dirname "$0")"

NGPU=${NGPU:-8}
declare -A CKPT=(
  [base]=ckpt/base
  [sdft_science]=ckpt/sdft_science
  [sdft_science_tooluse]=ckpt/sdft_science_tooluse
  [sft_science]=ckpt/sft_science
  [sft_science_tooluse]=ckpt/sft_science_tooluse
)
LOGDIR=results/fig3/eval_logs
mkdir -p "$LOGDIR"

# build the job list (ckpt task script)
JOBS=()
for name in base sdft_science sdft_science_tooluse sft_science sft_science_tooluse; do
  JOBS+=("$name science eval_science.py")
  JOBS+=("$name tooluse eval_tooluse.py")
done

run_one () {  # name task script gpu
  local name=$1 task=$2 script=$3 gpu=$4
  local out="results/fig3/eval/$name/$task"
  mkdir -p "$out"
  echo "[gpu $gpu] eval $name / $task -> $out"
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1 \
    uv run python "$script" --model_path "${CKPT[$name]}" --output_dir "$out" \
    > "$LOGDIR/${name}_${task}.log" 2>&1
}

i=0
n=${#JOBS[@]}
while [ $i -lt $n ]; do
  pids=()
  for ((g=0; g<NGPU && i<n; g++, i++)); do
    read -r name task script <<< "${JOBS[$i]}"
    run_one "$name" "$task" "$script" "$g" &
    pids+=($!)
  done
  # wait for this wave before starting the next (keeps one job per GPU)
  for p in "${pids[@]}"; do wait "$p"; done
done
echo "eval grid done."
