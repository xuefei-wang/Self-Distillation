#!/usr/bin/env bash
# Figure 3 reproduction (2-task sequential continual learning, Qwen3-8B, LoRA).
# Trains both arms sequentially (Science -> Tool Use), merging the LoRA adapter into a full
# model between stages so each stage-2 starts from the merged stage-1 model and the vLLM eval
# scripts can load every checkpoint directly.
#
# Produces the 4 merged checkpoints used by eval_grid.sh:
#   ckpt/sdft_science, ckpt/sdft_science_tooluse, ckpt/sft_science, ckpt/sft_science_tooluse
# (base eval target ckpt/base is built once via: uv run python nothinking.py --out ckpt/base)
#
# Env notes (this venv-only-CUDA box): every vLLM run needs VLLM_USE_FLASHINFER_SAMPLER=0;
# deepspeed must NOT be installed (accelerate imports it and it needs a CUDA toolkit).
# The two arms are independent -> run them on separate GPUs concurrently (SDFT_GPU, SFT_GPU).
set -euo pipefail
cd "$(dirname "$0")"

LR=${LR:-1e-4}
BASE=${BASE:-Qwen/Qwen3-8B}
SDFT_GPU=${SDFT_GPU:-0}
SFT_GPU=${SFT_GPU:-1}
PDBS=${PDBS:-4}          # per-device (generation) batch for SDFT; fewer vLLM sleep/wake cycles
export WANDB_MODE=offline
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

sdft_stage () {  # dataset init_path adapter_out merged_out
  local ds=$1 init=$2 adapter=$3 merged=$4
  CUDA_VISIBLE_DEVICES=$SDFT_GPU uv run python main.py --dataset_name "$ds" --model_name "$BASE" \
    --init_model_path "$init" --peft --learning_rate "$LR" --num_train_epochs 2 \
    --per_device_train_batch_size "$PDBS" --output_dir "$adapter"
  CUDA_VISIBLE_DEVICES=$SDFT_GPU uv run python merge_lora.py --base "$init" --adapter "$adapter" --out "$merged"
}

sft_stage () {  # dataset init_path adapter_out merged_out
  local ds=$1 init=$2 adapter=$3 merged=$4
  CUDA_VISIBLE_DEVICES=$SFT_GPU uv run python sft_main.py --dataset_name "$ds" --model_name "$BASE" \
    --init_model_path "$init" --peft --learning_rate "$LR" --num_train_epochs 1 --output_dir "$adapter"
  CUDA_VISIBLE_DEVICES=$SFT_GPU uv run python merge_lora.py --base "$init" --adapter "$adapter" --out "$merged"
}

# base eval dir (weights symlinked + non-thinking tokenizer)
[ -d ckpt/base ] || uv run python nothinking.py --out ckpt/base

# SDFT arm (GPU $SDFT_GPU): base -> science -> +tooluse
sdft_arm () {
  sdft_stage science "$BASE"             ckpt/sdft_science_adapter          ckpt/sdft_science
  sdft_stage tooluse ckpt/sdft_science   ckpt/sdft_science_tooluse_adapter  ckpt/sdft_science_tooluse
}
# SFT arm (GPU $SFT_GPU): base -> science -> +tooluse
sft_arm () {
  sft_stage science "$BASE"            ckpt/sft_science_adapter          ckpt/sft_science
  sft_stage tooluse ckpt/sft_science   ckpt/sft_science_tooluse_adapter  ckpt/sft_science_tooluse
}

sdft_arm &
sft_arm &
wait
echo "All 4 merged checkpoints ready. Next: bash eval_grid.sh && uv run python collect_accuracy.py && uv run python plot_fig3.py"
