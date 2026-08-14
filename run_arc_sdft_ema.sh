#!/usr/bin/env bash
# ARC-AGI-1 Arm A, SDFT with an EMA-LoRA demonstration-conditioned teacher (--ema_teacher).
# Same config as the static-teacher SDFT arm; the only difference is the teacher is an EMA of the
# trainable adapter instead of the plain base (adapter disabled). Trains + merges only; the
# held-out (val) and train-split evals are run afterwards by the comprehensive eval pass.
set -euo pipefail
cd "$(dirname "$0")"

LR=${LR:-1e-4}
BASE=${BASE:-Qwen/Qwen3-8B}
EPOCHS=${EPOCHS:-2}
GPU=${GPU:-7}
MAXPROMPT=${MAXPROMPT:-10240}
MAXCOMP=${MAXCOMP:-4096}
ALPHA=${ALPHA:-0.01}         # EMA rate: ema = (1-alpha)*ema + alpha*student, synced every step
export WANDB_MODE=offline
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
if [ "${used:-99999}" -gt 2000 ]; then
  echo "ABORT: GPU $GPU has ${used}MiB in use by another job; pick a free GPU via GPU=." >&2
  exit 1
fi

echo "== training SDFT-EMA (arm A) on GPU $GPU, alpha=$ALPHA =="
CUDA_VISIBLE_DEVICES=$GPU uv run python main.py --dataset_name arc --model_name "$BASE" \
  --peft --lora_r 16 --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 1 --num_prompts_per_batch 32 \
  --max_prompt_length "$MAXPROMPT" --max_completion_length "$MAXCOMP" \
  --ema_teacher --ref_model_mixup_alpha "$ALPHA" \
  --output_dir ckpt/arc_sdft_ema_armA_adapter

CUDA_VISIBLE_DEVICES=$GPU uv run python merge_lora.py --base "$BASE" \
  --adapter ckpt/arc_sdft_ema_armA_adapter --out ckpt/arc_sdft_ema_armA
echo "== SDFT-EMA training+merge done -> ckpt/arc_sdft_ema_armA =="
