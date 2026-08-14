#!/usr/bin/env bash
# ARC-AGI-1 Arm A (gold demonstration) x {SDFT, SFT}, Qwen3-8B, LoRA r=16.
#
# Two independent single-stage runs (base -> ARC), trained concurrently on separate GPUs,
# each merged to a full checkpoint, then evaluated on the held-out ARC evaluation split (400
# tasks) alongside the base model.
#
# Knowledge type #1 (gold demonstration) is the teacher-conditioning context for SDFT and the
# SFT target. ARC sequences are long, so max_prompt_length must fit question+demonstration
# (~12k tokens) or the SDFT teacher loses the demonstrated task grids (left-truncation).
#
# Env notes (venv-only-CUDA box): every vLLM run needs VLLM_USE_FLASHINFER_SAMPLER=0.
# Produces: ckpt/arc_sdft_armA, ckpt/arc_sft_armA and results/arc_armA/eval/<ckpt>/eval_results.json
set -euo pipefail
cd "$(dirname "$0")"

LR=${LR:-1e-4}
BASE=${BASE:-Qwen/Qwen3-8B}
EPOCHS=${EPOCHS:-2}
SDFT_GPU=${SDFT_GPU:-0}
SFT_GPU=${SFT_GPU:-1}
# eval runs base / sdft / sft concurrently on three GPUs (defaults reuse the two training GPUs
# plus one more). Shared box: pass these to avoid GPUs held by other sessions' jobs.
EVAL_GPU_BASE=${EVAL_GPU_BASE:-$SDFT_GPU}
EVAL_GPU_SDFT=${EVAL_GPU_SDFT:-$SFT_GPU}
EVAL_GPU_SFT=${EVAL_GPU_SFT:-2}
MAXPROMPT=${MAXPROMPT:-10240}   # fits question + gold demonstration for ~p95 of tasks;
                                # keeps SFT logits (seq x 151k vocab) under the 48GB cap
MAXCOMP=${MAXCOMP:-4096}        # student on-policy generation budget
PDBS=${PDBS:-1}                 # per-device batch (1 keeps the long teacher forward in 48GB)
NPPB=${NPPB:-32}                # effective prompts per optimizer step
export WANDB_MODE=offline
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

# base eval dir (weights symlinked + non-thinking tokenizer)
[ -d ckpt/base ] || uv run python nothinking.py --out ckpt/base

sdft_arm () {  # SDFT arm A on GPU $SDFT_GPU
  CUDA_VISIBLE_DEVICES=$SDFT_GPU uv run python main.py --dataset_name arc --model_name "$BASE" \
    --peft --lora_r 16 --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PDBS" --num_prompts_per_batch "$NPPB" \
    --max_prompt_length "$MAXPROMPT" --max_completion_length "$MAXCOMP" \
    --output_dir ckpt/arc_sdft_armA_adapter
  CUDA_VISIBLE_DEVICES=$SDFT_GPU uv run python merge_lora.py --base "$BASE" \
    --adapter ckpt/arc_sdft_armA_adapter --out ckpt/arc_sdft_armA
}

sft_arm () {   # SFT arm A on GPU $SFT_GPU
  # expandable_segments fights fragmentation for the wide-vocab logits; safe here (SFT has no vLLM,
  # unlike the SDFT arm where it is incompatible with vLLM's CuMemAllocator sleep mode).
  CUDA_VISIBLE_DEVICES=$SFT_GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python sft_main.py --dataset_name arc --model_name "$BASE" \
    --peft --lora_r 16 --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PDBS" --gradient_accumulation_steps "$NPPB" \
    --max_length "$MAXPROMPT" --output_dir ckpt/arc_sft_armA_adapter
  CUDA_VISIBLE_DEVICES=$SFT_GPU uv run python merge_lora.py --base "$BASE" \
    --adapter ckpt/arc_sft_armA_adapter --out ckpt/arc_sft_armA
}

# Preflight: shared box — abort if a chosen training GPU is already in use by another job
# (e.g. a parallel session's vLLM worker), rather than OOM-colliding mid-init.
check_gpu_free () {  # gpu label
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null)
  if [ "${used:-99999}" -gt 2000 ]; then
    echo "ABORT: GPU $1 ($2) has ${used}MiB in use by another job. Pick a free GPU via ${2}_GPU=." >&2
    exit 1
  fi
  echo "  GPU $1 ($2) free (${used}MiB)"
}
echo "== preflight GPU check =="
check_gpu_free "$SDFT_GPU" SDFT
check_gpu_free "$SFT_GPU" SFT

echo "== training SDFT + SFT (arm A) concurrently =="
sdft_arm &
sft_arm &
wait
echo "== training done; evaluating on held-out ARC eval split =="

eval_one () {  # name ckpt gpu
  local name=$1 ckpt=$2 gpu=$3
  local out="results/arc_armA/eval/$name"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1 \
    uv run python eval_arc.py --model_path "$ckpt" --output_dir "$out" \
    > "results/arc_armA/eval/${name}.log" 2>&1
}

mkdir -p results/arc_armA/eval
eval_one base           ckpt/base          "$EVAL_GPU_BASE" &
eval_one arc_sdft_armA  ckpt/arc_sdft_armA "$EVAL_GPU_SDFT" &
eval_one arc_sft_armA   ckpt/arc_sft_armA  "$EVAL_GPU_SFT" &
wait
echo "== done =="
for n in base arc_sdft_armA arc_sft_armA; do
  acc=$(python3 -c "import json;print(json.load(open('results/arc_armA/eval/$n/eval_results.json'))['accuracy'])" 2>/dev/null || echo NA)
  echo "  $n : accuracy=$acc"
done
