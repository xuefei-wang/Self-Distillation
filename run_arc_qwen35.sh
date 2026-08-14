#!/usr/bin/env bash
# ARC-AGI-1, three arms on Qwen3.5-9B (text-only, vision dropped): SFT, SDFT (static teacher),
# SDFT-EMA. Trains + merges each concurrently on its own GPU. Eval (both splits) runs afterward
# via the comprehensive eval pass.
#
# Uses the dedicated .venv-qwen35-train (torch 2.13 / vLLM 0.27 / transformers 5.14 / trl 0.24),
# the re-keyed text checkpoint, and LoRA targets that ALSO adapt the GatedDeltaNet
# (linear-attention) layers. Env: venv-only CUDA toolkit + flashinfer version-check bypass.
set -uo pipefail
cd "$(dirname "$0")"

VENV=$(pwd)/.venv-qwen35-train/bin/python
CU=$(cd .venv-qwen35-train/lib/python3.12/site-packages/nvidia/cu13 && pwd)
MODEL=${MODEL:-ckpt/qwen35_9b_text_v2}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-2}
MAXPROMPT=${MAXPROMPT:-10240}
MAXCOMP=${MAXCOMP:-4096}
SFT_MAXLEN=${SFT_MAXLEN:-8192}   # SFT loads the full 9B (no vLLM offload) + expanded-LoRA optimizer
                                 # + wide-vocab logits; a smaller cap than SDFT keeps it in 48GB
VLLM_UTIL=${VLLM_UTIL:-0.55}
LORA_TM=${LORA_TM:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkvz,in_proj_ba,out_proj}
SFT_GPU=${SFT_GPU:-4}
SDFT_GPU=${SDFT_GPU:-5}
EMA_GPU=${EMA_GPU:-6}

export CUDA_HOME=$CU
export LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

check_free () {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null)
  [ "${used:-99999}" -gt 2000 ] && { echo "ABORT: GPU $1 ($2) busy (${used}MiB)"; exit 1; }
  echo "  GPU $1 ($2) free"
}
echo "== preflight =="; check_free "$SFT_GPU" SFT; check_free "$SDFT_GPU" SDFT; check_free "$EMA_GPU" EMA

sft_arm () {  # SFT on $SFT_GPU (no vLLM; expandable_segments for the wide-vocab logits)
  CUDA_VISIBLE_DEVICES=$SFT_GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_PORT=29500 \
    $VENV sft_main.py --dataset_name arc --model_name "$MODEL" --peft --lora_r 16 \
    --lora_target_modules "$LORA_TM" --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --max_length "$SFT_MAXLEN" \
    --output_dir ckpt/arc_q35_sft_adapter
  CUDA_VISIBLE_DEVICES=$SFT_GPU $VENV merge_lora.py --base "$MODEL" \
    --adapter ckpt/arc_q35_sft_adapter --out ckpt/arc_q35_sft
}

sdft_arm () {  # $1=extra flag (--ema_teacher or ""), $2=name, $3=gpu, $4=master_port
  local extra=$1 name=$2 gpu=$3 port=$4
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$port \
    $VENV main.py --dataset_name arc --model_name "$MODEL" --peft --lora_r 16 \
    --lora_target_modules "$LORA_TM" --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 1 --num_prompts_per_batch 32 \
    --max_prompt_length "$MAXPROMPT" --max_completion_length "$MAXCOMP" \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" $extra \
    --output_dir ckpt/arc_q35_${name}_adapter
  CUDA_VISIBLE_DEVICES=$gpu $VENV merge_lora.py --base "$MODEL" \
    --adapter ckpt/arc_q35_${name}_adapter --out ckpt/arc_q35_${name}
}

echo "== training 3 arms on Qwen3.5-9B (SFT gpu$SFT_GPU, SDFT gpu$SDFT_GPU, SDFT-EMA gpu$EMA_GPU) =="
sft_arm                              > logs/arc_q35_sft.log      2>&1 &
sdft_arm ""             sdft     "$SDFT_GPU" 29501 > logs/arc_q35_sdft.log     2>&1 &
sdft_arm "--ema_teacher" sdft_ema "$EMA_GPU"  29502 > logs/arc_q35_sdft_ema.log 2>&1 &
wait
echo "== all 3 Qwen3.5 arms trained + merged -> ckpt/arc_q35_{sft,sdft,sdft_ema} =="
