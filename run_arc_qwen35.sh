#!/usr/bin/env bash
# ARC-AGI-1 grid on Qwen3.5-9B (text-only, vision dropped): {SDFT-EMA, SFT} x {gold, insight}
# knowledge. Each cell trains + merges on its own GPU, then scores EVERY epoch checkpoint on both
# eval splits (seen train tasks + held-out tasks) via eval_epochs.py on that same GPU.
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
MAXPROMPT=${MAXPROMPT:-8192}
MAXCOMP=${MAXCOMP:-2048}
SFT_MAXLEN=${SFT_MAXLEN:-8192}   # SFT loads the full 9B (no vLLM offload) + expanded-LoRA optimizer
                                 # + wide-vocab logits; a smaller cap than SDFT keeps it in 48GB
VLLM_UTIL=${VLLM_UTIL:-0.5}
# GatedDeltaNet input projections are SPLIT Linears in Qwen3_5GatedDeltaNet (in_proj_qkv/_z/_b/_a,
# transformers/models/qwen3_5/modeling_qwen3_5.py:435-438), NOT the fused Qwen3-Next
# in_proj_qkvz/in_proj_ba. peft drops unmatched target names silently, so the fused names left the
# 24 linear-attention layers un-adapted (152 wrapped Linears vs 248 with these names).
LORA_TM=${LORA_TM:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}

# Eval splits scored at every epoch (built by prep_arc.py)
TRAIN_EVAL=${TRAIN_EVAL:-data/arc_data/train_eval_data}
VAL_EVAL=${VAL_EVAL:-data/arc_data/val_eval_data}
# Insight-knowledge inputs (TBD — the two insight cells are skipped until these exist)
SFT_INSIGHT_DATA=${SFT_INSIGHT_DATA:-data/arc_data/sft_insight_data}
INSIGHT_PATH=${INSIGHT_PATH:-data/arc_data/insights.jsonl}

SFT_GOLD_GPU=${SFT_GOLD_GPU:-4}
SFT_INSIGHT_GPU=${SFT_INSIGHT_GPU:-5}
SDFT_GOLD_GPU=${SDFT_GOLD_GPU:-6}
SDFT_INSIGHT_GPU=${SDFT_INSIGHT_GPU:-7}

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
echo "== preflight =="
check_free "$SFT_GOLD_GPU"     sft_gold
check_free "$SFT_INSIGHT_GPU"  sft_insight
check_free "$SDFT_GOLD_GPU"    sdft_ema_gold
check_free "$SDFT_INSIGHT_GPU" sdft_ema_insight

eval_cell () {  # $1=cell name, $2=gpu — score every epoch checkpoint on both splits
  local name=$1 gpu=$2
  $VENV eval_epochs.py --base "$MODEL" --adapter_dir "ckpt/arc_q35_${name}_adapter" --gpu "$gpu" \
    --train_eval_data "$TRAIN_EVAL" --val_eval_data "$VAL_EVAL"
}

sft_cell () {  # $1=name $2=gpu $3=completions dir $4=port (no vLLM; expandable_segments for the wide-vocab logits)
  local name=$1 gpu=$2 data=$3 port=$4
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_PORT=$port \
    $VENV sft_main.py --dataset_name arc --model_name "$MODEL" --peft --lora_r 16 \
    --lora_target_modules "$LORA_TM" --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --max_length "$SFT_MAXLEN" \
    --arc_data_dir "$data" \
    --output_dir "ckpt/arc_q35_${name}_adapter" || { echo "FAIL $name: training"; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $VENV merge_lora.py --base "$MODEL" \
    --adapter "ckpt/arc_q35_${name}_adapter" --out "ckpt/arc_q35_${name}" || { echo "FAIL $name: merge"; return 1; }
  eval_cell "$name" "$gpu"
}

sdft_cell () {  # $1=cell name, $2=gpu, $3=master_port, $4..=extra flags (teacher knowledge)
  local name=$1 gpu=$2 port=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$port \
    $VENV main.py --dataset_name arc --model_name "$MODEL" --peft --lora_r 16 \
    --lora_target_modules "$LORA_TM" --learning_rate "$LR" --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 1 --num_prompts_per_batch 32 \
    --max_prompt_length "$MAXPROMPT" --max_completion_length "$MAXCOMP" \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" --ema_teacher "$@" \
    --output_dir "ckpt/arc_q35_${name}_adapter" || { echo "FAIL $name: training"; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $VENV merge_lora.py --base "$MODEL" \
    --adapter "ckpt/arc_q35_${name}_adapter" --out "ckpt/arc_q35_${name}" || { echo "FAIL $name: merge"; return 1; }
  eval_cell "$name" "$gpu"
}

echo "== training 4 cells (sft_gold gpu$SFT_GOLD_GPU, sft_insight gpu$SFT_INSIGHT_GPU,"
echo "   sdft_ema_gold gpu$SDFT_GOLD_GPU, sdft_ema_insight gpu$SDFT_INSIGHT_GPU) =="

sft_cell  sft_gold "$SFT_GOLD_GPU" data/arc_data/train_data 29500 \
  > logs/arc_q35_sft_gold.log 2>&1 &

if [ -d "$SFT_INSIGHT_DATA" ]; then
  sft_cell sft_insight "$SFT_INSIGHT_GPU" "$SFT_INSIGHT_DATA" 29501 \
    > logs/arc_q35_sft_insight.log 2>&1 &
else
  echo "SKIP sft_insight: completions dir '$SFT_INSIGHT_DATA' does not exist (set SFT_INSIGHT_DATA)"
fi

sdft_cell sdft_ema_gold "$SDFT_GOLD_GPU" 29502 --teacher_knowledge demonstration \
  > logs/arc_q35_sdft_ema_gold.log 2>&1 &

if [ -f "$INSIGHT_PATH" ]; then
  sdft_cell sdft_ema_insight "$SDFT_INSIGHT_GPU" 29503 \
    --teacher_knowledge insight --insight_path "$INSIGHT_PATH" \
    > logs/arc_q35_sdft_ema_insight.log 2>&1 &
else
  echo "SKIP sdft_ema_insight: insight jsonl '$INSIGHT_PATH' does not exist (set INSIGHT_PATH)"
fi

wait
echo "== grid done -> ckpt/arc_q35_<cell>_adapter/epoch_eval.json per cell =="
