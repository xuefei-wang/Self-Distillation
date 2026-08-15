#!/usr/bin/env bash
# The AS-RUN 6-epoch ARC-AGI-1 grid that produced the reported results (5 cells). This is the
# post-fix configuration and differs from run_arc_qwen35.sh (that one is the 2-epoch, pre-fix grid:
# EMA teacher, no sample_from_teacher_prompt, num_prompts_per_batch=31).
#
# Differences that matter here:
#   * SFT: prompt carries [system,user] (sft_main.py), 6 epochs, per-epoch eval on both splits.
#   * SDFT: --sample_from_teacher_prompt (roll the completion out from question+knowledge with the
#     student's own vLLM weights so the distillation target is CORRECT; importance sampling skipped),
#     FIXED teacher (no --ema_teacher), num_prompts_per_batch=3 (more optimizer steps; divides 93),
#     max_completion_length 4096.
#   * SDFT insight-rule (teacher_knowledge=insight) OOM'd at the 4096 cap on a wide-vocab kl_div
#     length spike; it was relaunched at 2048 + expandable_segments -> see relaunch_insight_rule.sh.
#     That 2048 value is used here so the grid reproduces cleanly.
set -uo pipefail
cd "$(dirname "$0")"
CU=$(cd .venv-qwen35-train/lib/python3.12/site-packages/nvidia/cu13 && pwd)
LORA_TM=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
export CUDA_HOME=$CU LD_LIBRARY_PATH=$CU/lib
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
V=.venv-qwen35-train/bin/python
MODEL=ckpt/qwen35_9b_text_v2
INSIGHT_PATH=$HOME/arc-train93-split/insight_knowledge.jsonl

sft_cell () {  # $1=name $2=gpu $3=train_data $4=port  (no vLLM; wide-vocab logits -> expandable_segments)
  local name=$1 gpu=$2 data=$3 port=$4
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_PORT=$port \
    $V sft_main.py --dataset_name arc --model_name "$MODEL" --peft --lora_r 16 \
    --lora_target_modules "$LORA_TM" --learning_rate 1e-4 --num_train_epochs 6 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --max_length 8192 \
    --arc_data_dir "$data" --output_dir "ckpt/arc_q35_${name}_adapter" \
    && CUDA_VISIBLE_DEVICES=$gpu $V merge_lora.py --base "$MODEL" \
       --adapter "ckpt/arc_q35_${name}_adapter" --out "ckpt/arc_q35_${name}" \
    && CUDA_VISIBLE_DEVICES=$gpu $V eval_epochs.py --base "$MODEL" \
       --adapter_dir "ckpt/arc_q35_${name}_adapter" --gpu "$gpu"
  echo "${name} done"
}

sdft_cell () {  # $1=name $2=gpu $3=port $4=maxcomp $5..=teacher flags
  local name=$1 gpu=$2 port=$3 maxcomp=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_PORT=$port \
    $V main.py --dataset_name arc --model_name "$MODEL" --peft --lora_r 16 \
    --lora_target_modules "$LORA_TM" --learning_rate 1e-4 --num_train_epochs 6 \
    --per_device_train_batch_size 1 --num_prompts_per_batch 3 --max_prompt_length 8192 \
    --max_completion_length "$maxcomp" --vllm_gpu_memory_utilization 0.5 \
    --sample_from_teacher_prompt "$@" --output_dir "ckpt/arc_q35_${name}_adapter" \
    && CUDA_VISIBLE_DEVICES=$gpu $V merge_lora.py --base "$MODEL" \
       --adapter "ckpt/arc_q35_${name}_adapter" --out "ckpt/arc_q35_${name}" \
    && CUDA_VISIBLE_DEVICES=$gpu $V eval_epochs.py --base "$MODEL" \
       --adapter_dir "ckpt/arc_q35_${name}_adapter" --gpu "$gpu"
  echo "${name} done"
}

# SFT arms (gold GPU4, insight GPU5)
sft_cell  sft_gold_e6    4 data/arc_data/train_data        29550 > logs/arc_q35_sft_gold_e6.log 2>&1 &
sft_cell  sft_insight_e6 5 data/arc_data/sft_insight_data  29551 > logs/arc_q35_sft_insight_e6.log 2>&1 &

# SDFT arms (gold-fixed GPU6 @4096, insight_demo GPU0 @4096, insight-rule GPU7 @2048 post-OOM)
sdft_cell sdft_fixstp_gold_e6 6 29566 4096 --teacher_knowledge demonstration \
  > logs/arc_q35_sdft_fixstp_gold_e6.log 2>&1 &
sdft_cell sdft_insightdemo_e6 0 29568 4096 --teacher_knowledge insight_demo \
  --insight_demo_dir data/arc_data/sft_insight_data > logs/arc_q35_sdft_insightdemo_e6.log 2>&1 &
sdft_cell sdft_fixstp_insight_e6 7 29567 2048 --teacher_knowledge insight \
  --insight_path "$INSIGHT_PATH" > logs/arc_q35_sdft_fixstp_insight_e6.log 2>&1 &

wait
echo "== 6-epoch grid done =="
