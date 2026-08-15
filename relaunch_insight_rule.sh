#!/usr/bin/env bash
# Relaunch SDFT insight-rule-fixed (6ep) after the wide-vocab kl_div OOM at step 94 (a 4096-token
# clipped completion x 248k vocab spiked 3.79 GiB with only 2.94 free + 1.49 fragmented).
# Fix: max_completion_length 4096 -> 2048 (healthy rollouts are <600 tok; 4096-clips are degenerate
# rambles) + expandable_segments:True to reclaim the fragmentation the error named.
set -uo pipefail
cd "$(dirname "$0")"
CU=$(cd .venv-qwen35-train/lib/python3.12/site-packages/nvidia/cu13 && pwd)
LORA_TM=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
export CUDA_HOME=$CU
export LD_LIBRARY_PATH=$CU/lib
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=.venv-qwen35-train/bin/python
name=sdft_fixstp_insight_e6; gpu=7
CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=29577 $V main.py --dataset_name arc --model_name ckpt/qwen35_9b_text_v2 \
  --peft --lora_r 16 --lora_target_modules "$LORA_TM" --learning_rate 1e-4 --num_train_epochs 6 \
  --per_device_train_batch_size 1 --num_prompts_per_batch 3 --max_prompt_length 8192 \
  --max_completion_length 2048 --vllm_gpu_memory_utilization 0.5 \
  --sample_from_teacher_prompt --teacher_knowledge insight --insight_path $HOME/arc-train93-split/insight_knowledge.jsonl \
  --output_dir ckpt/arc_q35_${name}_adapter \
  && CUDA_VISIBLE_DEVICES=$gpu $V merge_lora.py --base ckpt/qwen35_9b_text_v2 --adapter ckpt/arc_q35_${name}_adapter --out ckpt/arc_q35_${name} \
  && CUDA_VISIBLE_DEVICES=$gpu $V eval_epochs.py --base ckpt/qwen35_9b_text_v2 --adapter_dir ckpt/arc_q35_${name}_adapter --gpu $gpu
echo "relaunch fixstp_insight_e6 done"
