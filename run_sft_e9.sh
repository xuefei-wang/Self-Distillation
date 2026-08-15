#!/usr/bin/env bash
# Continue SFT-gold and SFT-insight TWO more epochs from the epoch-7 checkpoint (step 21 -> 27,
# i.e. num_train_epochs=9 resumed from the *_e7_adapter run). Fresh *_e9_adapter dir; eval scores
# the copied epoch-7 (sanity) + the new epoch-8 and epoch-9, with band breakdown. Same WITH-system
# data + eval sets as the original arms.
set -uo pipefail
cd "$(dirname "$0")"
CU=$(cd .venv-qwen35-train/lib/python3.12/site-packages/nvidia/cu13 && pwd)
LORA_TM=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
export CUDA_HOME=$CU LD_LIBRARY_PATH=$CU/lib
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
V=.venv-qwen35-train/bin/python
MODEL=ckpt/qwen35_9b_text_v2

run () {  # $1=arm $2=gpu $3=train_data $4=port
  local arm=$1 gpu=$2 data=$3 port=$4
  local src=ckpt/arc_q35_sft_${arm}_e7_adapter dst=ckpt/arc_q35_sft_${arm}_e9_adapter
  rm -rf "$dst"; mkdir -p "$dst"
  cp -r "$src/checkpoint-21" "$dst/checkpoint-21"   # epoch-7 checkpoint (optimizer/scheduler/rng)
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_PORT=$port \
    $V sft_main.py --dataset_name arc --model_name "$MODEL" --peft --lora_r 16 \
    --lora_target_modules "$LORA_TM" --learning_rate 1e-4 --num_train_epochs 9 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --max_length 8192 \
    --arc_data_dir "$data" --resume_from_checkpoint "$dst/checkpoint-21" --output_dir "$dst" \
    && CUDA_VISIBLE_DEVICES=$gpu $V eval_epochs.py --base "$MODEL" --adapter_dir "$dst" --gpu "$gpu"
  echo "sft_${arm}_e9 done"
}

run gold    6 data/arc_data/train_data        29580 > logs/sft_gold_e9.log 2>&1 &
run insight 7 data/arc_data/sft_insight_data  29581 > logs/sft_insight_e9.log 2>&1 &
wait
echo "== e9 continuation done =="
