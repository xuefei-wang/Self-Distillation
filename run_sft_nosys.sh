#!/usr/bin/env bash
# Ablation: the ORIGINAL SFT prompt setting = single user turn, NO system message (question
# intact). Isolates the effect of the system turn vs the corrected [system,user] arms. Trains gold
# + insight on the *_nosys datasets and scores every epoch on the *_nosys eval sets (train + eval
# prompts both system-stripped, so no train/eval mismatch). 6 epochs, else identical to the SFT arms.
set -uo pipefail
cd "$(dirname "$0")"
CU=$(cd .venv-qwen35-train/lib/python3.12/site-packages/nvidia/cu13 && pwd)
LORA_TM=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
export CUDA_HOME=$CU LD_LIBRARY_PATH=$CU/lib
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
V=.venv-qwen35-train/bin/python
MODEL=ckpt/qwen35_9b_text_v2
TE=data/arc_data/train_eval_data_nosys
VE=data/arc_data/val_eval_data_nosys

run () {  # $1=name $2=gpu $3=train_data $4=port
  local name=$1 gpu=$2 data=$3 port=$4
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_PORT=$port \
    $V sft_main.py --dataset_name arc --model_name "$MODEL" --peft --lora_r 16 \
    --lora_target_modules "$LORA_TM" --learning_rate 1e-4 --num_train_epochs 6 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --max_length 8192 \
    --arc_data_dir "$data" --output_dir "ckpt/arc_q35_${name}_adapter" \
    && CUDA_VISIBLE_DEVICES=$gpu $V merge_lora.py --base "$MODEL" \
       --adapter "ckpt/arc_q35_${name}_adapter" --out "ckpt/arc_q35_${name}" \
    && CUDA_VISIBLE_DEVICES=$gpu $V eval_epochs.py --base "$MODEL" \
       --adapter_dir "ckpt/arc_q35_${name}_adapter" --gpu "$gpu" \
       --train_eval_data "$TE" --val_eval_data "$VE"
  echo "${name} done"
}

run sft_gold_nosys_e6    4 data/arc_data/train_data_nosys        29560 > logs/sft_gold_nosys_e6.log 2>&1 &
run sft_insight_nosys_e6 5 data/arc_data/sft_insight_data_nosys  29561 > logs/sft_insight_nosys_e6.log 2>&1 &
wait
echo "== nosys ablation done =="
