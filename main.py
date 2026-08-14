from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Effective prompts per optimizer step (per_device_batch * grad_accum)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Prompts generated together per micro-batch. Higher = fewer vLLM sleep/wake "
                             "cycles per step (big speedup) at the cost of more activation memory. Must divide num_prompts_per_batch.")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science", "medical", "arc"])
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--peft", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated LoRA target module suffixes. For Qwen3.5's hybrid "
                             "layers, add the GatedDeltaNet projections in_proj_qkv,in_proj_z,"
                             "in_proj_b,in_proj_a,out_proj. These are SPLIT Linears in "
                             "Qwen3_5GatedDeltaNet; the fused Qwen3-Next names (in_proj_qkvz, "
                             "in_proj_ba) match nothing and peft drops them silently.")
    parser.add_argument("--ema_teacher", action="store_true",
                        help="LoRA only: use an EMA of the trainable adapter as the demonstration-"
                             "conditioned SDFT teacher, instead of the plain base model (adapter "
                             "disabled). EMA rate/cadence reuse --ref_model_mixup_alpha and "
                             "ref_model_sync_steps.")
    parser.add_argument("--init_model_path", type=str, default=None,
                        help="Path to init weights (merged checkpoint for stage 2). Defaults to --model_name.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45,
                        help="vLLM colocate GPU memory fraction. 0.45 fits an 8B student+vLLM on a 48GB GPU "
                             "once the separate teacher copy is dropped under --peft.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Cap optimizer steps (for smoke tests); -1 = full run")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Truncation cap for BOTH the student prompt and the (much longer, left-truncated) "
                             "teacher_prompt. For ARC set this large enough to fit question+demonstration "
                             "(~12k tokens) or the teacher loses the demonstrated task grids.")
    parser.add_argument("--max_completion_length", type=int, default=1024,
                        help="Student on-policy generation budget (max_new_tokens). ARC solutions are long; "
                             "size it so completions aren't clipped mid-answer.")
    parser.add_argument("--teacher_knowledge", type=str, default="demonstration",
                        choices=["demonstration", "target", "insight"],
                        help="ARC only. Privileged text the SDFT teacher is conditioned on: the full gold "
                             "'demonstration' CoT (default, ~6.9k tok, Arm A); the bare oracle answer grid "
                             "'target' (~0.1k tok); or the concise 'insight' knowledge (target where absent). "
                             "Compact contexts remove the teacher-prompt truncation (42%% of demonstration "
                             "teacher prompts exceed a 10240 cap) and the completion-length ratchet.")
    parser.add_argument("--arc_data_root", type=str, default="data/arc_agi_1",
                        help="ARC raw-task dir; read for --teacher_knowledge target/insight to get the oracle grids.")
    parser.add_argument("--insight_path", type=str, default=None,
                        help="Insight-knowledge jsonl (task_id, knowledge_text) for --teacher_knowledge insight.")
    return parser.parse_args()

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(train_dir) 

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


def load_medical_dataset(seed=42) -> Dataset:
    """Load and prepare medical dataset (HuatuoGPT-o1) with formatted prompts.

    Same on-disk schema as science (messages + output_text), so the same formatting applies."""
    path = 'data/medical_data/train_data'
    print(f"Loading medical dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


def _arc_target_text(arc_data_root, task_id):
    """The oracle answer in the student's output schema — the compact privileged text for
    --teacher_knowledge target (~0.1k tokens vs the ~6.9k-token gold demonstration)."""
    with open(os.path.join(arc_data_root, "training", f"{task_id}.json")) as f:
        task = json.load(f)
    return json.dumps({"outputs": [t["output"] for t in task["test"]]}, separators=(",", ":"))


def load_arc_dataset(seed=42, teacher_knowledge="demonstration",
                     arc_data_root="data/arc_agi_1", insight_path=None) -> Dataset:
    """Load the ARC-AGI-1 train set (built by prep_arc.py) and build the SDFT teacher_prompt.

    On-disk schema mirrors science (messages + output_text + task_id); messages holds a single
    user turn (the rollout question), the student sees that alone. `teacher_knowledge` selects
    the privileged text the teacher is conditioned on:
      demonstration  the full gold CoT (Arm A, ~6.9k tok) — original behaviour;
      target         the bare oracle answer grid (~0.1k tok);
      insight        the concise task-level insight, falling back to target where absent.
    The compact options remove the teacher-prompt truncation (42% of demonstration teacher
    prompts exceed a 10240 cap and are silently left-truncated, dropping the question) and the
    completion-length ratchet. The teacher instruction no longer asks for a thinking process:
    render() forces thinking off, so that clause only held teacher mass off the end token."""
    path = 'data/arc_data/train_data'
    print(f"Loading ARC dataset from {path} (teacher_knowledge={teacher_knowledge})")
    dataset = load_from_disk(path)

    insights = {}
    if teacher_knowledge == "insight":
        if not insight_path:
            raise ValueError("--teacher_knowledge insight requires --insight_path")
        with open(insight_path) as f:
            for line in f:
                o = json.loads(line)
                insights[o["task_id"]] = o["knowledge_text"]
        have = sum(1 for tid in dataset["task_id"] if tid in insights)
        print(f"insight coverage: {have}/{len(dataset)} tasks (rest fall back to target)")

    def privileged(example):
        if teacher_knowledge == "demonstration":
            return example["output_text"]
        tgt = _arc_target_text(arc_data_root, example["task_id"])
        if teacher_knowledge == "target":
            return tgt
        return insights.get(example["task_id"], tgt)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own.
""")
        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][0]['content'],
                    output_text=privileged(example),
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


def _preflight_teacher_lengths(dataset, tokenizer, max_prompt_length):
    """Make teacher-prompt truncation LOUD instead of silent. The trainer left-truncates the
    teacher prompt to max_prompt_length, which drops the question for over-cap rows. Print the
    over-cap fraction; raise if it exceeds 0.5 so a mis-sized cap fails fast rather than
    corrupting the teacher signal for most of the data."""
    over = 0
    for ex in dataset:
        txt = tokenizer.apply_chat_template(ex["teacher_prompt"], tokenize=False, add_generation_prompt=True)
        if len(tokenizer(txt).input_ids) > max_prompt_length:
            over += 1
    frac = over / max(len(dataset), 1)
    print(f"[preflight] teacher prompts over max_prompt_length={max_prompt_length}: "
          f"{over}/{len(dataset)} ({frac:.1%})")
    if frac > 0.5:
        raise ValueError(
            f"{frac:.1%} of teacher prompts exceed max_prompt_length={max_prompt_length} and would be "
            "left-truncated (question dropped). Raise --max_prompt_length or use a compact "
            "--teacher_knowledge (target/insight).")


if __name__ == "__main__":
    args = parse_args()
    # Register a text-only Qwen3.5 arch with vLLM (maps the VLM-canonical `model.language_model.*`
    # checkpoint layout onto vLLM's bare text model). No-op on the older Qwen3-8B stack. See
    # vllm_qwen35_patch for the rationale (vLLM #36275 / TRL #5269).
    import vllm_qwen35_patch
    vllm_qwen35_patch.register()
    init_path = args.init_model_path or args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        init_path,
        torch_dtype=torch.bfloat16,
    )
    # Under LoRA, the teacher is the student's own base with the adapter disabled (handled in
    # DistilTrainer.compute_loss), so we do NOT load a second full model — this is what lets an
    # 8B SDFT run fit alongside the vLLM rollout engine on a single 48GB GPU. Without --peft
    # (full fine-tuning) we still load a separate frozen teacher.
    teacher_model = None
    if not args.peft:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            init_path,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    from nothinking import patch_tokenizer
    patch_tokenizer(tokenizer)
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    elif args.dataset_name == "medical":
        dataset, _ = load_medical_dataset(args.seed)
    elif args.dataset_name == "arc":
        dataset, _ = load_arc_dataset(args.seed, args.teacher_knowledge, args.arc_data_root, args.insight_path)
        _preflight_teacher_lengths(dataset, tokenizer, args.max_prompt_length)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = max(1, args.num_prompts_per_batch // args.per_device_train_batch_size),
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs,
        max_steps = args.max_steps,
        # Recompute activations in backward to cut memory — needed for wide-vocab models like
        # Qwen3.5 (248k vocab) where the distillation kl_div over completion tokens is large.
        gradient_checkpointing = True,
        # Non-reentrant gradient checkpointing is required for multi-GPU DDP: the default reentrant
        # variant recomputes segments in backward and double-fires DDP's param-ready hooks
        # ("marked as ready twice") with a LoRA student. Harmless (and slightly cheaper) single-GPU.
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        num_iterations = 1,
        num_generations = 1,
        # Per-epoch checkpoints (checkpoint-<step>/ with the adapter) so eval_epochs.py can score
        # every epoch. A step cadence is useless here: 93 ARC prompts at num_prompts_per_batch=32
        # is only a couple of optimizer steps per epoch, so the old save_steps=100 never fired and
        # nothing but the final adapter was ever written.
        save_strategy = "epoch",
        max_grad_norm = 1,
        report_to = "wandb",
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = not args.peft,   # EMA teacher sync is incompatible with a LoRA student (param zip misaligns); use the static demo-conditioned teacher under LoRA
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        # LoRA-safe EMA teacher: a frozen "ema_teacher" adapter tracks the trainable one and
        # conditions the teacher, instead of the plain adapter-disabled base. Reuses the two
        # ref_model_* knobs above; no-op unless --ema_teacher (and --peft) is set.
        ema_teacher_lora = args.ema_teacher,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = 3,
        # Exclude budget-truncated completions (no EOS/pad tail) from the loss so a clipped
        # sample can't reinforce non-termination — the output half of the truncation guard.
        mask_truncated_completions = True,
    )
    peft_config = None
    if args.peft:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
            task_type="CAUSAL_LM",
        )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    # Persist the FINAL adapter at the top level (save_strategy="epoch" writes only
    # checkpoint-<step>/ dirs; without this the merge step has no final adapter to load).
    trainer.save_model(args.output_dir)
