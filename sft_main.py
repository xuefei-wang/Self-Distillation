import argparse
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def parse_args():
    parser = argparse.ArgumentParser(description="SFT baseline")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["science", "tooluse"], help="Dataset name")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--init_model_path", type=str, default=None,
                         help="Path to init weights (merged checkpoint for stage 2). Defaults to --model_name.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--peft", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_steps", type=int, default=-1, help="Cap optimizer steps (for smoke tests); -1 = full run")
    return parser.parse_args()


def build_dataset(name, seed=42) -> Dataset:
    """Build TRL prompt-completion SFT examples.

    Uses the (prompt, completion) conversational format rather than a single `messages`
    list so TRL masks the prompt tokens automatically and computes loss ONLY on the golden
    completion — without requiring `{% generation %}` tags in the chat template (Qwen3's
    template has none, which makes `assistant_only_loss=True` unusable)."""
    if name == "science":
        train_dir = "data/science_data/train_data"
        dataset = load_from_disk(train_dir)

        def format_example(example):
            return {
                "prompt": [example["messages"][0], example["messages"][1]],  # system, user
                "completion": [{"role": "assistant", "content": example["output_text"]}],
            }

        dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    elif name == "tooluse":
        train_dir = "data/tooluse_data/train_data"
        dataset = load_from_disk(train_dir)

        def format_example(example):
            return {
                "prompt": [{"role": "user", "content": example["prompt"]}],
                "completion": [{"role": "assistant", "content": "\n".join(example["golden_response"])}],
            }

        dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    else:
        raise ValueError(f"Invalid dataset name: {name}")

    dataset = dataset.shuffle(seed=seed)
    return dataset


if __name__ == "__main__":
    args = parse_args()
    init_path = args.init_model_path or args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        init_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    from nothinking import patch_tokenizer
    patch_tokenizer(tokenizer)

    dataset = build_dataset(args.dataset_name, args.seed)

    peft_config = None
    if args.peft:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
        max_length=3072,
        packing=False,
        # Loss is computed on the completion only: the (prompt, completion) dataset format masks
        # the prompt tokens automatically, so no assistant_only_loss / generation-tag template needed.
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
