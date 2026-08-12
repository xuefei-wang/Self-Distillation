import argparse
import os
import json
import re
import string
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from eval_science import extract_xml_answer  # reuse the same <answer> extractor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on the medical (HuatuoGPT-o1) test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to model_path)")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def load_model_and_tokenizer(model_path, gpu_memory_utilization=0.8):
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        max_model_len=4096,
        trust_remote_code=True,
    )
    return llm, tokenizer


def load_test_data():
    path = 'data/medical_data/eval_data'
    print(f"Loading medical test dataset from {path}")
    return Dataset.load_from_disk(path)


def generate_responses(llm, tokenizer, prompts, max_new_tokens=2048, temperature=0.0):
    formatted_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def _norm(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for containment matching."""
    s = s.lower().strip()
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return re.sub(r"\s+", " ", s).strip()


def evaluate_correctness(responses, answers):
    """Correct if the (short, verifiable) ground-truth answer appears inside the model's
    extracted <answer>, after normalization. Suited to HuatuoGPT-o1 verifiable-problem answers,
    which are short medical terms/phrases (median ~19 chars)."""
    results = []
    for response, answer in zip(responses, answers):
        pred = _norm(extract_xml_answer(response))
        gt = _norm(answer)
        results.append(1 if gt and gt in pred else 0)
    return results


def main():
    args = parse_args()
    llm, tokenizer = load_model_and_tokenizer(args.model_path)
    test_data = load_test_data()

    prompts = [example['prompt'] for example in test_data]
    answers = [example['answer'] for example in test_data]

    responses = generate_responses(llm, tokenizer, prompts, args.max_new_tokens, args.temperature)

    print("\nEvaluating responses...")
    scores = evaluate_correctness(responses, answers)
    accuracy = np.mean(scores)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    output_dir = args.output_dir if args.output_dir else args.model_path
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump({
            "accuracy": float(accuracy),
            "num_correct": int(sum(scores)),
            "num_total": len(scores),
            "per_sample_scores": scores,
            "config": {"model_path": args.model_path, "max_new_tokens": args.max_new_tokens,
                       "temperature": args.temperature, "grading": "normalized-containment"},
        }, f, indent=2)
    with open(os.path.join(output_dir, "eval_responses.json"), "w") as f:
        json.dump([
            {"prompt": prompts[i], "response": responses[i], "answer": answers[i], "correct": bool(scores[i])}
            for i in range(len(responses))
        ], f, indent=2)
    print(f"\nSaved results to {output_dir}/eval_results.json")


if __name__ == "__main__":
    main()
