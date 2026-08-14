"""Evaluate a model on the held-out ARC-AGI-1 evaluation split (400 tasks).

Metric: strict exact match. The model's parsed `outputs` must equal the oracle grids for
every test input of a task; a task scores 1 iff all its test outputs match exactly. This is
the standard ARC scoring and matches how the gold demonstrations were oracle-verified.

Data comes from data/arc_data/eval_data (built by prep_arc.py): columns prompt / task_id /
test_outputs. Prompts are rendered with the same rollout instruction used at train time, so
the eval distribution matches training.
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a model on the ARC-AGI-1 eval split")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None, help="Defaults to model_path")
    p.add_argument("--data_dir", type=str, default="data/arc_data/eval_data",
                   help="HF dataset dir to evaluate: eval_data (held-out val) or train_eval_data "
                        "(train-split). Must have columns prompt / task_id / test_outputs.")
    p.add_argument("--max_new_tokens", type=int, default=12288,
                   help="Generous budget: SFT imitates the long gold demonstrations (~up to 11.5k "
                        "tokens) and gets truncated before its final JSON at smaller caps. Models "
                        "that emit EOS early (base, SDFT) finish fast regardless, so this is fair.")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--max_model_len", type=int, default=20480,
                   help="vLLM context; long ARC prompts (~4k) + generation (~4k) need headroom")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return p.parse_args()


def _iter_brace_objects_with_outputs(text: str):
    """Yield every balanced {...} slice that json-parses to a dict containing "outputs",
    in left-to-right order. Grids use '[' brackets, so balancing only braces is sufficient."""
    for m in re.finditer(r'"outputs"', text):
        start = text.rfind("{", 0, m.start())
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "outputs" in obj:
                        yield obj
                    break


def extract_outputs(text: str):
    """Extract the model's final answer grids, matching the collection pipeline's parser.

    The rollout asks for exactly one JSON object; the canonical answer is the LAST fenced
    ```json ... ``` block. We therefore prefer the last fenced code block that parses to a
    dict with "outputs"; if none is fenced, fall back to the last brace-balanced "outputs"
    object anywhere in the text. Returns the outputs list, or None."""
    # 1) last fenced code block (```json ... ``` or bare ``` ... ```) with an outputs object
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    for block in reversed(fences):
        objs = list(_iter_brace_objects_with_outputs(block))
        if objs:
            return objs[-1]["outputs"]
    # 2) last strict outputs object anywhere in the raw text
    objs = list(_iter_brace_objects_with_outputs(text))
    if objs:
        return objs[-1]["outputs"]
    # 3) lenient recovery for malformed JSON (e.g. a stray bracket): after the last "outputs",
    #    pull the integer rows and reassemble a single grid. Only helps single-test tasks — a
    #    multi-test task recovered as one grid fails the length check in score_task, so this
    #    never awards false credit. Fixes raw-JSON answers that don't survive json.loads.
    idx = text.rfind('"outputs"')
    if idx != -1:
        rows = re.findall(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]", text[idx:])
        grid = []
        for row in rows:
            try:
                grid.append(json.loads(row))
            except json.JSONDecodeError:
                grid = []
                break
        if grid:
            return [grid]  # wrap the recovered 2D grid as a one-grid outputs list
    return None


def normalize_outputs(outputs):
    """Match the collection pipeline's leniency: a response that emits a single grid directly
    (outputs = [[int,...], ...], a 2D list of ints) instead of a list-of-grids gets wrapped to
    [grid]. Well-formed list-of-grids (3D) is returned unchanged."""
    if (isinstance(outputs, list) and outputs
            and isinstance(outputs[0], list) and outputs[0]
            and all(isinstance(v, int) for v in outputs[0])):
        return [outputs]
    return outputs


def grids_equal(a, b) -> bool:
    """Exact equality of two ARC grids (lists of lists of ints)."""
    try:
        if len(a) != len(b):
            return False
        return all(list(ra) == list(rb) for ra, rb in zip(a, b))
    except TypeError:
        return False


def score_task(parsed_outputs, oracle_outputs) -> int:
    """1 iff parsed outputs match the oracle for every test input, else 0."""
    parsed_outputs = normalize_outputs(parsed_outputs)
    if not isinstance(parsed_outputs, list) or len(parsed_outputs) != len(oracle_outputs):
        return 0
    return int(all(grids_equal(p, o) for p, o in zip(parsed_outputs, oracle_outputs)))


def rescore_dir(results_dir: str, oracle_by_task: dict) -> dict:
    """Re-score a finished eval from its saved eval_responses.json with the CURRENT parser,
    without re-generating. Overwrites eval_results.json and returns the summary. Lets every
    run be scored by one final parser version regardless of when it was generated."""
    resp_path = os.path.join(results_dir, "eval_responses.json")
    with open(resp_path) as f:
        responses = json.load(f)
    scores, parse_fail, per_task = [], 0, []
    for item in responses:
        tid = item["task_id"]
        parsed = extract_outputs(item["response"])
        if parsed is None:
            parse_fail += 1
            s = 0
        else:
            s = score_task(parsed, oracle_by_task[tid])
        scores.append(s)
        per_task.append({"task_id": tid, "correct": bool(s)})
    summary = {
        "accuracy": float(np.mean(scores)) if scores else 0.0,
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "parse_failed": parse_fail,
        "per_task": per_task,
        "rescored": True,
    }
    with open(os.path.join(results_dir, "eval_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=torch.bfloat16,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    data = Dataset.load_from_disk(args.data_dir)
    prompts = [ex["prompt"] for ex in data]
    task_ids = [ex["task_id"] for ex in data]
    oracle = [ex["test_outputs"] for ex in data]

    formatted = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    print(f"Generating for {len(formatted)} ARC eval tasks...")
    outputs = llm.generate(formatted, sampling)
    responses = [o.outputs[0].text for o in outputs]

    scores, parse_fail = [], 0
    for resp, orc in zip(responses, oracle):
        parsed = extract_outputs(resp)
        if parsed is None:
            parse_fail += 1
            scores.append(0)
        else:
            scores.append(score_task(parsed, orc))

    accuracy = float(np.mean(scores))
    print("\n" + "=" * 60)
    print("ARC-AGI-1 evaluation-split results:")
    print(f"  Tasks:        {len(scores)}")
    print(f"  Correct:      {sum(scores)}")
    print(f"  Parse failed: {parse_fail}")
    print(f"  Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    out_dir = args.output_dir or args.model_path
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump({
            "accuracy": accuracy,
            "num_correct": int(sum(scores)),
            "num_total": len(scores),
            "parse_failed": parse_fail,
            "per_task": [
                {"task_id": t, "correct": bool(s)} for t, s in zip(task_ids, scores)
            ],
            "config": vars(args),
        }, f, indent=2)
    with open(os.path.join(out_dir, "eval_responses.json"), "w") as f:
        json.dump([
            {"task_id": task_ids[i], "response": responses[i], "correct": bool(scores[i])}
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved results to {out_dir}/eval_results.json")


if __name__ == "__main__":
    main()
