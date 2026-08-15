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


def extract_outputs(text: str):
    """The FIRST JSON object carrying an "outputs" key, scanning with raw_decode so surrounding
    prose / ```json fences are tolerated. This is the arc-train93-split corpus grader: there is
    NO salvage path — no "last grid-shaped thing", no reassembling integers — because that graded
    truncated non-answers as correct (22% of positively-rewarded rollouts had no answer). Returns
    the whole dict (so the caller can enforce the one-key rule), or None."""
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        j = text.find("{", i)
        if j == -1:
            break
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and "outputs" in obj:
            return obj
        i = max(end, j + 1)
    return None


def grids_equal(a, b) -> bool:
    """Exact equality of two ARC grids (lists of lists of ints)."""
    try:
        if len(a) != len(b):
            return False
        return all(list(ra) == list(rb) for ra, rb in zip(a, b))
    except TypeError:
        return False


def _valid_grid(g) -> bool:
    """Rectangular, 1..30 rows/cols, integers 0-9. bool is a subclass of int and is rejected."""
    if not isinstance(g, list) or not g or len(g) > 30:
        return False
    width = None
    for row in g:
        if not isinstance(row, list) or not row or len(row) > 30:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v <= 9):
                return False
    return True


def format_valid(parsed, oracle_outputs) -> bool:
    """True iff `parsed` is a strict {"outputs": [grid, ...]} of the right length with well-formed
    grids — i.e. the model produced the required output FORMAT, regardless of whether the grids
    are correct. Reported alongside accuracy so a format failure (e.g. the right grid emitted with
    the wrong nesting or extra keys) can be told apart from a reasoning failure; one accuracy
    number reads both as 0 and hides format drift (see arc-train93-split/README.md)."""
    if not isinstance(parsed, dict) or set(parsed.keys()) != {"outputs"}:
        return False
    outs = parsed["outputs"]
    if not isinstance(outs, list) or len(outs) != len(oracle_outputs):
        return False
    return all(_valid_grid(g) for g in outs)


def score_task(parsed, oracle_outputs) -> int:
    """1 iff the parsed object is a strict {"outputs": [grid, ...]} that matches the oracle for
    every test input. Enforces the corpus grader: exactly the one key, a list of the right length,
    each grid well-formed. Any deviation scores 0 (no partial credit, no salvage)."""
    if not format_valid(parsed, oracle_outputs):
        return 0
    return int(all(grids_equal(p, o) for p, o in zip(parsed["outputs"], oracle_outputs)))


def rescore_dir(results_dir: str, oracle_by_task: dict) -> dict:
    """Re-score a finished eval from its saved eval_responses.json with the CURRENT parser,
    without re-generating. Overwrites eval_results.json and returns the summary. Lets every
    run be scored by one final parser version regardless of when it was generated."""
    resp_path = os.path.join(results_dir, "eval_responses.json")
    with open(resp_path) as f:
        responses = json.load(f)
    scores, parse_fail, per_task = [], 0, []
    fmt_valid, trunc = [], []
    for item in responses:
        tid = item["task_id"]
        oracle_outputs = oracle_by_task[tid]
        parsed = extract_outputs(item["response"])
        if parsed is None:
            parse_fail += 1
            s = 0
            fv = False
        else:
            s = score_task(parsed, oracle_outputs)
            fv = format_valid(parsed, oracle_outputs)
        scores.append(s)
        fmt_valid.append(fv)
        # finish_reason is saved by newer runs; older eval_responses.json lack it, in which case
        # truncation cannot be recovered from text and is reported as null rather than guessed.
        fr = item.get("finish_reason")
        is_trunc = item["truncated"] if "truncated" in item else (fr == "length" if fr else None)
        trunc.append(is_trunc)
        row = {"task_id": tid, "correct": bool(s), "format_valid": fv}
        if is_trunc is not None:
            row["truncated"] = bool(is_trunc)
        per_task.append(row)
    summary = {
        "accuracy": float(np.mean(scores)) if scores else 0.0,
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "parse_failed": parse_fail,
        "num_format_valid": int(sum(fmt_valid)),
        "format_valid_frac": float(np.mean(fmt_valid)) if fmt_valid else 0.0,
        "per_task": per_task,
        "rescored": True,
    }
    known_trunc = [t for t in trunc if t is not None]
    if known_trunc:
        summary["num_truncated"] = int(sum(known_trunc))
        summary["truncated_frac"] = float(np.mean(known_trunc))
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
    # Qwen3.5 ends a chat turn on <|im_end|> (eos), but its checkpoint ships no
    # generation_config and can also emit <|endoftext|> (248044) as a stop; include both so a
    # completion is never left running to the token budget by a missed stop id.
    stop_ids = [i for i in (tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|endoftext|>"))
                if i is not None and i >= 0]
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop_token_ids=stop_ids or None,
    )
    print(f"Generating for {len(formatted)} ARC eval tasks...")
    outputs = llm.generate(formatted, sampling)
    responses = [o.outputs[0].text for o in outputs]
    # "length" => the completion hit max_new_tokens and was cut off before it could emit its
    # answer/EOS; tracked so truncation is visible instead of silently counting as wrong.
    finish_reasons = [o.outputs[0].finish_reason for o in outputs]
    truncated = [fr == "length" for fr in finish_reasons]

    scores, parse_fail, fmt_valid = [], 0, []
    for resp, orc in zip(responses, oracle):
        parsed = extract_outputs(resp)
        if parsed is None:
            parse_fail += 1
            scores.append(0)
            fmt_valid.append(False)
        else:
            scores.append(score_task(parsed, orc))
            fmt_valid.append(format_valid(parsed, orc))

    accuracy = float(np.mean(scores))
    fmt_frac = float(np.mean(fmt_valid)) if fmt_valid else 0.0
    trunc_frac = float(np.mean(truncated)) if truncated else 0.0
    print("\n" + "=" * 60)
    print("ARC-AGI-1 evaluation-split results:")
    print(f"  Tasks:        {len(scores)}")
    print(f"  Correct:      {sum(scores)}")
    print(f"  Parse failed: {parse_fail}")
    print(f"  Format valid: {sum(fmt_valid)} ({fmt_frac*100:.2f}%)")
    print(f"  Truncated:    {sum(truncated)} ({trunc_frac*100:.2f}%)")
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
            "num_format_valid": int(sum(fmt_valid)),
            "format_valid_frac": fmt_frac,
            "num_truncated": int(sum(truncated)),
            "truncated_frac": trunc_frac,
            "per_task": [
                {"task_id": t, "correct": bool(s), "format_valid": bool(fv),
                 "truncated": bool(tr)}
                for t, s, fv, tr in zip(task_ids, scores, fmt_valid, truncated)
            ],
            "config": vars(args),
        }, f, indent=2)
    with open(os.path.join(out_dir, "eval_responses.json"), "w") as f:
        json.dump([
            {"task_id": task_ids[i], "response": responses[i], "correct": bool(scores[i]),
             "finish_reason": finish_reasons[i], "truncated": bool(truncated[i])}
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved results to {out_dir}/eval_results.json")


if __name__ == "__main__":
    main()
