"""Build the ARC-AGI-1 SDFT/SFT train set and the held-out eval set.

Two artifacts are written under data/arc_data/:

  train_data/  the 381 gold-demonstration tasks from the ARC-AGI-1 *training* split.
               Columns: messages=[{user: question}], output_text=<gold demonstration>.
               Mirrors the science schema so both main.py (SDFT) and sft_main.py (SFT)
               consume it. `output_text` is knowledge type #1 (Arm A: gold demonstration).

  eval_data/   the 400 ARC-AGI-1 *evaluation* split tasks (held out; disjoint from train).
               Columns: prompt=[{user: question}], task_id, test_outputs=<oracle grids>.

The `question` is rendered byte-exactly the way the gold-knowledge collection rendered it
(validated against gold_candidates.jsonl question_text): the rollout instruction plus a
compact JSON payload {"train":..., "test_inputs":...} with the test OUTPUTS stripped.
"""
import argparse
import json
import os
from datasets import Dataset

# rollout.md instruction, rendered verbatim as used during gold collection.
INSTRUCTION = (
    "Infer the transformation demonstrated by the ARC examples and apply it to every "
    "test input. Return exactly one JSON object with no extra keys, using the schema "
    '{"outputs":[<output grid for test 0>, ...]}.\n'
    "ARC task:\n"
)


def render_question(task: dict) -> str:
    """Reproduce the exact rollout question_text from a raw ARC task JSON.

    Verified byte-identical to gold_candidates.jsonl `question_text` (task 007bbfb7)."""
    payload = {
        "train": task["train"],
        "test_inputs": [t["input"] for t in task["test"]],
    }
    return INSTRUCTION + json.dumps(payload, separators=(",", ":"))


def load_task(data_root: str, split: str, task_id: str) -> dict:
    with open(os.path.join(data_root, split, f"{task_id}.json")) as f:
        return json.load(f)


def build_train(gold_path: str, data_root: str) -> Dataset:
    """Join gold demonstrations (training split) to their ARC payloads."""
    rows = []
    with open(gold_path) as f:
        for line in f:
            g = json.loads(line)
            task = load_task(data_root, "training", g["task_id"])
            question = render_question(task)
            rows.append({
                "messages": [{"role": "user", "content": question}],
                "output_text": g["demonstration"],
                "task_id": g["task_id"],
            })
    print(f"train: {len(rows)} gold-demonstration tasks")
    return Dataset.from_list(rows)


def build_split_eval(data_root: str, split: str) -> Dataset:
    """All tasks of an ARC split rendered as scorable eval rows (prompt + oracle outputs).

    Used for both the held-out `evaluation` split (val) and the `training` split (train-set
    eval, i.e. how well the model solves the tasks it was trained on)."""
    rows = []
    split_dir = os.path.join(data_root, split)
    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith(".json"):
            continue
        task_id = fname[:-5]
        task = load_task(data_root, split, task_id)
        rows.append({
            "prompt": [{"role": "user", "content": render_question(task)}],
            "task_id": task_id,
            "test_outputs": [t["output"] for t in task["test"]],
        })
    print(f"{split} eval: {len(rows)} tasks")
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold_path", default="data/arc_data/gold/gold_knowledge.jsonl")
    p.add_argument("--data_root", default="data/arc_agi_1")
    p.add_argument("--out_dir", default="data/arc_data")
    args = p.parse_args()

    train = build_train(args.gold_path, args.data_root)
    val_eval = build_split_eval(args.data_root, "evaluation")   # held-out (val)
    train_eval = build_split_eval(args.data_root, "training")   # train-set eval

    # test_outputs is a ragged list-of-grids; datasets handles nested lists fine. Save to disk.
    train.save_to_disk(os.path.join(args.out_dir, "train_data"))
    val_eval.save_to_disk(os.path.join(args.out_dir, "eval_data"))
    train_eval.save_to_disk(os.path.join(args.out_dir, "train_eval_data"))
    print(f"saved -> {args.out_dir}/train_data, eval_data (val), train_eval_data (train)")

    # quick sanity: token-ish length of the longest demonstration
    maxdemo = max(len(r["output_text"]) for r in train)
    maxq = max(len(r["messages"][0]["content"]) for r in train)
    print(f"max demonstration chars: {maxdemo} (~{maxdemo//4} tokens)")
    print(f"max question chars: {maxq} (~{maxq//4} tokens)")


if __name__ == "__main__":
    main()
