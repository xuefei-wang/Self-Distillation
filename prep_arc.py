"""Build the ARC-AGI-1 SDFT/SFT train set and the held-out eval set.

Two artifacts are written under data/arc_data/:

  train_data/  the 381 gold-demonstration tasks from the ARC-AGI-1 *training* split.
               Columns: messages=[{user: question}], output_text=<gold demonstration>.
               Mirrors the science schema so both main.py (SDFT) and sft_main.py (SFT)
               consume it. `output_text` is knowledge type #1 (Arm A: gold demonstration).

  eval_data/   the 400 ARC-AGI-1 *evaluation* split tasks (held out; disjoint from train).
               Columns: prompt=[{user: question}], task_id, test_outputs=<oracle grids>.

With --train_ids <path> the *training* split is instead cut into a train / val split:

  train_data/       the listed task_ids as gold-demonstration training rows.
  train_eval_data/  the same listed task_ids as scorable eval rows (train-set eval).
  val_eval_data/    the remaining training-split tasks as scorable eval rows (val).

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

# The system persona the 27B gold teacher and the published 9B eval number were generated with.
# No longer used in trained/eval prompts (the ablation made no-system the default, see _msgs);
# retained only as a generation-time crutch for eliciting good rollouts in gen_insight_sft.
SYSTEM = "You are a precise puzzle solver. Follow the output schema exactly."


def _msgs(question: str) -> list:
    """The chat turns every train/eval prompt carries: the question as a single user turn.
    No system turn by default -- the ablation showed the system persona is neutral-to-mildly-
    unhelpful (insight-nosys reached the grid's best delta), and the task instruction + schema
    already live in the user turn. SYSTEM is kept only as a generation-time crutch (gen_insight_sft)."""
    return [{"role": "user", "content": question}]


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


def build_train(gold_path: str, data_root: str, keep_ids: set | None = None) -> Dataset:
    """Join gold demonstrations (training split) to their ARC payloads.

    `keep_ids` restricts the rows to those task_ids (used by the --train_ids split)."""
    rows = []
    with open(gold_path) as f:
        for line in f:
            g = json.loads(line)
            if keep_ids is not None and g["task_id"] not in keep_ids:
                continue
            task = load_task(data_root, "training", g["task_id"])
            question = render_question(task)
            rows.append({
                "messages": _msgs(question),
                "output_text": g["demonstration"],
                "task_id": g["task_id"],
            })
    print(f"train: {len(rows)} gold-demonstration tasks")
    return Dataset.from_list(rows)


def build_train_corpus93(corpus_dir: str, data_root: str, keep_ids: set) -> Dataset:
    """Gold train rows from the arc-train93-split corpus (Qwen3.5-27B worked solutions).

    Uses golden_knowledge.jsonl's `style=="reasoning"` rows (the 93): output_text = the 27B
    `completion` (a non-thinking derivation ending in {"outputs":...}). The question is
    re-rendered from the raw task (byte-identical to the row's `question`) so it matches the
    eval rows and carries the system turn."""
    gold = {}
    with open(os.path.join(corpus_dir, "golden_knowledge.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("style") == "reasoning":
                gold[r["problem_id"]] = r["completion"]
    rows = []
    for tid in sorted(keep_ids):
        if tid not in gold:
            raise ValueError(f"train id {tid} has no reasoning-style gold completion in the corpus")
        task = load_task(data_root, "training", tid)
        rows.append({
            "messages": _msgs(render_question(task)),
            "output_text": gold[tid],
            "task_id": tid,
        })
    print(f"train (corpus93): {len(rows)} gold-completion tasks")
    return Dataset.from_list(rows)


def split_task_ids(data_root: str, split: str) -> list:
    """Sorted task_ids of an ARC split, read from the raw task files on disk."""
    split_dir = os.path.join(data_root, split)
    return sorted(f[:-5] for f in os.listdir(split_dir) if f.endswith(".json"))


def build_split_eval(data_root: str, split: str, keep_ids: set | None = None,
                     label: str | None = None) -> Dataset:
    """All tasks of an ARC split rendered as scorable eval rows (prompt + oracle outputs).

    Used for both the held-out `evaluation` split (val) and the `training` split (train-set
    eval, i.e. how well the model solves the tasks it was trained on). `keep_ids` restricts
    the rows to those task_ids (used by the --train_ids split)."""
    rows = []
    for task_id in split_task_ids(data_root, split):
        if keep_ids is not None and task_id not in keep_ids:
            continue
        task = load_task(data_root, split, task_id)
        rows.append({
            "prompt": _msgs(render_question(task)),
            "task_id": task_id,
            "test_outputs": [t["output"] for t in task["test"]],
        })
    print(f"{label or split} eval: {len(rows)} tasks")
    return Dataset.from_list(rows)


def load_train_ids(path: str) -> list:
    """Read the train-side task_ids from any of three forms:
      - a JSON array of ids;
      - the versioned split file splits/arc_train93_val307.json (a JSON object with a
        `train_93` key — the biased 93/307 split, see splits/README.md);
      - a newline-delimited list of ids."""
    with open(path) as f:
        text = f.read()
    stripped = text.strip()
    if stripped.startswith("{"):
        obj = json.loads(stripped)
        if "train_93" not in obj:
            raise ValueError(f"--train_ids {path}: JSON object has no 'train_93' key")
        ids = obj["train_93"]
    elif stripped.startswith("["):
        ids = json.loads(stripped)
    else:
        ids = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"--train_ids {path}: duplicate task_ids: {dupes}")
    return ids


def build_id_split(args, train_ids: list):
    """Cut the *training* split into the listed train ids and the remaining val ids.

    Raises if any train id is absent from the training split or has no gold demonstration:
    silently dropping one would shrink the train set unnoticed."""
    split_ids = split_task_ids(args.data_root, "training")
    split_set = set(split_ids)
    with open(args.gold_path) as f:
        gold_ids = {json.loads(line)["task_id"] for line in f}

    missing_split = [i for i in train_ids if i not in split_set]
    if missing_split:
        raise ValueError(
            f"--train_ids: {len(missing_split)} id(s) not in the training split "
            f"{os.path.join(args.data_root, 'training')}: {missing_split}"
        )
    missing_gold = [i for i in train_ids if i not in gold_ids]
    if missing_gold:
        raise ValueError(
            f"--train_ids: {len(missing_gold)} id(s) have no gold demonstration in "
            f"{args.gold_path}: {missing_gold}"
        )

    train_set = set(train_ids)
    val_set = split_set - train_set
    print(f"training split: {len(split_ids)} tasks -> train {len(train_set)} / val {len(val_set)}")

    train = build_train(args.gold_path, args.data_root, keep_ids=train_set)
    train_eval = build_split_eval(args.data_root, "training", keep_ids=train_set, label="train")
    val_eval = build_split_eval(args.data_root, "training", keep_ids=val_set, label="val")

    assert len(train) == len(train_set), f"train {len(train)} != {len(train_set)} train ids"
    assert len(train_eval) == len(train_set), f"train_eval {len(train_eval)} != {len(train_set)}"
    assert len(val_eval) == len(val_set), f"val_eval {len(val_eval)} != {len(val_set)}"
    assert not (train_set & val_set), "train and val ids overlap"
    assert train_set | val_set == split_set, "train + val ids != full training split"
    return train, train_eval, val_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold_path", default="data/arc_data/gold/gold_knowledge.jsonl")
    p.add_argument("--data_root", default="data/arc_agi_1")
    p.add_argument("--out_dir", default="data/arc_data")
    p.add_argument("--train_ids", default=None,
                   help="file listing the training-split task_ids to train on (JSON array or "
                        "one id per line); the remaining training tasks become val_eval_data")
    p.add_argument("--corpus93", default=None,
                   help="path to the arc-train93-split corpus dir (golden_knowledge.jsonl + "
                        "split_train93_val307.json). Builds train_data from the 27B reasoning "
                        "completions for train_93 and val_eval_data from val_307.")
    args = p.parse_args()

    # test_outputs is a ragged list-of-grids; datasets handles nested lists fine. Save to disk.
    if args.corpus93:
        split = json.load(open(os.path.join(args.corpus93, "split_train93_val307.json")))
        train_ids, val_ids = set(split["train_93"]), set(split["val_307"])
        split_set = set(split_task_ids(args.data_root, "training"))
        assert train_ids <= split_set and val_ids <= split_set, "split ids not all in training split"
        assert not (train_ids & val_ids), "train_93 and val_307 overlap"
        assert train_ids | val_ids == split_set, "train_93 + val_307 != training split"
        print(f"corpus93: train {len(train_ids)} / val {len(val_ids)}")
        train = build_train_corpus93(args.corpus93, args.data_root, train_ids)
        train_eval = build_split_eval(args.data_root, "training", keep_ids=train_ids, label="train")
        val_eval = build_split_eval(args.data_root, "training", keep_ids=val_ids, label="val")
        assert len(train) == len(train_ids) == 93, f"train {len(train)} != 93"
        assert len(train_eval) == len(train_ids) and len(val_eval) == len(val_ids)
        train.save_to_disk(os.path.join(args.out_dir, "train_data"))
        train_eval.save_to_disk(os.path.join(args.out_dir, "train_eval_data"))
        val_eval.save_to_disk(os.path.join(args.out_dir, "val_eval_data"))
        print(f"saved -> {args.out_dir}/train_data ({len(train)}), "
              f"train_eval_data ({len(train_eval)}), val_eval_data ({len(val_eval)})")
    elif args.train_ids:
        train_ids = load_train_ids(args.train_ids)
        print(f"--train_ids {args.train_ids}: {len(train_ids)} task ids")
        train, train_eval, val_eval = build_id_split(args, train_ids)
        train.save_to_disk(os.path.join(args.out_dir, "train_data"))
        train_eval.save_to_disk(os.path.join(args.out_dir, "train_eval_data"))
        val_eval.save_to_disk(os.path.join(args.out_dir, "val_eval_data"))
        print(f"saved -> {args.out_dir}/train_data ({len(train)}), "
              f"train_eval_data ({len(train_eval)}), val_eval_data ({len(val_eval)})")
    else:
        train = build_train(args.gold_path, args.data_root)
        val_eval = build_split_eval(args.data_root, "evaluation")   # held-out (val)
        train_eval = build_split_eval(args.data_root, "training")   # train-set eval
        train.save_to_disk(os.path.join(args.out_dir, "train_data"))
        val_eval.save_to_disk(os.path.join(args.out_dir, "eval_data"))
        train_eval.save_to_disk(os.path.join(args.out_dir, "train_eval_data"))
        print(f"saved -> {args.out_dir}/train_data ({len(train)}), "
              f"eval_data ({len(val_eval)}, val), train_eval_data ({len(train_eval)}, train)")

    # quick sanity: token-ish length of the longest demonstration
    maxdemo = max(len(r["output_text"]) for r in train)
    maxq = max(len(r["messages"][-1]["content"]) for r in train)  # [-1] = user turn; [0] is system
    print(f"max demonstration chars: {maxdemo} (~{maxdemo//4} tokens)")
    print(f"max question chars: {maxq} (~{maxq//4} tokens)")


if __name__ == "__main__":
    main()
