"""Sidecar: stream the running ARC Arm-A training log into MLflow.

The two arms (SDFT, SFT) write interleaved metric dicts to one driver log. This process is
NON-INVASIVE — it never touches the training jobs. It re-scans the log each poll, extracts the
Trainer's flat `{...}` metric dicts in order, classifies each into the SDFT or SFT arm by its
keys, and logs new ones to MLflow (stateless MlflowClient, so one process feeds several runs).
Final held-out eval accuracies are logged from results/arc_armA/eval/<name>/eval_results.json.

Because the whole log is on disk, restarting mid-run replays history from step 1.

Runs (experiment "arc_armA"): base, sdft_armA, sft_armA.
View with:  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
import argparse
import ast
import json
import os
import re
import time

import mlflow
from mlflow.tracking import MlflowClient

DICT_RE = re.compile(r"\{[^{}]*\}")

# classify a flat metric dict into an arm by its distinctive keys
SDFT_MARKERS = ("clipped_ratio", "kl_approx", "importance_sampling", "rewards")
SFT_MARKERS = ("mean_token_accuracy",)

# metric-name characters MLflow rejects -> underscore
BAD_METRIC_CHARS = re.compile(r"[^A-Za-z0-9_\-./: ]")


def classify(d: dict):
    keys = " ".join(d.keys())
    if any(m in keys for m in SDFT_MARKERS):
        return "sdft"
    if any(m in keys for m in SFT_MARKERS):
        return "sft"
    return None  # e.g. train_runtime summary or unrelated brace literal


def parse_metric_dicts(text: str):
    """Yield (arm, dict) for every flat metric dict in the log, in file order."""
    for m in DICT_RE.finditer(text):
        frag = m.group(0)
        if "loss" not in frag and "accuracy" not in frag and "reward" not in frag:
            continue  # cheap prefilter; skips set literals like {'kv_cache','weights'}
        try:
            d = ast.literal_eval(frag)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(d, dict):
            continue
        arm = classify(d)
        if arm:
            yield arm, d


def log_new(client, run_id, dicts, already, arm):
    """Log metric dicts beyond the `already` count for one arm; return new count."""
    for step in range(already, len(dicts)):
        d = dicts[step]
        for k, v in d.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            key = BAD_METRIC_CHARS.sub("_", k)
            try:
                client.log_metric(run_id, key, float(v), step=step)
            except Exception:
                pass
    return len(dicts)


def maybe_log_eval(client, run_id, name, logged_eval, oracle_by_task):
    """Re-score the eval from its saved responses with the CURRENT parser (consistent across
    all runs) once the responses appear, then log. Returns True when done."""
    if logged_eval:
        return True
    from eval_arc import rescore_dir
    results_dir = f"results/arc_armA/eval/{name}"
    if not os.path.exists(os.path.join(results_dir, "eval_responses.json")):
        return False
    try:
        r = rescore_dir(results_dir, oracle_by_task)
    except (json.JSONDecodeError, OSError, KeyError):
        return False
    client.log_metric(run_id, "eval_accuracy", float(r["accuracy"]))
    client.log_metric(run_id, "eval_num_correct", float(r["num_correct"]))
    client.log_metric(run_id, "eval_parse_failed", float(r["parse_failed"]))
    client.set_tag(run_id, "eval_done", "true")
    print(f"[mlflow] {name}: eval_accuracy={r['accuracy']:.4f} "
          f"({r['num_correct']}/{r['num_total']}, parse_failed={r['parse_failed']})")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="/tmp/claude-23749/arc_armA_full2.log")
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="arc_armA")
    p.add_argument("--poll", type=float, default=20.0)
    p.add_argument("--max_hours", type=float, default=8.0)
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    exp_id = exp.experiment_id if exp else client.create_experiment(args.experiment)

    # oracle grids for consistent re-scoring of every eval
    from datasets import load_from_disk
    _ev = load_from_disk("data/arc_data/eval_data")
    oracle_by_task = {x["task_id"]: x["test_outputs"] for x in _ev}

    common = {
        "base_model": "Qwen/Qwen3-8B", "lora_r": 16, "lora_alpha": 32,
        "num_train_epochs": 2, "learning_rate": 1e-4,
        "arm": "A_gold_demonstration", "task": "arc_agi_1",
        "eval_split": "evaluation(held-out,400)", "train_tasks": 381,
    }
    runs = {}
    for name, extra in [
        ("base", {"method": "base", "trainable": "none"}),
        ("sdft_armA", {"method": "SDFT", "max_prompt_length": 10240,
                       "max_completion_length": 4096}),
        ("sft_armA", {"method": "SFT", "max_length": 10240}),
    ]:
        r = client.create_run(exp_id, run_name=name)
        runs[name] = r.info.run_id
        params = {"method": extra.pop("method"), **common, **extra}
        if name == "base":
            params = {"method": "base", "base_model": common["base_model"],
                      "task": "arc_agi_1", "eval_split": common["eval_split"]}
        for k, v in params.items():
            client.log_param(runs[name], k, v)
    print(f"[mlflow] experiment '{args.experiment}' ({exp_id}) -> runs: "
          + ", ".join(f"{n}={rid[:8]}" for n, rid in runs.items()))

    seen = {"sdft": 0, "sft": 0}
    eval_done = {"base": False, "sdft_armA": False, "sft_armA": False}
    deadline = time.time() + args.max_hours * 3600

    while time.time() < deadline:
        if os.path.exists(args.log):
            with open(args.log, errors="ignore") as f:
                text = f.read()
            by_arm = {"sdft": [], "sft": []}
            for arm, d in parse_metric_dicts(text):
                by_arm[arm].append(d)
            seen["sdft"] = log_new(client, runs["sdft_armA"], by_arm["sdft"], seen["sdft"], "sdft")
            seen["sft"] = log_new(client, runs["sft_armA"], by_arm["sft"], seen["sft"], "sft")

        for name in ("base", "sdft_armA", "sft_armA"):
            eval_done[name] = maybe_log_eval(client, runs[name], name, eval_done[name], oracle_by_task)

        if all(eval_done.values()):
            print("[mlflow] all evals logged; finalizing runs.")
            break
        time.sleep(args.poll)

    for name, rid in runs.items():
        status = "FINISHED" if eval_done[name] else "KILLED"
        client.set_terminated(rid, status)
    print(f"[mlflow] done. steps logged: sdft={seen['sdft']} sft={seen['sft']}. "
          f"View: uv run mlflow ui --backend-store-uri {args.tracking_uri} --port 5000")


if __name__ == "__main__":
    main()
