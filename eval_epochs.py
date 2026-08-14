"""Score every per-epoch checkpoint of a trained LoRA run on both ARC eval splits.

Post-hoc driver, not an inline callback: SFT never loads a vLLM engine and SDFT's colocated
engine is memory-tight, so generating inside the training process would OOM the 48GB GPU.
Instead, training writes checkpoint-<step>/ adapters (save_strategy="epoch"), and this script
walks them afterwards, merging + evaluating each one in a FRESH subprocess so the vLLM engine
and the merged full model are torn down between epochs.

Scoring is not reimplemented here: merge_lora.py and eval_arc.py are called as subprocesses
and their eval_results.json is read back.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Per-epoch eval of a LoRA run's checkpoints")
    p.add_argument("--base", required=True, help="Base/init model the adapter was trained on")
    p.add_argument("--adapter_dir", required=True,
                   help="Training output_dir holding the per-epoch checkpoint-<step>/ subdirs")
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES value for the merge/eval subprocesses")
    p.add_argument("--train_eval_data", default="data/arc_data/train_eval_data",
                   help="HF dataset dir for the seen-tasks split (prompt/task_id/test_outputs)")
    p.add_argument("--val_eval_data", default="data/arc_data/val_eval_data",
                   help="HF dataset dir for the held-out split (prompt/task_id/test_outputs)")
    p.add_argument("--out", default=None,
                   help="Summary json path. Defaults to <adapter_dir>/epoch_eval.json")
    p.add_argument("--merged_root", default=None,
                   help="Where to write the merged per-epoch models. Defaults to "
                        "<adapter_dir>/_merged")
    p.add_argument("--keep_merged", action="store_true",
                   help="Keep each merged full model (~18GB each). Default: delete after eval.")
    p.add_argument("--max_new_tokens", type=int, default=12288, help="Passed to eval_arc.py")
    p.add_argument("--max_model_len", type=int, default=20480, help="Passed to eval_arc.py")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="Passed to eval_arc.py")
    return p.parse_args()


def find_checkpoints(adapter_dir):
    """checkpoint-<global_step> dirs holding an adapter, ordered by step (= epoch order)."""
    found = []
    for name in os.listdir(adapter_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if not m:
            continue
        path = os.path.join(adapter_dir, name)
        if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
            print(f"[skip] {path} has no adapter_model.safetensors")
            continue
        found.append((int(m.group(1)), path))
    return sorted(found)


def checkpoint_epoch(ckpt_path, fallback):
    """Epoch number recorded by the Trainer, else the 1-based checkpoint index."""
    state = os.path.join(ckpt_path, "trainer_state.json")
    if os.path.exists(state):
        with open(state) as f:
            ep = json.load(f).get("epoch")
        if ep is not None:
            return round(float(ep), 4)
    return fallback


def run(cmd, env):
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env)
    return r.returncode


def main():
    args = parse_args()
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    py = sys.executable

    ckpts = find_checkpoints(args.adapter_dir)
    if not ckpts:
        print(f"ERROR: no checkpoint-*/ with an adapter under {args.adapter_dir}. "
              "Was the run trained with save_strategy='epoch'?")
        return 1
    print(f"Found {len(ckpts)} epoch checkpoints: {[os.path.basename(p) for _, p in ckpts]}")

    merged_root = args.merged_root or os.path.join(args.adapter_dir, "_merged")
    out_path = args.out or os.path.join(args.adapter_dir, "epoch_eval.json")
    rows = []

    for i, (step, ckpt) in enumerate(ckpts, start=1):
        epoch = checkpoint_epoch(ckpt, i)
        merged = os.path.join(merged_root, f"epoch{i}")
        rc = run([py, "merge_lora.py", "--base", args.base, "--adapter", ckpt, "--out", merged], env)
        if rc != 0:
            print(f"ERROR: merge failed for {ckpt} (rc={rc}); skipping this epoch")
            continue

        row = {"epoch": epoch, "step": step, "checkpoint": ckpt}
        for split, data_dir in (("train", args.train_eval_data), ("val", args.val_eval_data)):
            if not os.path.exists(data_dir):
                print(f"[skip] {split} split: {data_dir} does not exist")
                continue
            res_dir = os.path.join(merged_root, f"epoch{i}_{split}")
            rc = run([py, "eval_arc.py", "--model_path", merged, "--data_dir", data_dir,
                      "--output_dir", res_dir,
                      "--max_new_tokens", str(args.max_new_tokens),
                      "--max_model_len", str(args.max_model_len),
                      "--gpu_memory_utilization", str(args.gpu_memory_utilization)], env)
            if rc != 0:
                print(f"ERROR: eval failed for {split} at {ckpt} (rc={rc})")
                continue
            with open(os.path.join(res_dir, "eval_results.json")) as f:
                res = json.load(f)
            row[f"{split}_acc"] = res["accuracy"]
            row[f"{split}_correct"] = res["num_correct"]
            row[f"{split}_total"] = res["num_total"]

        rows.append(row)
        if not args.keep_merged:
            shutil.rmtree(merged, ignore_errors=True)

        # Write incrementally so a crash mid-grid still leaves the finished epochs on disk.
        with open(out_path, "w") as f:
            json.dump({"base": args.base, "adapter_dir": args.adapter_dir, "epochs": rows}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"per-epoch results for {args.adapter_dir}")
    for r in rows:
        print(f"  epoch {r['epoch']} (step {r['step']}): "
              f"train {r.get('train_correct', '-')}/{r.get('train_total', '-')} "
              f"({r.get('train_acc', float('nan')):.4f})  "
              f"val {r.get('val_correct', '-')}/{r.get('val_total', '-')} "
              f"({r.get('val_acc', float('nan')):.4f})")
    print("=" * 60)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
