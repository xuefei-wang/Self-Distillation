"""Log one training arm's live metrics from its log file into an existing MLflow experiment.

Companion to mlflow_arc_logger.py (which handles the base/sdft/sft trio). Use this for extra
arms trained on their own log — e.g. the SDFT-EMA arm. Reuses a run by name if it already
exists (safe to restart), tails the log, and streams the arm's metric dicts. Eval accuracies
are logged separately by the comprehensive both-splits eval pass.
"""
import argparse
import time

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_arc_logger import parse_metric_dicts, BAD_METRIC_CHARS


def get_or_create_run(client, exp_id, run_name):
    runs = client.search_runs([exp_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
    if runs:
        return runs[0].info.run_id
    return client.create_run(exp_id, run_name=run_name).info.run_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--arm", choices=["sdft", "sft"], default="sdft")
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="arc_armA")
    p.add_argument("--params", default="", help="k=v,k=v params to log once")
    p.add_argument("--poll", type=float, default=20.0)
    p.add_argument("--max_hours", type=float, default=8.0)
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    exp_id = exp.experiment_id if exp else client.create_experiment(args.experiment)
    run_id = get_or_create_run(client, exp_id, args.run_name)
    for kv in filter(None, args.params.split(",")):
        k, _, v = kv.partition("=")
        try:
            client.log_param(run_id, k.strip(), v.strip())
        except Exception:
            pass
    print(f"[mlflow] logging '{args.run_name}' ({run_id[:8]}) from {args.log}")

    seen = 0
    deadline = time.time() + args.max_hours * 3600
    import os
    while time.time() < deadline:
        if os.path.exists(args.log):
            with open(args.log, errors="ignore") as f:
                text = f.read()
            dicts = [d for arm, d in parse_metric_dicts(text) if arm == args.arm]
            for step in range(seen, len(dicts)):
                for k, v in dicts[step].items():
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        continue
                    try:
                        client.log_metric(run_id, BAD_METRIC_CHARS.sub("_", k), float(v), step=step)
                    except Exception:
                        pass
            seen = len(dicts)
            # stop once training is done (merge line appears) and metrics are drained
            if "training+merge done" in text and seen > 0:
                print(f"[mlflow] training complete; logged {seen} steps.")
                break
        time.sleep(args.poll)
    client.set_terminated(run_id, "FINISHED")


if __name__ == "__main__":
    main()
