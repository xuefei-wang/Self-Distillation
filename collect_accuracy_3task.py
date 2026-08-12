"""Collect the 3-task (Science -> Tool Use -> Medical) accuracies into accuracy_3task.json.

7 checkpoints x 3 tasks = 21 numbers. Missing evals are reported, not fatal."""
import json
import os

CKPTS = [
    "base",
    "sdft_science", "sdft_science_tooluse", "sdft_science_tooluse_medical",
    "sft_science", "sft_science_tooluse", "sft_science_tooluse_medical",
]
TASKS = ["science", "tooluse", "medical"]

out = {}
missing = []
for c in CKPTS:
    out[c] = {}
    for t in TASKS:
        p = f"results/fig3/eval/{c}/{t}/eval_results.json"
        if os.path.exists(p):
            out[c][t] = json.load(open(p))["accuracy"]
        else:
            missing.append(f"{c}/{t}")

os.makedirs("results/fig3", exist_ok=True)
json.dump(out, open("results/fig3/accuracy_3task.json", "w"), indent=2)
print(json.dumps(out, indent=2))
if missing:
    print("\nMISSING:", ", ".join(missing))
