"""Collect the 10 per-(checkpoint, task) accuracies into results/fig3/accuracy.json."""
import json
import os

CKPTS = ["base", "sdft_science", "sdft_science_tooluse", "sft_science", "sft_science_tooluse"]
TASKS = ["science", "tooluse"]

out = {}
missing = []
for c in CKPTS:
    out[c] = {}
    for t in TASKS:
        p = f"results/fig3/eval/{c}/{t}/eval_results.json"
        if not os.path.exists(p):
            missing.append(p)
            continue
        out[c][t] = json.load(open(p))["accuracy"]

os.makedirs("results/fig3", exist_ok=True)
json.dump(out, open("results/fig3/accuracy.json", "w"), indent=2)
print(json.dumps(out, indent=2))
if missing:
    print("\nWARNING: missing eval results:")
    for m in missing:
        print("  ", m)
