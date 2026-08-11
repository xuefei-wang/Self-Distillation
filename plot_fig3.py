"""Figure 3 reproduction plot: sequential continual learning (SDFT vs SFT).

Reads results/fig3/accuracy.json (produced by collect_accuracy.py) with per-checkpoint,
per-task accuracies, normalizes each task's curve (0 = base accuracy, 1 = best accuracy
across both methods and all stages), and draws a 2-panel figure mirroring the paper's
Figure 3. Also prints the raw + normalized tables.

Stage index: 0 = base, 1 = after Science, 2 = after Tool Use.
  SDFT: base -> sdft_science -> sdft_science_tooluse
  SFT : base -> sft_science  -> sft_science_tooluse
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC_PATH = "results/fig3/accuracy.json"
OUT_PATH = "results/fig3/figure3_reproduction.png"
TASKS = ["science", "tooluse"]
STAGE_LABELS = ["base", "+Science", "+ToolUse"]


def series(acc, method, task):
    """Accuracy of `method` on `task` at stages [base, after-science, after-tooluse]."""
    return [
        acc["base"][task],
        acc[f"{method}_science"][task],
        acc[f"{method}_science_tooluse"][task],
    ]


def main():
    acc = json.load(open(ACC_PATH))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    print("\n=== Figure 3 reproduction: raw accuracies and normalized values ===")
    for ax, task in zip(axes, TASKS):
        base = acc["base"][task]
        raw = {m: series(acc, m, task) for m in ["sdft", "sft"]}
        maxall = max(max(v) for v in raw.values())
        denom = (maxall - base) if (maxall - base) != 0 else 1.0

        print(f"\n[{task}] base={base:.4f} max={maxall:.4f}")
        for m, style in [("sdft", "-o"), ("sft", "--s")]:
            norm = [(x - base) / denom for x in raw[m]]
            ax.plot([0, 1, 2], norm, style, label=m.upper(), linewidth=2, markersize=7)
            print(f"  {m.upper():4s} raw={[round(x, 4) for x in raw[m]]}  norm={[round(x, 3) for x in norm]}")

        ax.set_title(f"{task} accuracy (normalized)")
        ax.set_xlabel("training stage")
        ax.set_ylabel("normalized accuracy")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(STAGE_LABELS)
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Figure 3 reproduction (2-task sequential, Qwen3-8B, LoRA)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
