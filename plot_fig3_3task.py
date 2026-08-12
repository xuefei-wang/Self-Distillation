"""3-task Figure 3: sequential continual learning over Science -> Tool Use -> Medical (SDFT vs SFT).

Reads results/fig3/accuracy_3task.json, normalizes each task per the paper (0 = base accuracy,
1 = best accuracy across both methods & all stages), and draws a 3-panel figure. Stages:
0=base, 1=+Science, 2=+ToolUse, 3=+Medical.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC_PATH = "results/fig3/accuracy_3task.json"
OUT_PATH = "results/fig3/figure3_reproduction_3task.png"
TASKS = ["science", "tooluse", "medical"]
STAGE_LABELS = ["base", "+Science", "+ToolUse", "+Medical"]

# checkpoint at each stage, per method
CKPT_AT_STAGE = {
    "sdft": ["base", "sdft_science", "sdft_science_tooluse", "sdft_science_tooluse_medical"],
    "sft":  ["base", "sft_science", "sft_science_tooluse", "sft_science_tooluse_medical"],
}


def series(acc, method, task):
    return [acc[c][task] for c in CKPT_AT_STAGE[method]]


def main():
    acc = json.load(open(ACC_PATH))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    print("\n=== 3-task Figure 3: raw accuracies and normalized values ===")
    for ax, task in zip(axes, TASKS):
        base = acc["base"][task]
        raw = {m: series(acc, m, task) for m in ["sdft", "sft"]}
        maxall = max(max(v) for v in raw.values())
        denom = (maxall - base) if (maxall - base) != 0 else 1.0

        print(f"\n[{task}] base={base:.4f} max={maxall:.4f}")
        for m, style in [("sdft", "-o"), ("sft", "--s")]:
            norm = [(x - base) / denom for x in raw[m]]
            ax.plot(range(4), norm, style, label=m.upper(), linewidth=2, markersize=7)
            print(f"  {m.upper():4s} raw={[round(x,4) for x in raw[m]]}  norm={[round(x,3) for x in norm]}")

        ax.set_title(f"{task} accuracy (normalized)")
        ax.set_xlabel("training stage")
        ax.set_ylabel("normalized accuracy")
        ax.set_xticks(range(4))
        ax.set_xticklabels(STAGE_LABELS, rotation=15)
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Figure 3 reproduction — 3-task sequential (Qwen3-8B, LoRA): Science -> Tool Use -> Medical")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
