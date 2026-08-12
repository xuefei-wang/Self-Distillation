"""Figure 3 in the paper's layout: two panels (SDFT | SFT), one line per task, over the
sequential training stages, per-task linearly normalized (0 = base accuracy, 1 = max across
both algorithms).

Differences from the paper's Figure 3 (data limitations, not style):
  * x-axis is discrete TRAINING STAGE (base, +Science, +ToolUse, +Medical) rather than continuous
    gradient steps — we evaluated only at stage boundaries, so the lines connect 4 points instead
    of tracing the within-stage curve.
  * 1 seed -> no confidence bands.
  * Task order here is Science -> Tool Use -> Medical (our choice); the paper uses
    Tool Use -> Science -> Medical.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC_PATH = "results/fig3/accuracy_3task.json"
OUT_PATH = "results/fig3/figure3_reproduction_paper_style.png"

TASKS = ["tooluse", "science", "medical"]          # legend/plot order (matches paper's grouping)
TASK_LABEL = {"tooluse": "Tool Use", "science": "Science Q&A", "medical": "Medical"}
# paper-like colours: Tool Use light cyan, Science medium blue.
# Medical is de-emphasized (grey, dashed): its per-task normalization is degenerate here
# (base 0.181 ~= max 0.195 -> tiny denominator), so its normalized line is noise, not signal.
TASK_COLOR = {"tooluse": "#4Fc3d6", "science": "#3a78b5", "medical": "#b3b3b3"}
# per-task line style: (linestyle, linewidth, alpha, markersize, zorder)
TASK_STYLE = {
    "tooluse": ("-", 2.4, 1.0, 7, 3),
    "science": ("-", 2.4, 1.0, 7, 3),
    "medical": ("--", 1.6, 0.7, 5, 2),
}

# our sequential order and the checkpoint at each stage, per method
STAGE_LABELS = ["base", "+Science", "+ToolUse", "+Medical"]
PHASE_LABELS = ["Train on\nScience", "Train on\nTool Use", "Train on\nMedical"]
CKPT_AT_STAGE = {
    "sdft": ["base", "sdft_science", "sdft_science_tooluse", "sdft_science_tooluse_medical"],
    "sft":  ["base", "sft_science", "sft_science_tooluse", "sft_science_tooluse_medical"],
}


def normalizers(acc):
    """Per-task (base, denom) with 0=base, 1=max across BOTH methods & all stages."""
    norm = {}
    for task in TASKS:
        base = acc["base"][task]
        allvals = [acc[c][task] for m in CKPT_AT_STAGE for c in CKPT_AT_STAGE[m]]
        maxall = max(allvals)
        denom = (maxall - base) or 1.0
        norm[task] = (base, denom)
    return norm


def main():
    acc = json.load(open(ACC_PATH))
    norm = normalizers(acc)
    x = list(range(4))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, method, title in zip(axes, ["sdft", "sft"], ["(a) SDFT", "(b) SFT"]):
        for task in TASKS:
            base, denom = norm[task]
            y = [(acc[c][task] - base) / denom for c in CKPT_AT_STAGE[method]]
            ls, lw, al, ms, zo = TASK_STYLE[task]
            label = TASK_LABEL[task] + (" (norm. degenerate)" if task == "medical" else "")
            ax.plot(x, y, ls, marker="o", color=TASK_COLOR[task], linewidth=lw, markersize=ms,
                    alpha=al, label=label, zorder=zo)

        # phase dividers (between the 3 training stages) + top labels, paper-style
        for xd in (0.5, 1.5, 2.5):
            ax.axvline(xd, color="0.5", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
        for cx, plabel in zip((0.5, 1.5, 2.5), PHASE_LABELS):
            ax.text(cx, 1.16, plabel, ha="center", va="top", fontsize=9, color="0.25")

        ax.axhline(0.0, color="0.6", linewidth=0.8, alpha=0.6, zorder=1)
        ax.set_title(title, y=-0.22, fontsize=12)
        ax.set_xlabel("Training stage")
        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_LABELS, fontsize=9)
        ax.set_ylim(-0.6, 1.25)
        ax.grid(axis="y", alpha=0.25)
        ax.margins(x=0.02)
    axes[0].set_ylabel("Normalized Performance")
    axes[1].legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle("Figure 3 reproduction (Qwen3-8B, LoRA, 1 seed): sequential continual learning",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")

    # console: the story in raw + normalized
    for method in ["sdft", "sft"]:
        print(f"\n[{method.upper()}]")
        for task in TASKS:
            base, denom = norm[task]
            raw = [round(acc[c][task], 3) for c in CKPT_AT_STAGE[method]]
            nrm = [round((acc[c][task] - base) / denom, 2) for c in CKPT_AT_STAGE[method]]
            print(f"  {TASK_LABEL[task]:11s} raw={raw}  norm={nrm}")


if __name__ == "__main__":
    main()
