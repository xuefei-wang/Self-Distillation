"""Render train/val curves over 6 epochs. Colors are fixed per arm and shared across every
figure; all panels use the same figsize so the plots line up. Numbers load straight from each
run's epoch_eval.json (no hand-transcription)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATHS = {
    "SFT-gold":          "ckpt/arc_q35_sft_gold_e6_adapter/epoch_eval.json",
    "SFT-insight":       "ckpt/arc_q35_sft_insight_e6_adapter/epoch_eval.json",
    "SDFT-gold-fixed":   "ckpt/arc_q35_sdft_fixstp_gold_e6_adapter/epoch_eval.json",
    "SDFT-insight_demo": "ckpt/arc_q35_sdft_insightdemo_e6_adapter/epoch_eval.json",
    "SDFT-insight-rule": "ckpt/arc_q35_sdft_fixstp_insight_e6_adapter/epoch_eval.json",
}
# one fixed color per arm, reused in every figure
COLOR = {
    "SFT-gold":          "#1f77b4",
    "SFT-insight":       "#ff7f0e",
    "SDFT-gold-fixed":   "#2ca02c",
    "SDFT-insight_demo": "#d62728",
    "SDFT-insight-rule": "#9467bd",
}
TRAIN_N, VAL_N = 93, 307
FIGSIZE = (7.0, 5.0)   # identical for every panel

def load(arm):
    d = json.load(open(PATHS[arm]))
    ep = [int(e["epoch"]) for e in d["epochs"]]
    tr = [100.0 * e["train_correct"] / TRAIN_N for e in d["epochs"]]
    va = [100.0 * e["val_correct"] / VAL_N for e in d["epochs"]]
    return ep, tr, va

def plot(arms, which, title, outfile):
    idx = 1 if which == "train" else 2   # tuple position from load()
    ylab = "train accuracy (%)  [/93 seen]" if which == "train" else "val accuracy (%)  [/307 held-out]"
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for arm in arms:
        data = load(arm)
        ax.plot(data[0], data[idx], "o-", color=COLOR[arm], label=arm, linewidth=2, markersize=6)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print("wrote", outfile)

OUT = "/tmp/claude-23749/-home-xwang3-Projects-Self-Distillation/312be7cd-3970-4e21-9218-c1135dc80242/scratchpad"
four = ["SFT-gold", "SFT-insight", "SDFT-gold-fixed", "SDFT-insight_demo"]
two  = ["SDFT-insight_demo", "SDFT-insight-rule"]
plot(four, "train", "SFT vs SDFT — train accuracy over epochs", f"{OUT}/curves_4arms_train.png")
plot(four, "val",   "SFT vs SDFT — val accuracy over epochs",   f"{OUT}/curves_4arms_val.png")
plot(two,  "train", "SDFT insight_demo vs insight-rule — train", f"{OUT}/curves_insight_train.png")
plot(two,  "val",   "SDFT insight_demo vs insight-rule — val",   f"{OUT}/curves_insight_val.png")
