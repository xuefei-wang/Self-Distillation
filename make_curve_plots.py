"""Render train/val curves over epochs. Colors are fixed per arm and shared across every figure;
all panels use the same figsize. Numbers load straight from each run's epoch_eval.json.

SFT arms are stitched across the continuation runs: epochs 1-6 from the *_e6 run + epochs 7-9 from
the *_e9 run (which resumed from the epoch-7 checkpoint). The 6->7 seam carries ~+-2 eval noise."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_N, VAL_N = 93, 307
FIGSIZE = (7.0, 5.0)
OUT = "/tmp/claude-23749/-home-xwang3-Projects-Self-Distillation/312be7cd-3970-4e21-9218-c1135dc80242/scratchpad"

# one fixed color per arm, reused in every figure
COLOR = {
    "SFT-gold":          "#1f77b4",
    "SFT-insight":       "#ff7f0e",
    "SDFT-gold-fixed":   "#2ca02c",
    "SDFT-insight_demo": "#d62728",
    "SDFT-insight-rule": "#9467bd",
}

def _read(path, epfilter=None):
    d = json.load(open(path))
    out = []
    for e in d["epochs"]:
        ep = int(e["epoch"])
        if epfilter and not epfilter(ep):
            continue
        out.append((ep, 100.0*e["train_correct"]/TRAIN_N, 100.0*e["val_correct"]/VAL_N))
    return out

def series(sources):
    """sources: list of (path, epfilter). Concatenate, sort by epoch."""
    rows = []
    for path, epf in sources:
        rows += _read(path, epf)
    rows.sort(key=lambda r: r[0])
    ep = [r[0] for r in rows]; tr = [r[1] for r in rows]; va = [r[2] for r in rows]
    return ep, tr, va

# SFT arms stitched to epoch 9 (e6: 1-6, e9: 7-9); SDFT arms are single 6-epoch runs.
CURVES = {
    "SFT-gold":          [("ckpt/arc_q35_sft_gold_e6_adapter/epoch_eval.json", None),
                          ("ckpt/arc_q35_sft_gold_e9_adapter/epoch_eval.json", lambda e: e >= 7)],
    "SFT-insight":       [("ckpt/arc_q35_sft_insight_e6_adapter/epoch_eval.json", None),
                          ("ckpt/arc_q35_sft_insight_e9_adapter/epoch_eval.json", lambda e: e >= 7)],
    "SDFT-gold-fixed":   [("ckpt/arc_q35_sdft_fixstp_gold_e6_adapter/epoch_eval.json", None)],
    "SDFT-insight_demo": [("ckpt/arc_q35_sdft_insightdemo_e6_adapter/epoch_eval.json", None)],
    "SDFT-insight-rule": [("ckpt/arc_q35_sdft_fixstp_insight_e6_adapter/epoch_eval.json", None)],
}
# system-vs-no-system ablation (6-epoch, question intact in both)
NOSYS = {
    "gold  [system,user]": ("ckpt/arc_q35_sft_gold_e6_adapter/epoch_eval.json",        "#1f77b4", "-"),
    "gold  [user] only":   ("ckpt/arc_q35_sft_gold_nosys_e6_adapter/epoch_eval.json",  "#1f77b4", "--"),
    "insight  [system,user]": ("ckpt/arc_q35_sft_insight_e6_adapter/epoch_eval.json",       "#ff7f0e", "-"),
    "insight  [user] only":   ("ckpt/arc_q35_sft_insight_nosys_e6_adapter/epoch_eval.json", "#ff7f0e", "--"),
}

def _finish(ax, which, title):
    ax.set_xlabel("epoch")
    ax.set_ylabel("train accuracy (%)  [/93 seen]" if which == "train" else "val accuracy (%)  [/307 held-out]")
    ax.set_title(title); ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

def plot_arms(arms, which, title, outfile):
    idx = 1 if which == "train" else 2
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xmax = 1
    for arm in arms:
        ep, tr, va = series(CURVES[arm]); xmax = max(xmax, max(ep))
        ax.plot(ep, (tr if idx == 1 else va), "o-", color=COLOR[arm], label=arm, linewidth=2, markersize=6)
    ax.set_xticks(range(1, xmax + 1)); _finish(ax, which, title)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig); print("wrote", outfile)

def plot_nosys(which, title, outfile):
    idx = 1 if which == "train" else 2
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, (path, color, ls) in NOSYS.items():
        ep, tr, va = series([(path, None)])
        ax.plot(ep, (tr if idx == 1 else va), marker="o", linestyle=ls, color=color, label=label, linewidth=2, markersize=5)
    ax.set_xticks(range(1, 7)); _finish(ax, which, title)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig); print("wrote", outfile)

four = ["SFT-gold", "SFT-insight", "SDFT-gold-fixed", "SDFT-insight_demo"]
two  = ["SDFT-insight_demo", "SDFT-insight-rule"]
plot_arms(four, "train", "SFT vs SDFT — train accuracy (SFT to epoch 9)", f"{OUT}/curves_4arms_train.png")
plot_arms(four, "val",   "SFT vs SDFT — val accuracy (SFT to epoch 9)",   f"{OUT}/curves_4arms_val.png")
plot_arms(two,  "train", "SDFT insight_demo vs insight-rule — train",     f"{OUT}/curves_insight_train.png")
plot_arms(two,  "val",   "SDFT insight_demo vs insight-rule — val",       f"{OUT}/curves_insight_val.png")
plot_nosys("train", "System message vs none (question intact) — train",   f"{OUT}/curves_sysablation_train.png")
plot_nosys("val",   "System message vs none (question intact) — val",     f"{OUT}/curves_sysablation_val.png")
