# ARC-AGI-1 train/val split (versioned)

`arc_train93_val307.json` pins the 93/307 split of the ARC-AGI-1 **training** set used by
the grid. It carries the id lists (`train_93`, `val_307`), per-task bands (`split_problems`,
consumed by `eval_epochs.py --band_map`), and the selection rule (`split.rule`).

Reproduce the datasets from it:

```bash
python prep_arc.py --train_ids splits/arc_train93_val307.json
```

`prep_arc.load_train_ids` reads the `train_93` key directly; `build_id_split` then makes
`train_data` / `train_eval_data` (the 93) and `val_eval_data` (the remaining 307), and asserts
the two id sets are disjoint and cover the full training split.

## The split is SELECTION-BIASED, not random

`split.rule`:

> train = the 93 tasks with a full golden reasoning trace (style=='reasoning');
> val = the other 307 ARC-AGI-1 training tasks.

So `train` and `val` are **not exchangeable populations**. Measured band distributions:

- train_93: mostly `delta` (the band with solve-rate headroom); **zero** `never`/`usual`.
- val_307: ~58% `never` (tasks essentially never solved), plus `delta`, `mixed`, `usual`, `rare`.

Consequences to keep in mind when reading results:

- Val accuracy is **not** an unbiased held-out estimate of train-task performance, and the
  train-vs-val curve is **not** a generalization gap — the two axes measure different task
  mixes. Compare bands (esp. `delta`), not the raw 307-aggregate.
- The 93 were selected by the insight pipeline (tasks with a usable reasoning trace), i.e. the
  selection code is `gen_insight_sft.py` (+ the external insight corpus). That selector, not a
  random seed, determines membership.

## External inputs (not vendored)

`gen_insight_sft.py` still reads the insight corpus (`~/arc-train93-split/insight_knowledge.jsonl`,
~2.7 MB) and the insight pipeline (`~/insight-knowledge-arc`). Only the split file above is
vendored here; the corpus and pipeline remain external and must be present to regenerate the
insight-derived SFT targets.
