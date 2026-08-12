# Figure 3 Reproduction — Results

Reproduction of Figure 3 ("long-horizon sequential continual learning") from
*Self-Distillation Enables Continual Learning* (arXiv 2601.19897), adapted to available
resources. One model is trained **sequentially** on two Skill-Learning tasks (Science Q&A →
Tool Use) under **SDFT** and **SFT**, evaluated on both tasks after each stage, and normalized
per task (0 = base accuracy, 1 = best accuracy across both methods & stages).

![Figure 3 reproduction](figure3_reproduction.png)

## Headline result

The central claim reproduces: **SDFT accumulates skills without forgetting; SFT forgets the
earlier skill.** On the Science axis, after learning Tool Use:
- **SDFT retains Science** (normalized 1.0 → 0.94; raw 0.578 → 0.568).
- **SFT catastrophically forgets Science** (normalized 0.49 → **−0.35**; raw 0.493 → 0.355,
  i.e. *below* the base model's 0.412).

Both methods learn the new task (Tool Use) at stage 2. SFT reaches a higher absolute Tool Use
number (0.660 vs SDFT 0.619); SDFT trades a little new-task gain for near-perfect retention of
the earlier skill — exactly the Figure 3 trade-off.

## Raw accuracies (accuracy.json)

| checkpoint | science | tooluse |
|---|---|---|
| base (Qwen3-8B) | 0.4122 | 0.5979 |
| SDFT after Science | 0.5779 | 0.5876 |
| SDFT after Science+ToolUse | 0.5680 | 0.6186 |
| SFT after Science | 0.4931 | 0.5979 |
| SFT after Science+ToolUse | 0.3550 | 0.6598 |

Eval sets: science 507 examples (exact-match on `<answer>`), tooluse 97 examples (tool-call
regex match). 1 seed.

## Deviations from the paper (all deliberate, for resources/hardware)

- **2 tasks, not 3.** The paper's Figure 3 uses Science → Tool Use → **Medical**; the Medical
  (HuatuoGPT-o1) dataset is not shipped in this repo, so this is a faithful 2-task reduction.
- **Qwen3-8B**, not Qwen2.5-7B (paper) or Qwen3.5-9B (originally requested). Qwen3.5-9B needs
  `transformers>=5` / `vllm>=0.17` and is a hybrid-linear-attention MoE the SDFT trainer wasn't
  built for; Qwen3-8B is a standard dense model the pinned stack supports.
- **LoRA** (r=16, α=32) instead of full fine-tuning — fits an 8B model + vLLM colocate on one
  48 GB A6000 (paper used a 140 GB H200).
- **Thinking mode OFF.** Qwen3 defaults to emitting `<think>` blocks; the paper's Qwen2.5-Instruct
  has no thinking mode. We force non-thinking (probe: science exact-match 5/8 off vs 1/8 on) so
  Qwen3 behaves like the paper's instruct model and matches the `<reasoning>/<answer>` format.
- **No EMA teacher under LoRA.** SDFT's teacher is the demonstration-conditioned *base* model
  (student's own base with the LoRA adapter disabled); the slow EMA teacher (α=0.01) is disabled
  because its parameter-sync is incompatible with a LoRA student. The on-policy mechanism that
  drives the anti-forgetting result is preserved.
- **LR 1e-4** (LoRA needs a higher LR than the paper's full-FT 1e-5). SDFT 2 epochs, SFT 1 epoch,
  effective batch 32, cosine + 10 warmup steps, bf16, seed 42.
- 1 seed → no error bars; small eval sets → some noise in absolute numbers. The reproduction
  targets the qualitative trend, which holds clearly.

## How to reproduce

```bash
# 0. env (pinned) + non-thinking base eval dir
uv venv --python 3.12 && uv pip install -r requirements.txt
uv pip uninstall deepspeed              # accelerate imports it; it needs a CUDA toolkit this box lacks
uv run python nothinking.py --out ckpt/base

# 1. train both arms sequentially (SDFT on 4 GPUs via DDP, SFT single-GPU) + merge between stages
LR=1e-4 bash run_fig3.sh                 # or drive stages manually (see the workspace ledger)
#    every vLLM run needs: VLLM_USE_FLASHINFER_SAMPLER=0   (venv-only CUDA)

# 2. evaluate the 5 checkpoints on both tasks, collect, plot
bash eval_grid.sh                        # avoid GPUs used by other jobs
uv run python collect_accuracy.py
uv run python plot_fig3.py               # -> results/fig3/figure3_reproduction.png
```

Design spec: `docs/superpowers/specs/2026-08-11-figure3-reproduction-design.md`.
Implementation plan: `docs/superpowers/plans/2026-08-11-figure3-reproduction.md`.
