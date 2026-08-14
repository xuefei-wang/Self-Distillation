# ARC-AGI-1 SDFT: our setup vs `recursive-knowledge/know-vec@arc-sdft`

**Date:** 2026-08-14 · **Ours:** `Self-Distillation` (Qwen3-8B, TRL-GRPO-derived `DistilTrainer`) ·
**Theirs:** `know-vec/sdft` (Qwen3.5-9B, from-scratch SDFT loop) · branch `arc-sdft`.

Comparison prompted by "our loss curve looks very different from theirs." Findings below were
produced by three grounded subagents (objective/loss, data/prompt, LoRA/optim) and the
load-bearing numbers were independently re-verified (teacher-prompt token measurement,
config lines, insight coverage). Every claim is cited `file:line`.

---

## TL;DR

- **The objective is the same function.** Both are forward KL `KL(teacher‖student)`, full-vocab,
  on-policy, per-sequence token-mean then batch-mean, logits divided by sampling temperature.
  Theirs was literally ported from our `_compute_loss` (`know-vec/sdft/src/sdft/loss.py:12-13`).
  So curve differences come from **what is fed to the loss**, not the loss math.
- **Our loss *values* are not comparable to theirs** even in principle: we skip the first 3
  loss tokens (`main.py:num_loss_tokens_to_skip=3`) and multiply each sequence by a TIS weight;
  they do neither. Different base model + vocab (151k vs 248k) on top.
- **Two truncation problems, both ours, both measured** (details below). Root cause of both:
  we condition the teacher on a ~6.9k-token gold CoT demonstration.
- **Fix implemented** (this branch): selectable compact teacher context + drop the
  thinking-process clause + truncation preflight + `mask_truncated_completions`.

---

## The two truncation problems (measured on our 381-row ARC train set)

### A. Output-side length ratchet
The teacher prompt ends `", including the thinking process."` (`main.py:159`) while `nothinking.py`
forces `enable_thinking=False`, so render emits a closed-empty `<think></think>`. The instruction
can't be met, the teacher holds mass off EOS, and truncated samples carry no EOS anywhere → no
position teaches stopping. It ratchets.

Our training-side evidence (`mlflow.db`, run `sdft_armA`, `completions/mean_length`):

| step | 0 | 5 | 11 | 13 | 17 |
|---|---|---|---|---|---|
| mean completion len | 300 | 375 | 900 | 1254 | 929 |

~3× growth in ~15 steps — the same shape `know-vec` measured (542→1803, 35%→55% truncated;
`arc1-gold.yaml:105-116`) and fixed by dropping the same one clause. Our `clipped_ratio` stayed
0–6% only because our completion budget is 4096; the pathology surfaces at **eval** instead.

### B. Input-side teacher-prompt truncation (silent, ~42%)
Our teacher is conditioned on the full gold CoT demonstration (demo p50 6,897 tok), making teacher
prompts p50 **8,627** / max **29,686 tok**. At `MAXPROMPT=10240` with `truncation_side="left"`
(`distil_trainer.py:1448`), **161/381 = 42.3%** of teacher prompts are silently left-truncated —
and because order is *question → example → demo → "now answer"*, left-truncation drops the
**question/task grids**, leaving the teacher scoring the student against a headless CoT for a
problem it can no longer see. Uncounted, unlogged, untested.

`know-vec` avoids this by conditioning on the bare answer grid (`privileged_key: target`,
p50 312 chars) and **dropping** over-cap rows with a 1% fail-loud guard (`data.py:168-198`).

### Eval-side confirmation (our `results/arc_armA/eval/`)
Dominant failure is `Parse failed` (no parseable grid), not wrong answers; and it is prompt-cap
sensitive — re-evaluating the SFT checkpoint at a 12288 cap recovered 53 tasks:

| run | prompt cap | Correct | Parse-failed | Accuracy |
|---|---|---|---|---|
| base | 10240 | 7 | 258 (64.5%) | 1.75% |
| SFT armA | 10240 | 4 | 253 (63.0%) | 1.00% |
| SFT armA (re-eval) | 12288 | 5 | **200 (50.0%)** | 1.25% |

---

## Full config comparison

### LoRA / optimizer / schedule
| | OURS | THEIRS |
|---|---|---|
| base model | `Qwen/Qwen3-8B` (`run_arc_armA.sh:18`) | `Qwen/Qwen3.5-9B` (`arc1-gold.yaml:62`) |
| LoRA r | 16 | 16 |
| lora_alpha / scaling | **32 → 2.0** (`main.py:26`) | **16 (=r) → 1.0** (`lora.py:54`) |
| lora_dropout | **0.05** (`main.py:27`) | **0.0** (`lora.py:40`) |
| target_modules | 7 named projections (`main.py:254`) | `all-linear` (`lora.py:30`) |
| optimizer | AdamW-fused, β(0.9,0.999), wd 0, clip 1.0 | AdamW, β(0.9,0.999), wd 0, clip 1.0 |
| lr / scheduler | 1e-4 / cosine (`main.py:215,217`) | 1e-4 / cosine (`arc1-gold.yaml:159`, `loop.py:242`) |
| warmup | ratio 0.1 → ~3/22 steps (~14%) | 10/416 steps (~2.4%) |
| effective batch | **32 prompts × 1 gen / step** | **1 prompt × 4 gens / step** |
| optimizer steps | ~22 (2 epochs, 381 rows, 29 dropped/epoch by the //32 floor) | 416 (1 epoch, 416 rows) |
| grad checkpointing | on | on |
| adapter merge | merged to full ckpt (`merge_lora.py`) | adapter loaded live, no merge |
| seed | 42 | 0 |

### Objective / sampling
| | OURS | THEIRS |
|---|---|---|
| divergence | forward KL `KL(teacher‖student)`, α=0 (`distil_trainer.py:1771-1791`) | same, α=0 (`loss.py:164-179`, `arc1-gold.yaml:69`) |
| tokens | on-policy student samples | on-policy student samples |
| teacher | student weights, adapter disabled (`main.py:189-193`) | same, `disabled()` (`teacher.py:111-133`) |
| sampler | **vLLM colocate**, generate; score with HF | **HF for both** sampler and scorer (`arc1-gold.yaml:74-78`) |
| IS correction | **TIS on**, per-seq weight, cap 2.0 (`distil_trainer.py:1793-1797`) | **none** (documented gap, `sampler.py:105-109`) |
| loss-token skip | **first 3 skipped** (`main.py:num_loss_tokens_to_skip=3`) | 0 (`recipe.py` `skip_first:0`) |
| mask truncated | was **off** (default) → now **on** (this branch) | not implemented (measured only) |
| n generations | 1 (`main.py:232`) | 4 (`arc1-gold.yaml:79`) |
| temperature | 1.0 | 1.0 |

### Data / prompt / length
| | OURS | THEIRS |
|---|---|---|
| teacher conditioned on | full gold CoT demo, p50 6,897 tok (`prep_arc.py:57`) | bare answer grid `target`, p50 312 chars (`arc1-gold.yaml:93`) |
| thinking-process clause | **present** (`main.py:159`) → now **removed** | removed (`data.py:87-93`, `template: worked_answer`) |
| thinking mode | off (`nothinking.py`) | off (`data.py:133`) |
| prompt cap / over-length | 10240, silent **left-truncate** (42.3% teacher) | 12288, **drop** + 1% guard (0 rows) |
| train gen budget | 4096 | 3072 |
| eval budget / protocol | 12288 greedy, 400 tasks, truncation not recorded | 3072 greedy, 100/418, `format_valid`+`truncated_frac` recorded |

---

## Fix implemented (this branch)

`main.py` — ARC path:
1. **Selectable teacher context** `--teacher_knowledge {demonstration|target|insight}`
   (default `demonstration`, preserves Arm A). `target` = bare oracle grid;
   `insight` = concise task-level insight with `target` fallback (insight covers 178/381 = 47%).
2. **Dropped the thinking-process clause** from the ARC teacher template (matches `know-vec`).
3. **`_preflight_teacher_lengths`** — prints the over-cap teacher-prompt fraction and raises
   above 0.5, so silent left-truncation becomes loud.
4. **`mask_truncated_completions=True`** — clipped completions excluded from the loss.

Not changed: the other three datasets (tooluse/science/medical) still carry the same clause on
line 61/87/122 — fix there too if those arms are revisited.

---

## Verification

- **Teacher-prompt truncation** (offline, real tokenizer, 381 rows, cap 10240):
  `demonstration` 42.3% over → `target` **1.6%** over (residual 6 are giant-grid tasks whose
  *student* prompt is also near the cap; not teacher-fixable).
- **Completion-length flattening**: short SDFT smoke, GPU 7, `--teacher_knowledge target`,
  `--max_steps 10`, budget 4096. `completions/mean_length` per step:
  343, 386, 68, 166, 368, 404, 276, 306, 304, 162 — a flat ~68–404 band, `clipped_ratio=0.0`
  every step, single-completion max 1214, `importance_sampling_ratio` mean ~0.996. No ratchet,
  vs the demonstration arm's monotonic 300→930. The live preflight logged `6/381 (1.6%)` over the
  10240 cap, confirming problem B is fixed in-run.

### Quick real arms — 20-step SDFT, 400 held-out eval

| arm | teacher context | accuracy | correct | parse-failed |
|---|---|---|---|---|
| base (untrained) | — | 1.75% | 7/400 | 258/400 |
| demonstration-SFT (peer Arm A) | full gold CoT | 1.00% | 4/400 | 253/400 |
| **target-SDFT** | bare oracle grid | **3.25%** | 13/400 | **3**/400 |
| **insight-SDFT** | concise insight (229/381), target fallback | **3.25%** | 13/400 | **18**/400 |

20 gradient steps (NPPB 8), caps 5120/2048, LR 1e-4, Qwen3-8B, greedy eval @12288.
The **parse-failure wall collapses 255 → 3–18** — the direct payoff of fixing both
truncations (the model now emits short, terminating, well-formed grids) — and accuracy
comes off the ~1% floor to ~2× base / 3× demonstration-SFT. Directional only (the
baseline arm trains 2 full epochs); target and insight tie on accuracy, target formats
slightly tighter. Base's real capability was partly hidden behind format failures (13
right surfaced vs base's 7), not created by 20 steps of training.

---

## Caveats worth carrying

- **`--ema_teacher` was a no-op — fixed in this branch.** `run_arc_sdft_ema.sh` passes
  `--ema_teacher`, but `main.py` did not forward `ema_teacher_lora` to `DistilConfig`, so that
  arm trained the plain adapter-disabled teacher, identical to the static arm. Now wired, so the
  flag activates the LoRA EMA-teacher adapter (`add_ema_teacher_adapter` + `EMATeacherAdapterCallback`,
  `distil_trainer.py:380,638`), which tracks the trainable adapter via `ref_model_mixup_alpha`.
- **Absolute loss values are not comparable across the two repos** (skip-first 3 vs 0, TIS
  weight vs none, 151k vs 248k vocab, different base). Compare *shape* and *eval accuracy*, never
  the raw loss magnitude.
- **`loss_type` defaults to `"dapo"` but is never read** in our trainer (`distil_trainer.py:417`
  only); neither side applies DAPO token-count normalization.
