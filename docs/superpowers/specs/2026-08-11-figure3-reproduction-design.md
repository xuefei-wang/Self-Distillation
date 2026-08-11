# Figure 3 Reproduction — Sequential Continual Learning (Qwen3-8B, LoRA)

Date: 2026-08-11
Paper: "Self-Distillation Enables Continual Learning" (arXiv 2601.19897)

## Goal

Reproduce the *story* of **Figure 3** — long-horizon sequential continual learning
comparing **SDFT vs SFT** — using the resources and data available in this repo.

Figure 3 (paper): one model is trained sequentially on three Skill-Learning tasks
(Science Q&A → Tool Use → Medical); after each stage it is evaluated on *all* tasks.
Curves are linearly normalized per task (0 = base-model accuracy, 1 = max accuracy
across both methods). SDFT accumulates skills; SFT oscillates/forgets.

## Scope decisions (agreed with user)

- **2-task reduction:** Science Q&A → Tool Use only. The Medical dataset is not shipped
  in the repo (README: "coming soon"), so a faithful 3-task run is impossible today.
  Two tasks are sufficient to exhibit the forget-vs-accumulate contrast.
- **Model:** **Qwen/Qwen3-8B** (standard dense, supported by the repo's pinned
  `transformers==4.57.1` + `vllm==0.12.0` + `trl==0.24.0` — no library upgrade needed).
  Chosen over Qwen3.5-9B, which needs `transformers>=5.x` / `vllm>=0.17`, is a hybrid
  linear-attention MoE the SDFT trainer was not built for, and would require porting the
  custom `DistilTrainer`.
- **Fine-tuning:** **LoRA** (fits an 8B model + vLLM colocate on one 48 GB A6000).
- **Seeds:** 1 seed (user directive).

## Non-goals

- No 3-task / Medical run.
- No full hyperparameter sweep (paper Table 3). A minimal LR check on stage 1 only.
- No edits to `distil_trainer.py` / `distil_config.py` / the two eval scripts.

## Environment

- `uv venv` + `uv pip install -r requirements.txt` (pinned versions unchanged).
- Hardware: 8× NVIDIA RTX A6000 (48 GB each). System CUDA present.
- vLLM colocate mode as in `main.py` (`vllm_gpu_memory_utilization=0.3`,
  `vllm_enable_sleep_mode=True`).
- **Integration risk to validate first:** Qwen3-8B "thinking" mode may emit
  `<think>…</think>` and break the science `<answer>` extraction / tool-call regex.
  Validate base-model parseability up front; if broken, pin `enable_thinking=False`
  in the eval chat template (and, if needed, the training prompt formatting).

## Architecture

### Training arms (each trained sequentially, LoRA merged between stages)

Both arms start from the same base Qwen3-8B and train Science first, then Tool Use on
the *merged* result of stage 1 — a genuine "single model trained sequentially."

- **SDFT arm** (existing `DistilTrainer`, `beta=0.0` forward-KL, on-policy self-distill):
  1. base → LoRA train Science → **merge adapter into base** → `ckpt/sdft_science`
  2. `ckpt/sdft_science` → LoRA train Tool Use → merge → `ckpt/sdft_science_tooluse`
- **SFT arm** (new `sft_main.py`, standard off-policy supervised FT on demonstrations):
  1. base → LoRA train Science → merge → `ckpt/sft_science`
  2. `ckpt/sft_science` → LoRA train Tool Use → merge → `ckpt/sft_science_tooluse`

Merging (via `peft` `merge_and_unload` + `save_pretrained` + tokenizer) produces full
model directories the vLLM eval scripts load directly with `--model_path` (avoids
editing the eval scripts to support LoRA adapters).

### SFT baseline (`sft_main.py`)

- TRL `SFTTrainer` on demonstrations formatted as chat: `prompt` (user) → golden
  completion (assistant). Golden text:
  - Science: `output_text` (train_data field used by `main.py`'s science loader).
  - Tool Use: `'\n'.join(golden_response)` (matches `main.py`'s tooluse loader).
- Same LoRA config, bf16, cosine + 10 warmup steps, weight decay 0, max grad norm 1.
- **1 epoch** (paper: SFT overfits past 1 epoch on Skill Learning).

### SDFT training (reuse `main.py` path)

- Reuse `main.py`'s dataset loaders and `DistilConfig`; add `--model_name Qwen/Qwen3-8B`,
  a `peft_config` (LoRA), and `--resume-from`/`--model_name` pointing at the previous
  merged checkpoint for stage 2. May extend `main.py` with a `--peft` flag + a
  `--init_model_path` (defaults to `--model_name`) rather than a new trainer.
- **2 epochs** (paper: SDFT benefits from ~2 epochs on Skill Learning).

### Evaluation grid

Run both eval scripts on **5 checkpoints**, each (checkpoint, task) pair writing to a
distinct `--output_dir` (the scripts write a fixed `eval_results.json`):

| checkpoint                     | eval Science | eval ToolUse |
|--------------------------------|:-----------:|:-----------:|
| base Qwen3-8B                  | ✔ | ✔ |
| sdft_science                   | ✔ | ✔ |
| sdft_science_tooluse           | ✔ | ✔ |
| sft_science                    | ✔ | ✔ |
| sft_science_tooluse            | ✔ | ✔ |

→ 10 accuracy numbers collected into `results/fig3/accuracy.json`.

### Plot (`plot_fig3.py`)

- Read the 10 accuracies. Per task `t`, normalize:
  `norm = (acc - base_t) / (max_over_all_arms_and_stages_t - base_t)` (clip to [0, ~1]).
- Two panels (Science, Tool Use). X-axis = training stage (0=base, 1=after Science,
  2=after Tool Use). One line per method (SDFT, SFT) per panel.
- Expectation to sanity-check: SDFT Science stays high through stage 2; SFT Science
  drops at stage 2 (forgetting); both rise on Tool Use at stage 2.
- Save `results/fig3/figure3_reproduction.png`.

## Hyperparameters (adapted from Table 3 for LoRA)

- LoRA: `r=16`, `alpha=32`, dropout 0.05, target attention+MLP proj modules.
- LR: **1e-4** (LoRA needs a higher LR than the paper's full-FT 1e-5). Optional stage-1
  LR check over {5e-5, 1e-4}, pick best Science accuracy for the full run.
- Batch: 32 prompts/batch (grad-accum), per-device batch 1.
- Schedule: cosine, 10 warmup steps, bf16, weight decay 0, max grad norm 1.
- SDFT: 2 epochs, EMA α 0.01, forward-KL (`beta=0.0`), 1 on-policy rollout, max gen 2048.
- SFT: 1 epoch.
- Seed: 42.

## Parallelism

8× A6000 → run independent training stages/arms on separate GPUs concurrently via
subagents (`CUDA_VISIBLE_DEVICES` per run). Stage 2 depends on stage 1 within an arm,
but the SDFT and SFT arms (and LR variants) are independent → dispatch in parallel.
Wall-clock ≈ one sequential arm rather than four runs end-to-end.

## New / changed files

- **New:** `sft_main.py` (SFT baseline), `merge_lora.py` (adapter→full-model merge helper),
  `run_fig3.sh` (orchestrator), `plot_fig3.py`, `results/fig3/` (outputs).
- **Changed (minimal):** `main.py` — add `--peft` (LoRA) and `--init_model_path` so
  stage 2 can start from a merged checkpoint.
- **Unchanged:** `distil_trainer.py`, `distil_config.py`, `eval_science.py`,
  `eval_tooluse.py`.

## Success criteria

1. Environment installs; base Qwen3-8B loads and produces parseable eval outputs.
2. All 4 training runs complete and merge without error.
3. All 10 evals produce an `eval_results.json` with a numeric accuracy.
4. `figure3_reproduction.png` renders both panels.
5. Qualitative match to Figure 3: SDFT retains Science after learning Tool Use; SFT's
   Science accuracy drops at stage 2 (visible forgetting). If the trend does not appear,
   report the actual numbers plainly rather than forcing the narrative.

## Risks

- Qwen3 thinking-mode output format vs eval parsers (mitigation above).
- LoRA LR/rank may under-fit vs full FT; mitigated by the small LR check and by the fact
  the figure is about a *trend*, not absolute SOTA numbers.
- vLLM colocate + LoRA merge memory on 48 GB; mitigated by low `gpu_memory_utilization`
  and merging on CPU/offload if needed.
- Science/Tool Use eval sets may be small → noisy accuracy; 1 seed means no error bars
  (accepted by user).
