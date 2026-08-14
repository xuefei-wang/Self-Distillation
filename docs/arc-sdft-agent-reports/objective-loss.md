I have all six questions covered on both sides. Report follows.

---

# SDFT objective comparison: OURS vs THEIRS

**Question restated:** for the two SDFT-on-ARC-AGI-1 codebases, establish the exact distillation loss (direction/coefficient, on-policy vs teacher-forced), teacher construction, importance-sampling correction, loss normalization, temperature handling, and completions-per-prompt — with file:line evidence — then rank the differences that would change the loss curve.

**Paths pinned:**
- **OURS** = `/home/xwang3/Projects/Self-Distillation` (`distil_trainer.py`, `distil_config.py`, `main.py`, `run_arc_armA.sh`). Env: `transformers 4.57.1`, `trl 0.24.0` (`$ .venv/bin/python -c "import transformers,trl;print(...)"` → `transformers 4.57.1 ... trl 0.24.0`).
- **THEIRS** = `/tmp/claude-23749/-home-xwang3-Projects-Self-Distillation/299f8f15-91ed-45bc-ac4e-bbc910d3f288/scratchpad/know-vec/sdft` (abbreviated `THEIRS/` below).

**Note on the brief:** there is **no `objective*.py` in THEIRS**. `$ find THEIRS/src -name '*.py'` returns 29 files; the objective lives in `src/sdft/loss.py` (409 lines). Searched `src/`, `recipes/`, `scripts/`; no file matching `objective*`.

---

## Verified facts

### 1. The distillation loss — direction and coefficient

**Both sides implement the identical alpha-family switch, and THEIRS says so explicitly** (`THEIRS/src/sdft/loss.py:12-13`: `"PORTED FROM THE REFERENCE, NOT FROM THE PAPER. / vendor/self-distillation/distil_trainer.py:_compute_loss, pinned at d775732."`).

**OURS** — `distil_trainer.py:1771-1791`:
```python
        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        if self.alpha == 0: #Forward KL
            kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)
        elif self.alpha == 1: #Reverse KL
            kl_loss = kl_div(teacher_all_logps, all_logps, reduction="none", log_target=True)
        else:
            ...
            kl_loss = alpha * kl_teacher + (1 - alpha) * kl_student
        per_token_loss = kl_loss.sum(-1)
```

**THEIRS** — `THEIRS/src/sdft/loss.py:164-179`:
```python
def _divergence_slice(student_logps, teacher_logps, alpha) -> torch.Tensor:
    if alpha == FORWARD_KL:
        kl = F.kl_div(student_logps, teacher_logps, reduction="none", log_target=True)
    elif alpha == REVERSE_KL:
        kl = F.kl_div(teacher_logps, student_logps, reduction="none", log_target=True)
    else:
        ...
        kl = alpha * F.kl_div(mixture, teacher_logps, ...) + (1 - alpha) * F.kl_div(mixture, student_logps, ...)
    return kl.sum(-1)
```

**Alpha actually used on ARC — both 0.0 (forward KL branch):**
- OURS: `distil_config.py:486-490` `alpha: float = field(default=0.0, ... "If 0.0 (default), the forward KL is used."`; `main.py:221-256` (the `DistilConfig(...)` for ARC) does **not** pass `alpha`. `$ sed -n '221,256p' main.py | grep -oE '^\s+[a-z_]+\s*='` lists 29 keys, none of them `alpha`.
- THEIRS: `THEIRS/recipes/arc1-gold.yaml:68-70`:
  ```yaml
  objective:
    alpha: 0.0            # forward KL, the SDFT reference's direction
    support: full
  ```
  and `THEIRS/src/sdft/loss.py:46-48` `FORWARD_KL = 0.0 / REVERSE_KL = 1.0 / DEFAULT_ALPHA = FORWARD_KL`.

**Both are therefore `F.kl_div(student, teacher)` = `KL(teacher ‖ student)`** — the argument-order caveat is documented on both sides (OURS `distil_trainer.py:1772`; THEIRS `loss.py:143-150`).

**OUR `beta` is NOT part of the loss.** `$ grep -n "per_token_kl" distil_trainer.py` → exactly two hits: `1767` (computed under `if self.beta != 0.0:`) and `1822` (`mean_kl = masked_batch_mean(per_token_kl)`, a logged metric). It is never added to `loss`. `distil_config.py:479-484`: `beta: float = field(default=0.0, ... "If 0.0 (default), the reference model is not loaded"`; ARC does not set it. THEIRS has no beta-equivalent term (searched `loss.py`, `loop.py`; no reference-KL penalty).

**Both are on-policy (student samples its own completions), not teacher-forced.**
- OURS: `distil_config.py:492-499` `generate_from_teacher: bool = field(default=False, ... "If False (default), use the student model for generation"`; not set in `main.py`. `distil_trainer.py:1433`: `generation_prompts = teacher_prompts if self.generate_from_teacher else prompts`.
- THEIRS: `THEIRS/src/sdft/loop.py:117-121`:
  ```python
    # 1. On-policy samples, from the student, with the learned part ON.
    completion_ids, completion_mask = sampler.sample(
        model, student_prompts, n_generations=settings.n_generations, ...)
  ```
  and `THEIRS/recipes/arc1-gold.yaml:71` `tokens: on_policy     # the student samples what it is scored on`. `THEIRS/src/sdft/recipe.py:240` `TOKENS = ("on_policy",)` — the off-policy branch was removed from this loop (`loop.py:159-161`).

### 2. Teacher construction and conditioning

**OURS** — `distil_trainer.py:1733-1743`:
```python
            if self.ref_model is not None:
                teacher_model = self.ref_model
                teacher_ctx = nullcontext()
            elif self.ema_teacher_lora:
                teacher_model = self.model
                teacher_ctx = self._ema_teacher_adapter_context()
            else:
                teacher_model = self.model
                teacher_ctx = self.accelerator.unwrap_model(self.model).disable_adapter()
```
For the ARC LoRA arm, `main.py:201-206` sets `teacher_model = None` unless `not args.peft`, and `run_arc_armA.sh:41` passes `--peft` ⇒ **teacher = the student's own weights with the LoRA adapter disabled**. Conditioning is the privileged prompt: `distil_trainer.py:1707` `teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)`, where `teacher_prompt` is built in `main.py:152-169`:
```python
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")
```
with `output_text` = the ARC gold demonstration (`main.py:143-147` docstring: `"the teacher is conditioned on question + full gold demonstration; the student sees the question alone"`).

**THEIRS** — `THEIRS/recipes/arc1-gold.yaml:64-66`:
```yaml
teacher:
  weights: context      # the same weights with the arm's contribution off
  prompt: privileged    # the task WITH its verified answer
```
Implementation `THEIRS/src/sdft/teacher.py:111-133` (`ContextTeacher`), whose whole body is `with self.trainee.disabled(): logits = completion_logits(...)`, and `THEIRS/src/sdft/trainees/lora.py:68-69`:
```python
    def disabled(self):
        return self.model.disable_adapter()
```
Prompt selection `THEIRS/src/sdft/loop.py:129-130`:
```python
    seen = teacher_prompts if settings.teacher.prompt == "privileged" else student_prompts
    teacher_batch = pack(repeat_prompts(seen, n), completion_ids, completion_mask, pad_id)
```
Privileged field for ARC is the benchmark's own output grid: `arc1-gold.yaml:93` `privileged_key: target`; template `arc1-gold.yaml:124` `template: worked_answer` → `THEIRS/src/sdft/data.py:87-93`:
```python
WORKED_ANSWER_TEMPLATE = """{prompt}

This is an example for a response to the question:
{privileged}

Now answer with a response of your own.
"""
```
i.e. **THEIRS deliberately dropped the clause `", including the thinking process."` that OURS still uses** — documented at `THEIRS/src/sdft/data.py:60-93` and `arc1-gold.yaml:95-124` with measured completion-length growth (`arc1-gold.yaml:107-111`: `1-5  542/35% … 11-15  1803/55%`).

**EMA / frozen alternatives.** THEIRS has `EmaTeacher` (`teacher.py:194-247`) and `FrozenTeacher` (`teacher.py:136-191`), not selected by the ARC recipes. OURS has `ema_teacher_lora` (`distil_config.py:501-511`) — **but `main.py` never forwards it**: `$ grep -ni "ema" main.py` → lines `28,29,31,110,145,251` only; line 28 is the `--ema_teacher` argparse flag, and the `DistilConfig(...)` key list (lines 221-256) contains no `ema_teacher_lora`. So `run_arc_sdft_ema.sh:31`'s `--ema_teacher` is a no-op against `main.py` as it stands on disk, and that arm trains with the plain adapter-disabled teacher.

### 3. Importance sampling / vLLM-vs-HF correction

**OURS: yes, TIS, and it is ON for ARC.**
- Enabled explicitly: `main.py:254` `vllm_importance_sampling_correction = True,` (also the config default, `distil_config.py:631-639`, help text: `"Whether to apply Truncated Importance Sampling (TIS) between vLLM completion logprobs and recomputed logprobs."`).
- Cap: `distil_config.py:641-647` `vllm_importance_sampling_cap: float = field(default=2.0, ... "Truncation parameter C"`; ARC does not override it.
- Ratio construction, `distil_trainer.py:1551-1555`:
  ```python
            if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
  ```
- Applied to the loss, `distil_trainer.py:1793-1797`:
  ```python
        if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights
  ```
  → a **single scalar weight per sequence** (mean of clamped per-token ratios), broadcast over tokens.
- `importance_sampling_level` is **not used in this loss**. `$ grep -n "importance_sampling_level" distil_trainer.py` → `419` (assignment `self.importance_sampling_level = args.importance_sampling_level`) and `1816` (a comment inside `masked_batch_mean`). Its default is `"token"` (`distil_config.py:535-536`), but line 1795 unconditionally averages to sequence level.
- Sampler is vLLM colocate: `main.py:223-227` `use_vllm = True, vllm_mode="colocate", ...`; sampling params `distil_trainer.py:1231-1241` (`"n": 1, ... "temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_completion_length, "logprobs": 0`), and `distil_trainer.py:587-588` `# Important so temperature scaling/logit tweaking affects the TIS log probs` / `logprobs_mode="processed_logprobs",`.

**THEIRS: no IS correction of any kind; sampler is HF `model.generate`.**
- `$ grep -rni 'importance' THEIRS/{src,recipes,scripts,tests,docs}` → 5 hits, all prose disclaimers, no code. Verbatim, `THEIRS/src/sdft/sampler.py:105-109`:
  ```
    3. THE NUMERICS DIFFER. vLLM and HF do not agree token for token even on
       identical weights, so samples are slightly off-policy after a perfect
       sync. The reference implementation carries an importance-sampling
       correction for exactly this; this class does not apply one, and that is a
       known gap rather than a solved problem.
  ```
- Second angle: `$ grep -rniE '\bTIS\b|is_ratio|sampling_correction|importance_sampling' THEIRS/{src,recipes,scripts}` → **no output**.
- The ARC recipe avoids the mismatch by using HF for both sampling and scoring — `THEIRS/recipes/arc1-gold.yaml:74-78`:
  ```yaml
  sampler:
    # HF, not vLLM: the sampler and the scorer are then the SAME engine, which
    # removes the vLLM/HF numeric mismatch the reference corrects with token-level
    # importance sampling and this project does not.
    kind: hf
  ```
  Dispatch at `THEIRS/src/sdft/recipe.py:482-483`: `if settings["kind"] == "hf": return HFSampler(top_p=settings["top_p"])`; generation at `THEIRS/src/sdft/sampler.py:69-74` (`model.generate(..., do_sample=True, temperature=temperature, top_p=self.top_p, num_return_sequences=n_generations, ...)`).

### 4. Loss normalization and truncated completions

**Both use per-sequence token-mean, then mean over sequences. Neither applies DAPO token-count normalization.**

OURS — `distil_trainer.py:1802-1803`:
```python
        loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps
```
THEIRS — `THEIRS/src/sdft/loss.py:289-299` + `loop.py:308`:
```python
def masked_sequence_mean(per_token, mask):
    """Token mean inside each sequence, then mean over sequences.
    NOT a global token mean, and the difference is a real modelling choice: ..."""
    per_sequence = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
    return per_sequence.mean()
```
```python
                (loss / settings.grad_accum).backward()
```

- **DAPO normalization absent in OURS despite `loss_type` defaulting to `"dapo"`** (`distil_config.py:564-565`): `$ grep -n "loss_type" distil_trainer.py` → **one hit**, line `417` `self.loss_type = args.loss_type`. It is never read again.
- **No double grad-accum scaling in OURS**: `distil_trainer.py:474` passes `compute_loss_func="non-None value to disable scaling"` (comment at 469-473), and `transformers/trainer.py:4059-4064` only re-divides when `self.compute_loss_func is None`.
- `current_gradient_accumulation_steps` is set by the HF Trainer: `$ grep -rn "current_gradient_accumulation_steps" .venv/.../transformers/trainer.py` → `2621: self.current_gradient_accumulation_steps = len(batch_samples)`.

**Truncated completions: excluded on neither side, for ARC.**
- OURS: the mechanism exists — `distil_trainer.py:1500-1504`:
  ```python
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
  ```
  but `distil_config.py:583-584` `mask_truncated_completions: bool = field(default=False,` and `main.py` does not set it (not in the 29-key list) ⇒ **off**.
- THEIRS: no such option exists; truncation is only *measured*, `THEIRS/src/sdft/loop.py:152-155`:
  ```python
    # The share of samples the budget cut off rather than the model ending. ...
    stats["truncated_frac"] = float((completion_mask[:, -1] == 1).float().mean())
  ```
  Searched `loss.py`, `loop.py`, `sampler.py`, `data.py` and all recipes for `mask_truncated`/`truncat` — only `truncated_frac` and prose.

**Skip-first differs.** OURS `main.py:255` `num_loss_tokens_to_skip = 3,` → `distil_trainer.py:1697-1703` zeroes the first 3 completion positions in `loss_completion_mask`. THEIRS' equivalent is `skip_first` (`loss.py:302-315`, `loop.py:69` `skip_first: int = 0 # their num_loss_tokens_to_skip`), default `0` (`recipe.py` objective block: `"skip_first": 0,`), and `$ grep -rn "skip_first" THEIRS/recipes/` → **no output**, so ARC runs at 0.

### 5. Temperature handling in the loss

**Identical treatment: both divide logits by the sampling temperature, for student and teacher alike, and both run at T=1.0 on ARC.**

OURS — `distil_trainer.py:891-893`, inside `_get_per_token_logps_and_entropies`, which is called for **both** the student (`1712`) and the teacher (`1745`):
```python
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
```
`self.temperature = args.temperature` (`distil_trainer.py:405`), default `1.0` (`distil_config.py:343-345`), not overridden in `main.py`.

THEIRS — `THEIRS/src/sdft/loss.py:108` (end of `completion_logits`): `return logits[:, :-1, :][:, -n_completion:, :] / temperature`, with the rationale at `loss.py:74-77` (`"The division by the SAMPLING temperature is not cosmetic. The samples came from the tempered distribution ..."`). Called for the student at `loop.py:143-145` and for the teacher at `teacher.py:128-129`. `arc1-gold.yaml:87` `temperature: 1.0`.

### 6. n_generations / completions per prompt, and gradient averaging

**OURS: 1 completion per prompt, 32 prompts per optimizer step.**
- `main.py:245` `num_generations = 1,`
- `run_arc_armA.sh:30-31` `PDBS=${PDBS:-1}` / `NPPB=${NPPB:-32}`, passed as `--per_device_train_batch_size 1 --num_prompts_per_batch 32` (`run_arc_armA.sh:42`), and `main.py:235` `gradient_accumulation_steps = max(1, args.num_prompts_per_batch // args.per_device_train_batch_size)` ⇒ 32.
- Repetition is via `RepeatSampler(..., mini_repeat_count=self.num_generations, ...)` (`distil_trainer.py:732-739`), so with `num_generations=1` each prompt appears once. Colocate vLLM also hard-codes `"n": 1` per device (`distil_trainer.py:1232`).
- Averaging: `.mean()` over the micro-batch (1 sequence) at `1802`, `/32` at `1803`, summed by autograd across the 32 accumulated micro-batches.

**THEIRS: 4 completions per prompt, 1 prompt per optimizer step, no accumulation.**
- `THEIRS/recipes/arc1-gold.yaml:79` `n_generations: 4`
- `arc1-gold.yaml:153-155`:
  ```yaml
  steps: 416
  prompts_per_step: 1
  grad_accum: 1
  ```
  with the rationale at `arc1-gold.yaml:150-152`: `"The gradient of a single prompt is noisy, and n_generations=4 is what averages it -- four on-policy samples of the same prompt, not four different prompts."`
- Averaging: `masked_sequence_mean` takes `.mean()` over the 4 sequences (`loss.py:299`), then `loss / grad_accum` = `/1` (`loop.py:308`).

### Other pinned ARC settings (context for the ranking)

| | OURS | THEIRS |
|---|---|---|
| model | `Qwen/Qwen3-8B` (`run_arc_armA.sh:18`) | `Qwen/Qwen3.5-9B` (`arc1-gold.yaml:62`) |
| lr | `1e-4` (`run_arc_armA.sh:17`) | `1.0e-4` (`arc1-gold.yaml:159`) |
| schedule | `warmup_ratio=0.1`, `cosine` (`main.py:229-230`) | `warmup_steps: 10` (`arc1-gold.yaml:160`), cosine (`loop.py:242-246`) |
| epochs/steps | `EPOCHS=2` (`run_arc_armA.sh:11`) | `steps: 416` = 1 epoch (`arc1-gold.yaml:148-153`) |
| completion budget | `MAXCOMP=4096` (`run_arc_armA.sh:29`) | `max_new_tokens: 3072` (`arc1-gold.yaml:86`) |
| prompt cap | `MAXPROMPT=10240`, left-truncated (`run_arc_armA.sh:27`; `distil_trainer.py:1448,1466-1477`) | `max_prompt_tokens: 12288`, `max_dropped_frac: 0.01` (`arc1-gold.yaml:140-141`) |
| LoRA | `r=16, lora_alpha=32, dropout=0.05`, targets `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` (`main.py:260-263`) | `r: 16`, `lora_alpha=r` ⇒ 16, `dropout=0.0`, targets `all-linear` (`arc1-gold-lora16.yaml:18-27`; `trainees/lora.py:39-59`) |
| thinking mode | patched OFF by default (`nothinking.py:23-24,42`) while the teacher prompt still asks for "the thinking process" (`main.py:159`) | thinking off (`data.py:133` `enable_thinking=False`) and the clause removed (`data.py:87-93`) |
| max_grad_norm | `1` (`main.py:247`) | `1.0` (`arc1-gold.yaml:161`) |

---

## Inferences (labeled — reasoning on top of the facts above)

- **The two objectives are mathematically the same function** (forward-KL, `KL(teacher‖student)`, full vocabulary, per-sequence token-mean then batch-mean, both temperatures divided out). Every difference in the loss curve therefore comes from *what is fed to it*, not from the divergence formula. THEIRS states it was transcribed from OURS' `_compute_loss` (`loss.py:12-13`), and the two code blocks are line-for-line parallel.
- **OURS' loss carries a multiplicative TIS weight that THEIRS' does not.** Since `ratio = exp(old_logps − vllm_logps)` clamped at 2.0 and averaged over the sequence, a well-synced step gives weights near 1 and a poorly-synced one gives weights in (0, 2]. This scales OUR reported loss and its gradient per sequence; THEIRS has no such factor. (Inference from `distil_trainer.py:1793-1797` + the absence grep on THEIRS.)
- **OURS' step-level loss should look far smoother than THEIRS'.** 32 prompts × 1 sample averaged per step vs 1 prompt × 4 samples. Prompt-to-prompt variance on ARC is large (prompt lengths p50 884 → max 8512, `arc1-gold.yaml:131`), so THEIRS' per-step curve is dominated by which single task was drawn.
- **OURS' teacher prompt is the "ratcheting" one THEIRS measured and abandoned.** OURS combines `enable_thinking` forced off (`nothinking.py`) with a teacher instruction to include "the thinking process" (`main.py:159`) — exactly the configuration THEIRS documents at `data.py:68-82` as keeping teacher mass off the end token and growing completions. If that mechanism reproduces on Qwen3-8B, OUR completions should lengthen over training toward the 4096 cap, raising cost and changing the per-token loss composition. **Not verified empirically here** — I did not inspect OUR run logs/W&B.
- **OURS' skip-first=3 removes the highest-divergence positions.** The first completion tokens are where student and teacher differ most (the teacher has just read the answer). Excluding 3 vs 0 should push OUR reported per-token divergence measurably below THEIRS at equal model quality. (Inference from `distil_trainer.py:1697-1703` vs `skip_first: 0`.)
- **OURS' LoRA is scaled 2× and dropout-noisy relative to THEIRS.** `lora_alpha/r = 32/16 = 2` vs `alpha = r ⇒ 1` (THEIRS' comment `trainees/lora.py:51-53`: `"alpha = r keeps the effective scale at 1 whatever the rank"`), and `dropout=0.05` vs `0.0`. At the same nominal `lr=1e-4` the effective adapter step is roughly doubled in OURS, and dropout injects stochasticity into the *student* forward that is absent from the teacher forward (which runs with the adapter disabled) — an extra, non-shared noise source in OUR per-token divergence.
- **`--ema_teacher` in `run_arc_sdft_ema.sh` almost certainly did nothing**, because `main.py` never forwards it to `DistilConfig`. That arm's loss curve should be indistinguishable in construction from the static-teacher arm. (Inference from the two greps; I did not run it to confirm.)

---

## Unknowns / caveats

- **I did not verify runtime behavior on either side** — no training run was executed. All claims are static reads of the files on disk as of this session.
- **Whether OUR ARC completions actually lengthened / truncated**: I did not open `wandb/`, `mlflow.db`, or `results/`. THEIRS' length-ratchet numbers (`arc1-gold.yaml:105-116`) are *their* measurements on Qwen3.5-9B, quoted as such; they are not evidence about OUR Qwen3-8B run.
- **THEIRS' `pyproject.toml`/`uv.lock` versions were not read**, so I cannot pin their transformers/peft/torch versions. `THEIRS/src/sdft/loss.py:19-20` claims `"it pins transformers 4.57 and trl 0.24"` about OURS — which matches what I measured in OUR venv, but I did not confirm THEIRS' own installed versions.
- **Worktree caveat:** the shell resets into `/home/xwang3/Projects/Self-Distillation/.claude/worktrees/cheeky-snacking-duckling`. I read the main checkout paths. `$ diff` shows `distil_trainer.py` and `distil_config.py` **IDENTICAL** between the two; `main.py` differs by exactly one hunk — a 13-line vLLM `ModelRegistry.register_model("Qwen3_5ForCausalLM", ...)` block present only in the main checkout. **No loss-relevant difference.**
- **THEIRS' `topk` support and `coarsen()`** (`loss.py:241-286`) and the `ServedTeacher`/`VLLMSampler` paths exist but are **not** used by `arc1-gold.yaml` (`support: full`, `kind: hf`). I did not audit them for correctness.
- I did not read `THEIRS/DECISIONS.md` (61 KB) or `docs/SDFT.md` beyond the grep hit; they may contain further rationale.

---

## Differences that would change the LOSS CURVE (ranked)

1. **TIS weight, present only in OURS.** OURS multiplies each sequence's loss by `mean(clamp(exp(old−vllm_logps), max=2.0))` (`distil_trainer.py:1793-1797`); THEIRS has no IS term at all (`sampler.py:105-109`, absence grep). Compounding this: OURS *generates with vLLM* and *scores with HF* (two engines, genuinely off-policy samples), while THEIRS uses HF for both (`arc1-gold.yaml:74-78`), so OUR curve carries both the mismatch and its partial correction, and THEIRS carries neither. Largest structural difference in the loss expression itself.

2. **Batch composition: 32 prompts × 1 sample (OURS) vs 1 prompt × 4 samples (THEIRS).** `main.py:245` + `run_arc_armA.sh:30-31` vs `arc1-gold.yaml:79,153-155`. Changes the *variance* of the plotted per-step loss by roughly an order of magnitude, and changes what the gradient is an average over (across-task vs within-task). Two curves drawn from these will not be visually comparable even if the method is identical.

3. **Teacher prompt clause + forced non-thinking.** OURS: `"...including the thinking process."` (`main.py:159`) with `enable_thinking` patched to default-off (`nothinking.py:23-24`). THEIRS: clause removed (`data.py:87-93`, selected by `arc1-gold.yaml:124 template: worked_answer`) precisely because it drove completions 542→1803 tokens in 15 steps and truncation 35%→55% (`arc1-gold.yaml:105-116`). This changes completion length, truncation fraction, and therefore the token population the per-token divergence averages over — a slow drift rather than a level shift.

4. **`num_loss_tokens_to_skip = 3` (OURS) vs `skip_first = 0` (THEIRS).** `main.py:255` / `distil_trainer.py:1697-1703` vs `recipe.py` objective default. Drops the three highest-divergence positions from every sequence ⇒ a systematic downward level shift in OUR reported loss.

5. **LoRA effective scale and dropout.** `lora_alpha=32, r=16, dropout=0.05` (`main.py:260-263`) vs `lora_alpha=r=16, dropout=0.0` (`trainees/lora.py:44-59`, `arc1-gold-lora16.yaml:24`). At identical `lr=1e-4` OURS takes ~2× the effective adapter step (faster initial descent, higher instability risk), and dropout adds per-step noise to the student forward only.

6. **Adapter coverage.** OURS targets 7 named projections (`main.py:262`); THEIRS targets `all-linear` (`trainees/lora.py:30, 26-27` rationale, `arc1-gold-lora16.yaml:26-29`). On OUR Qwen3-8B all-attention stack the 7-name list is near-complete; on THEIR Qwen3.5-9B hybrid stack it would have reached 8 of 32 layers, which is why they chose `all-linear`. Different fractions of the network are trainable ⇒ different achievable loss floors.

7. **Epochs / total steps.** OURS 2 epochs (`run_arc_armA.sh:11`) with warmup_ratio 0.1 and cosine over the whole run; THEIRS exactly 1 epoch = 416 steps with a 10-step warmup (`arc1-gold.yaml:153,160`). The cosine LR shapes differ, so late-run curvature differs even with identical everything else.

8. **Budgets: completion 4096 vs 3072, prompt 10240 (left-truncated, silent) vs 12288 (0 rows dropped, checked by `max_dropped_frac: 0.01`).** `run_arc_armA.sh:27-29` vs `arc1-gold.yaml:86,140-141`. OUR left-truncation at 10240 can silently remove part of the gold demonstration from the teacher context on long tasks — a *weaker teacher* on exactly the hardest rows, which raises the loss floor there. THEIRS explicitly sized 12288 to drop zero rows (`arc1-gold.yaml:126-140`).

9. **Model and vocabulary: Qwen3-8B vs Qwen3.5-9B (248,320-token vocab, `loss.py:24`).** Full-vocabulary KL magnitudes are not comparable across different vocabularies and different base models; absolute loss values from the two runs should never be plotted on one axis.

10. **(Bookkeeping, not a curve difference)** `loss_type` defaults to `"dapo"` in OURS (`distil_config.py:564-565`) but is never used (`$ grep -n "loss_type" distil_trainer.py` → line 417 only), so neither side applies DAPO token-count normalization. And OURS' `beta` KL-to-reference term is computed-but-unused (`distil_trainer.py:1767,1822`); setting `beta≠0` would change logging and add a reference forward pass, but would **not** change the loss.

---

## Methods (commands run)

```
ls -la /home/xwang3/Projects/Self-Distillation/
ls -laR <THEIRS>                                  # full tree
find <THEIRS>/src <THEIRS>/recipes <THEIRS>/scripts -type f | sort
find <THEIRS>/src -name '*.py' -exec wc -l {} +   # 29 files, 6511 lines; no objective*.py
Read <THEIRS>/src/sdft/loss.py                    # full, 409 lines
Read <THEIRS>/src/sdft/loop.py                    # full, 462 lines
Read <THEIRS>/src/sdft/teacher.py                 # full, 333 lines
Read <THEIRS>/src/sdft/sampler.py                 # full, 205 lines
Read <THEIRS>/src/sdft/data.py                    # full, 247 lines
Read <THEIRS>/src/sdft/trainees/lora.py           # full, 126 lines
grep -n '' <THEIRS>/recipes/{_base,arc1-gold,arc1-gold-lora16}.yaml
grep -rni 'importance' <THEIRS>/{src,recipes,scripts,tests,docs}      # 5 prose hits, no code
grep -rniE '\bTIS\b|is_ratio|sampling_correction|importance_sampling' <THEIRS>/{src,recipes,scripts}   # no output
grep -n 'alpha|tokens|on_policy|sampler|...' <THEIRS>/src/sdft/recipe.py
sed -n '50,90p;375,424p;470,504p' <THEIRS>/src/sdft/recipe.py
grep -rn "skip_first|top_p" <THEIRS>/recipes/                          # no output for skip_first
grep -n "build_sampler|train(|gradient_checkpointing|..." <THEIRS>/scripts/train_sdft.py

cd /home/xwang3/Projects/Self-Distillation
grep -n "def |_compute_loss|kl_div|jsd|beta|alpha" distil_trainer.py
Read distil_trainer.py 1682-1831, 833-922, 1490-1569, 1228-1297, 1425-1489, 705-749, 462-491
grep -n "per_token_kl" distil_trainer.py                               # 1767, 1822 only
grep -n "loss_type" distil_trainer.py                                  # 417 only
grep -n "importance_sampling_level" distil_trainer.py                  # 419, 1816 only
grep -n "model_accepts_loss_kwargs|compute_loss_func" distil_trainer.py
grep -n "current_gradient_accumulation_steps" distil_trainer.py        # 1803 only
grep -n "importance_sampling|truncated|mask_truncated_completions" distil_trainer.py
Read distil_config.py 479-548, 530-649; sed -n '306,312p;343,348p'
Read main.py (full), run_arc_armA.sh (full), run_arc_sdft_ema.sh, nothinking.py
grep -ni "ema" main.py                                                 # argparse only, not in DistilConfig(...)
sed -n '221,256p' main.py | grep -oE '^\s+[a-z_]+\s*='                 # 29 config keys
.venv/bin/python -c "import transformers,trl; print(...)"              # 4.57.1 / 0.24.0
grep -rn "current_gradient_accumulation_steps" .venv/.../transformers/trainer.py   # 2621, 4064
sed -n '4040,4075p' .venv/.../transformers/trainer.py
diff <worktree>/main.py /home/xwang3/Projects/Self-Distillation/main.py            # 1 hunk, vLLM arch registration
diff -q <worktree>/{distil_trainer,distil_config}.py <main>/...                    # IDENTICAL
```