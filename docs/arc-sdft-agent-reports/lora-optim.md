## Answer

The two setups differ far more in **schedule** than in optimizer: ours is **32 prompts/optimizer step × 22 steps (2 epochs over 381 rows), LoRA α/r = 32/16 = 2.0, dropout 0.05, 7 named projections, seed 42**; theirs is **1 prompt/step × 4 generations × 416 steps (1 epoch over 416 rows), LoRA α/r = 16/16 = 1.0, dropout 0.0, `all-linear`, seed 0**. Both use bf16, AdamW, LR 1e-4, cosine, max_grad_norm 1.0, gradient checkpointing on.

---

## Verified facts

### 1. LoRA config

**OURS (SDFT arm)** — `main.py:257-264`:
```
257	    peft_config = None
258	    if args.peft:
259	        from peft import LoraConfig
260	        peft_config = LoraConfig(
261	            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
262	            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
263	            task_type="CAUSAL_LM",
264	        )
```
Argparse defaults (`main.py:25-27`), **EXPLICIT in our code, not framework defaults**:
```
25	    parser.add_argument("--lora_r", type=int, default=16)
26	    parser.add_argument("--lora_alpha", type=int, default=32)
27	    parser.add_argument("--lora_dropout", type=float, default=0.05)
```
`run_arc_armA.sh:41` passes `--peft --lora_r 16` only; `$ grep -n "lora_alpha\|lora_dropout" run_arc_armA.sh` → `NO MATCH` ⇒ **alpha=32, dropout=0.05 in force**. `bias` is not passed ⇒ peft default `"none"`. Confirmed empirically by the SFT arm's written adapter, `ckpt/arc_sft_armA_adapter/adapter_config.json`: `"lora_alpha": 32`, `"lora_dropout": 0.05`, `"r": 16`, `"bias": "none"`, `"use_rslora": false`, `"task_type": "CAUSAL_LM"`, `"target_modules": ["v_proj","up_proj","o_proj","q_proj","k_proj","gate_proj","down_proj"]`. (Identical LoraConfig in `sft_main.py:105-109`.) `ckpt/arc_sdft_armA_adapter/` is **empty** (`$ ls -la` → only `.`/`..`), so no SDFT adapter_config exists on disk to cross-check.

**Effective scaling (ours) = lora_alpha / r = 32/16 = 2.0** — peft 0.17.1 `.venv/lib/python3.12/site-packages/peft/tuners/lora/layer.py:224`: `self.scaling[adapter_name] = lora_alpha / r` (line 222 is the `use_rslora` branch; `use_rslora` default `False`, `peft/tuners/lora/config.py:355-356`).

**THEIRS** — `src/sdft/trainees/lora.py:30`, `:39-59`:
```
30	DEFAULT_TARGETS = "all-linear"
39	    def __init__(self, model, *, r: int = 16, alpha: int | None = None,
40	                 dropout: float = 0.0, target_modules=DEFAULT_TARGETS,
...
49	            config = LoraConfig(
50	                r=r,
51	                # alpha = r keeps the effective scale at 1 whatever the rank, so
52	                # a rank sweep changes capacity and not the learning rate in
53	                # disguise.
54	                lora_alpha=r if alpha is None else alpha,
55	                lora_dropout=dropout,
56	                target_modules=target_modules,
57	                bias="none",
58	                task_type="CAUSAL_LM",
59	            )
```
`recipes/arc1-gold-lora16.yaml:18-29` sets only `kind: lora` / `r: 16`, and explicitly documents no `target_modules`:
```
24	  r: 16
25	  # NO `target_modules`, which takes LoraTrainee's default of `all-linear`.
```
⇒ **alpha = r = 16, dropout = 0.0, target_modules="all-linear", bias="none", task_type="CAUSAL_LM". Effective scaling = 16/16 = 1.0.** Their comment claims 43,278,336 trainable params (`arc1-gold-lora16.yaml:12`) — their number, not measured by me.

### 2. Base model / precision / gradient checkpointing

| | OURS | THEIRS |
|---|---|---|
| model id | `Qwen/Qwen3-8B` — `run_arc_armA.sh:18` `BASE=${BASE:-Qwen/Qwen3-8B}` | `Qwen/Qwen3.5-9B` — `recipes/arc1-gold.yaml:62` `model: Qwen/Qwen3.5-9B` |
| load dtype | `main.py:193-196` `AutoModelForCausalLM.from_pretrained(init_path, torch_dtype=torch.bfloat16,)` | `qwen35/src/qwen35/load.py:78` `model = Qwen3_5ForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)` with `dtype=torch.bfloat16` default (`load.py:57`, `:88`), called at `sdft/scripts/train_sdft.py:230` `model, tokenizer, stop_id = load_qwen35(recipe.model, device=device)` |
| trainer precision | EXPLICIT `main.py:232-233` `bf16 = True, fp16 = False,` | no HF Trainer; pure bf16 params |
| grad ckpt | **DEFAULT-ON**: `main.py` never sets `gradient_checkpointing`; `distil_config.py:259-260` `gradient_checkpointing: bool = field(default=True,` — only kwargs are explicit (`main.py:243` `gradient_checkpointing_kwargs = {"use_reentrant": False},`) | EXPLICIT `arc1-gold.yaml:168` `gradient_checkpointing: true`; applied at `scripts/train_sdft.py:233,239` `if recipe.optim["gradient_checkpointing"]: … model.gradient_checkpointing_enable()` |

Our SFT arm: `sft_main.py:132` `gradient_checkpointing=(args.dataset_name == "arc"),` ⇒ True for ARC.

### 3. Optimizer

**OURS** — `DistilConfig(TrainingArguments)` (`distil_config.py:22`), `DistilTrainer(BaseTrainer)` (`distil_trainer.py:219`) passes `optimizers=optimizers` (default `(None, None)`, `distil_trainer.py:321`, `:468`) to HF `Trainer` ⇒ **no custom optimizer**; `$ grep -n "create_optimizer\|AdamW" distil_trainer.py` shows only docstring text.
- optimizer type: **DEFAULT** `transformers/training_args.py:1292-1299`: `default_optim = "adamw_torch"` … `if is_torch_greater_or_equal_than_2_8: default_optim = "adamw_torch_fused"` — installed torch is `2.9.0+cu128` (`$ .venv/bin/python -c "import torch;print(torch.__version__)"`) ⇒ **`adamw_torch_fused`**.
- betas/eps: **DEFAULTS** `training_args.py:917-919` `adam_beta1 … default=0.9`, `adam_beta2 … default=0.999`, `adam_epsilon … default=1e-8`.
- weight_decay: **DEFAULT** `training_args.py:916` `weight_decay: float = field(default=0.0, …)` (not set in `main.py`).
- LR: **EXPLICIT** `run_arc_armA.sh:17` `LR=${LR:-1e-4}` → `main.py:228` `learning_rate = args.learning_rate,` (DistilConfig's own default would be 1e-6, `distil_config.py:248-249`).
- scheduler: **EXPLICIT** `main.py:230` `lr_scheduler_type = "cosine",` (HF default is `"linear"`, `training_args.py:927-928`).
- warmup: **EXPLICIT ratio** `main.py:229` `warmup_ratio = 0.1,`; steps derived `training_args.py:2555-2557` `warmup_steps = (self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio))`.
- max_grad_norm: **EXPLICIT** `main.py:247` `max_grad_norm = 1,`.

Our SFT arm (`sft_main.py:118-121`): **EXPLICIT** `warmup_steps=10,` `lr_scheduler_type="cosine",` `weight_decay=0.0,` `max_grad_norm=1.0,`; LR `1e-4` from `run_arc_armA.sh:54`.

**THEIRS** — `src/sdft/loop.py:238-246`:
```
238	    optimizer = torch.optim.AdamW(params, lr=settings.lr,
239	                                  weight_decay=settings.weight_decay)
...
242	    def lr_at(step: int) -> float:
243	        if step < settings.warmup_steps:
244	            return settings.lr * (step + 1) / settings.warmup_steps
245	        progress = (step - settings.warmup_steps) / max(settings.steps - settings.warmup_steps, 1)
246	        return settings.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
```
applied per step at `loop.py:388-390`:
```
388	        for group in optimizer.param_groups:
389	            group["lr"] = lr_at(step)
390	        grad_norm = torch.nn.utils.clip_grad_norm_(params, settings.max_grad_norm)
```
- betas/eps: **torch DEFAULTS** (not passed) → `torch.optim.AdamW` defaults `betas=(0.9, 0.999), eps=1e-8` (torch 2.x; I did not open their venv's torch — see Unknowns).
- weight_decay: **DEFAULT 0.0** — `loop.py:75` `weight_decay: float = 0.0`, `recipe.py:134` `"weight_decay": 0.0,`; not overridden in `arc1-gold.yaml`.
- LR: **EXPLICIT** `arc1-gold.yaml:159` `lr: 1.0e-4`.
- warmup: **EXPLICIT** `arc1-gold.yaml:160` `warmup_steps: 10`.
- max_grad_norm: **EXPLICIT** `arc1-gold.yaml:161` `max_grad_norm: 1.0`.
- Recipe→Settings wiring verified at `recipe.py:396-406` (`lr=optim["lr"], warmup_steps=optim["warmup_steps"], weight_decay=optim["weight_decay"], max_grad_norm=optim["max_grad_norm"], steps=optim["steps"], save_every=optim["save_every"], seed=optim["seed"]`).

### 4. Schedule / effective batch

**OURS (SDFT)** — `run_arc_armA.sh:19,30,31`:
```
19	EPOCHS=${EPOCHS:-2}
30	PDBS=${PDBS:-1}                 # per-device batch (1 keeps the long teacher forward in 48GB)
31	NPPB=${NPPB:-32}                # effective prompts per optimizer step
```
`main.py:234-235`: `per_device_train_batch_size = args.per_device_train_batch_size,` / `gradient_accumulation_steps = max(1, args.num_prompts_per_batch // args.per_device_train_batch_size),` ⇒ grad_accum = 32/1 = **32; effective prompts per optimizer step = 32**. `main.py:245` `num_generations = 1,` and `main.py:244` `num_iterations = 1,`.
Dataset: `$ .venv/bin/python -c "from datasets import load_from_disk; d=load_from_disk('data/arc_data/train_data'); print(len(d))"` → **381**.

**OURS (SFT)** — `run_arc_armA.sh:55` `--per_device_train_batch_size "$PDBS" --gradient_accumulation_steps "$NPPB"` ⇒ 1 × 32 = **32 prompts/step**, 2 epochs.

**THEIRS** — `arc1-gold.yaml:143-154`:
```
143	optim:
144	  # ONE EPOCH, ONE PROMPT A STEP. sdft/loop.py:276 records the identity
145	  #
146	  #     steps x prompts_per_step x grad_accum = len(pairs)
147	  #
148	  # and 416 x 1 x 1 = 416 rows exactly.
153	  steps: 416
154	  prompts_per_step: 1
155	  grad_accum: 1
```
`$ wc -l data/arc/train_rows_gold.jsonl` → **416**. Sampler `arc1-gold.yaml:79` `n_generations: 4` ⇒ 4 on-policy samples of the same prompt per step (`arc1-gold.yaml:150-152`: "n_generations=4 is what averages it — four on-policy samples of the same prompt, not four different prompts"). Total: **416 optimizer steps, 1 prompt (4 completions) each, 1 epoch.**

### 5. Checkpoint cadence / merge

- OURS SDFT: **EXPLICIT** `main.py:246` `save_steps = 100,`; `save_strategy` **DEFAULT** `"steps"` (`training_args.py:987-988`). Final adapter written by `main.py:276` `trainer.save_model(args.output_dir)`. Merge is a separate process: `run_arc_armA.sh:45-46` `uv run python merge_lora.py --base "$BASE" --adapter ckpt/arc_sdft_armA_adapter --out ckpt/arc_sdft_armA`, implemented `merge_lora.py:14-17` (`PeftModel.from_pretrained` → `merge_and_unload()` → `save_pretrained`).
- OURS SFT: `sft_main.py:124` `save_strategy="no",` + `sft_main.py:146` `trainer.save_model(...)`, then same `merge_lora.py` (`run_arc_armA.sh:57-58`).
- THEIRS: `arc1-gold.yaml:164` `save_every: 20` ("21 checkpoints over the epoch", `:162-163`); `loop.py:287-291` writes `checkpoints/step{step:06d}` via `trainee.save(path)`; `lora.py:75-79` saves the **adapter only** ("about 25 MB, never the 18 GB base"). **No merge step exists**: `$ grep -rn "merge_and_unload\|merge_adapter" src scripts` → no matches; evaluation loads the adapter live (`lora.py:47` `PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True)`, wired from `scripts/evaluate.py:144-150`).

### 6. Seed

- OURS: **DEFAULT 42** — `main.py:23` `parser.add_argument("--seed", type=int, default=42, …)`, `main.py:222` `seed=args.seed,`; `$ grep -n seed run_arc_armA.sh` → `NO MATCH`. (HF `TrainingArguments` default is also 42, `training_args.py:1063`.) `sft_main.py:24` same default, `sft_main.py:125` `seed=args.seed,`.
- THEIRS: **EXPLICIT 0** — `arc1-gold.yaml:169` `seed: 0` (also the schema default, `recipe.py:148`); used for both `set_seed` (`train_sdft.py:207`) and the data order generator (`loop.py:240` `generator = torch.Generator().manual_seed(settings.seed)`).

---

## Inferences (labeled)

1. **Ours runs ~22 optimizer steps total (SDFT).** Derivation: `distil_trainer.py:687` `"batch_size": self._train_batch_size * self.args.steps_per_generation`; `distil_config.py:678-680` sets `steps_per_generation = gradient_accumulation_steps` when unset; `RepeatSampler.__len__` = `(num_samples // batch_size) * batch_size * mini_repeat_count * repeat_count` (`trl/trainer/utils.py:1765`); HF step math `transformers/trainer.py:5682-5689`. This collapses to **steps/epoch = N // NPPB**. **Validated empirically** against a completed run of the same code: science N=2674, NPPB=32, 2 epochs → 2674//32 = 83 → predicted 166; `ckpt/sdft_science_adapter/checkpoint-166/trainer_state.json` → `{'epoch': 2.0, 'global_step': 166, 'max_steps': 166, 'num_train_epochs': 2}`. For ARC: 381//32 = 11/epoch × 2 = **22 steps**, consuming 352 of 381 rows per epoch (29 dropped by the floor).
2. **Ours SFT arm ≈ 24 optimizer steps** — standard Trainer path: len_dataloader = 381 (bs 1), `trainer.py:5682-5689` → 381//32 + 1 = 12/epoch × 2 = 24. Not validated against a saved trainer_state (SFT uses `save_strategy="no"`).
3. **Ours' warmup ≈ 3 steps of 22** (`math.ceil(22 * 0.1) = 3`, from `training_args.py:2555-2557`), i.e. ~14% of the run, vs theirs **10 of 416** (~2.4%). Our SFT arm's `warmup_steps=10` of ~24 steps = ~42% of the run.
4. **Optimizer math is effectively identical** (AdamW, 0.9/0.999, eps 1e-8, wd 0.0, clip 1.0 both sides); the only nominal difference is fused vs non-fused kernel. Not a loss-curve driver.
5. **Per-step gradient composition differs by ~8×** in sequences: ours 32 prompts × 1 generation = 32 sequences/step; theirs 1 prompt × 4 generations = 4 sequences/step.

---

## LoRA/optim/schedule differences that would change the LOSS CURVE (ranked)

1. **Optimizer-step count & prompts per step: 22 steps × 32 prompts (ours) vs 416 steps × 1 prompt (theirs).** (`run_arc_armA.sh:19,30,31` + `main.py:235`; `arc1-gold.yaml:153-155`.) Ours produces a 22-point curve from large, low-variance batches; theirs a 416-point curve from single-prompt (4-sample) gradients. Cosine decay is stretched over completely different horizons, so LR-vs-progress is not comparable at any matching x-axis.
2. **LoRA scaling 2.0 vs 1.0** (α=32/r=16 vs α=16/r=16; `main.py:26` vs `lora.py:54`). At the same LR 1e-4, our adapter's contribution to the forward is doubled per unit of learned weight — an effective-LR difference in disguise (exactly what their comment at `lora.py:51-53` says they avoid).
3. **Warmup fraction: ~3/22 ≈ 14% (ours, ratio 0.1) vs 10/416 ≈ 2.4% (theirs).** With only 22 steps, ~3 of them are ramping and the cosine then collapses to ~0 within ~19 steps — the visible "loss goes flat" region is a schedule artifact.
4. **lora_dropout 0.05 vs 0.0** (`main.py:27` vs `lora.py:40`). Ours injects stochastic dropout into the *student* forward while the teacher is the same weights with the adapter disabled (`main.py:197-200`) — adds variance to the student-vs-teacher divergence at every step; theirs is deterministic.
5. **Adapted module set / model: 7 named projections on dense Qwen3-8B vs `all-linear` on hybrid Qwen3.5-9B** (`main.py:262` vs `lora.py:30` + `arc1-gold-lora16.yaml:25-29`). Different parameter count and different layer coverage (their file notes a `q_proj`-style list would adapt only 8 of 32 layers on the hybrid stack).
6. **Generations per prompt: 1 (ours, `main.py:245`) vs 4 (theirs, `arc1-gold.yaml:79`)** — theirs averages 4 on-policy samples per prompt, ours does not, changing per-step loss variance.
7. **Teacher prompt template.** Ours uses "*Now answer with a response of your own, including the thinking process.*" (`main.py:159`); theirs deliberately switched to `template: worked_answer` (`arc1-gold.yaml:124`) because that exact clause caused a completion-length ratchet (`arc1-gold.yaml:95-124`, with measured tokens/truncation table at `:105-116`). This directly moves loss and step time.
8. **Loss-token masking: `num_loss_tokens_to_skip = 3` (ours, `main.py:255`) vs `skip_first: 0` (theirs, `recipe.py:68`)** — different token sets enter the loss, so absolute loss values are not comparable.
9. **Epochs & data coverage: 2 epochs over 381 rows, 29 rows dropped per epoch by the batch floor (inference #1) vs 1 exact epoch over 416 rows** (`arc1-gold.yaml:144-148`, `loop.py:261-263`).
10. **Sampler engine / IS correction:** ours vLLM colocate + `vllm_importance_sampling_correction = True` (`main.py:223-227,254`); theirs HF sampler to make sampler and scorer the same engine (`arc1-gold.yaml:74-78`). Different numeric path into the same objective.
11. **Length budgets:** ours prompt 10240 / completion 4096 (`run_arc_armA.sh:27,29`) vs theirs prompt 12288 / sample 3072 (`arc1-gold.yaml:140`, `:86`). Theirs measured 12288 as the cap that drops 0 of 416 rows; 10240 likely truncates some teacher prompts (their table at `:130-137` shows teacher max 11244).
12. **Seed 42 vs 0** (`main.py:23` vs `arc1-gold.yaml:169`) — different data order and sampling noise; irrelevant to method, relevant to curve-to-curve overlay.

---

## Methods (commands run)

```
ls -la /home/xwang3/Projects/Self-Distillation/ ; ls -la <THEIRS>
find <THEIRS> -type f \( -name "*.py" -o -name "*.yaml" \)
cat -n <THEIRS>/recipes/arc1-gold.yaml ; cat -n <THEIRS>/recipes/arc1-gold-lora16.yaml ; cat -n <THEIRS>/recipes/_base.yaml
cat -n <THEIRS>/src/sdft/trainees/lora.py
grep -n "" <THEIRS>/src/sdft/recipe.py | sed -n '1,200p'
grep -n "AdamW\|betas\|weight_decay\|lr\b\|scheduler\|warmup\|max_grad_norm\|seed\|save_every\|dtype\|gradient_checkpointing\|prompts_per_step\|grad_accum\|steps" <THEIRS>/src/sdft/loop.py
awk 'NR>=X && NR<=Y' <THEIRS>/src/sdft/loop.py            # lines 60-100, 230-300, 385-395
grep -rn "AutoModelForCausalLM\|gradient_checkpointing_enable" <THEIRS>/src <THEIRS>/scripts
awk 'NR>=190 && NR<=260' <THEIRS>/scripts/train_sdft.py
awk 'NR>=20 && NR<=105' <THEIRS>/../qwen35/src/qwen35/load.py
grep -n "def settings" -A 45 <THEIRS>/src/sdft/recipe.py
wc -l <THEIRS>/data/arc/train_rows_gold.jsonl                      # 416
grep -n "name = \"peft\"" -A 2 <THEIRS>/uv.lock                    # 0.20.0
grep -rn "merge_and_unload\|merge_adapter" <THEIRS>/src <THEIRS>/scripts   # no matches
cat -n /home/xwang3/Projects/Self-Distillation/{main.py,run_arc_armA.sh,merge_lora.py,sft_main.py}
grep -n "gradient_checkpointing\|learning_rate\|logging_steps" distil_config.py ; awk 'NR>=244 && NR<=275' distil_config.py
grep -n "create_optimizer\|AdamW\|optimizer\|peft_config\|num_generations\|steps_per_generation" distil_trainer.py
awk 'NR>=675 && NR<=760' distil_trainer.py ; grep -n "steps_per_generation" distil_config.py
.venv/bin/python -c "import transformers,trl,peft,torch; print(...)"   # 4.57.1 / 0.24.0 / 0.17.1 / 2.9.0+cu128
grep -n "^    weight_decay\|adam_beta\|max_grad_norm\|lr_scheduler_type\|warmup\|seed\|save_strategy\|save_steps\|default_optim" .venv/.../transformers/training_args.py
grep -n "def get_warmup_steps" -A 10 .venv/.../transformers/training_args.py
awk 'NR>=5670 && NR<=5712' .venv/.../transformers/trainer.py
grep -rn "self.scaling\[adapter_name\] = " .venv/.../peft/tuners/lora/layer.py
grep -rn "def __len__" -A 4 .venv/.../trl/trainer/utils.py ; awk 'NR>=525 && NR<=560' .venv/.../trl/models/utils.py
.venv/bin/python -c "from datasets import load_from_disk; ..."     # arc 381, science 2674
python3 -c "import json; json.load(open('ckpt/sdft_science_adapter/checkpoint-166/trainer_state.json'))"
cat ckpt/arc_sft_armA_adapter/adapter_config.json ; ls -la ckpt/arc_sdft_armA_adapter   # empty
grep -n "seed\|lora_alpha\|lora_dropout" run_arc_armA.sh            # NO MATCH (all three)
```

## Unknowns / caveats

- **Their actual runtime values are un-recorded here**: `$ find <THEIRS> -name run_meta.json` → nothing; the snapshot has no `runs/` directory, so I verified their config from recipes/code only, not from a resolved run record. A `--set optim.lr=…` override at launch would not be visible to me.
- **Their torch/peft versions were not introspected** (no venv in the snapshot). `uv.lock` pins `peft 0.20.0`; `pyproject.toml:10` requires `transformers>=5.13`. The `lora_alpha / r` scaling formula I cite is from **peft 0.17.1** installed in *our* venv; I did not confirm peft 0.20 keeps it (it almost certainly does, but that is an inference).
- **`torch.optim.AdamW` default betas/eps for their torch build** were not read from their installed torch — stated from the documented torch defaults, not verified on disk.
- **Our SDFT ARC adapter directory is empty**, so I could not confirm the SDFT run's realized `adapter_config.json`; the LoRA facts for the SDFT arm come from `main.py` + argparse defaults, cross-checked against the SFT arm's identical `LoraConfig` and its written config.
- **Our SFT arm's 24-step figure is unvalidated** (no `trainer_state.json`, `save_strategy="no"`).
- I did **not** verify whether any of these ARC runs actually completed; `wandb/offline-run-20260813_*/files/output.log` files are empty and no ARC `trainer_state.json` exists, so no empirical step count for the ARC runs was obtainable.