# Figure 3 Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Figure 3's SDFT-vs-SFT sequential continual-learning trend on Science Q&A → Tool Use with Qwen3-8B + LoRA, on 8× A6000.

**Architecture:** Two arms (SDFT via the existing `DistilTrainer`; SFT via a new TRL `SFTTrainer` script) each trained sequentially with LoRA, merging the adapter into a full model between stages so vLLM eval scripts load them directly. Both eval scripts run on 5 checkpoints → 10 accuracies → per-task normalized 2-panel plot.

**Tech Stack:** Python 3.12, uv, PyTorch 2.9, transformers 4.57.1, trl 0.24.0, peft 0.17.1, vllm 0.12.0, datasets 4.3.0, matplotlib.

## Global Constraints

- Model: `Qwen/Qwen3-8B` (do NOT upgrade transformers/vllm/trl; pinned versions support it).
- Do NOT edit `distil_trainer.py`, `distil_config.py`, `eval_science.py`, `eval_tooluse.py`.
- Fine-tuning: LoRA `r=16 alpha=32 dropout=0.05`; LR 1e-4; bf16; cosine; 10 warmup steps; weight decay 0; max grad norm 1; seed 42; batch = 32 prompts (grad-accum), per-device batch 1.
- SDFT: 2 epochs, EMA α 0.01, forward-KL (`beta=0.0`), 1 rollout, max gen 2048. SFT: 1 epoch.
- Tasks & order: Science Q&A → Tool Use. 1 seed only.
- All outputs under `results/fig3/` and `ckpt/`. Never commit model weights or datasets.
- Use `uv run python …` for all Python; `CUDA_VISIBLE_DEVICES` to place runs on specific GPUs.

---

### Task 1: Environment setup + base-model eval parseability check

**Files:**
- Create: `.venv/` (via uv), `results/fig3/` (dir)
- Create: `scripts/check_base_parseable.py`

**Interfaces:**
- Produces: a working `.venv`; confirmation that base `Qwen/Qwen3-8B` produces outputs the two eval parsers can read (science `<answer>…</answer>`, tooluse `Action Input: {…}`), and the correct `enable_thinking` setting to use.

- [ ] **Step 1: Create venv and install deps**

Run:
```bash
cd /home/xwang3/Projects/Self-Distillation
uv venv --python 3.12
uv pip install -r requirements.txt
```
Expected: all pins resolve. If flashinfer/vLLM JIT errors appear, note them; vLLM colocate needs a working CUDA — see repo gotchas.

- [ ] **Step 2: Confirm Qwen3-8B loads in transformers**

Run:
```bash
uv run python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('Qwen/Qwen3-8B'); print(c.model_type)"
```
Expected: prints `qwen3` (not an unknown-arch error). This confirms the pinned transformers supports it.

- [ ] **Step 3: Write the base-parseability probe**

Create `scripts/check_base_parseable.py`: load a few science eval prompts and a few tooluse eval prompts, generate with vLLM from base `Qwen/Qwen3-8B` (greedy, max 2048), and run the *actual* eval extractors on the output. Try both `enable_thinking=True` and `enable_thinking=False` in the chat template and report, for each, how many outputs are parseable.

```python
import sys
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# reuse the repo's own extractors so the probe matches eval exactly
from eval_science import extract_xml_answer
import eval_tooluse

def probe(model="Qwen/Qwen3-8B", n=8):
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, gpu_memory_utilization=0.6, dtype="bfloat16",
              max_model_len=4096, trust_remote_code=True)
    sci = Dataset.load_from_disk("data/science_data/eval_data").select(range(n))
    tu = Dataset.load_from_disk("data/tooluse_data/eval_data").select(range(n))
    for thinking in (True, False):
        # science
        sp = SamplingParams(temperature=0.0, max_tokens=2048)
        sci_prompts = [tok.apply_chat_template(e["prompt"], tokenize=False,
                        add_generation_prompt=True, enable_thinking=thinking) for e in sci]
        sci_out = [o.outputs[0].text for o in llm.generate(sci_prompts, sp)]
        sci_ok = sum(1 for o in sci_out if extract_xml_answer(o) != "")
        print(f"enable_thinking={thinking}: science parseable {sci_ok}/{n}")
        # tooluse — count outputs containing an Action Input JSON block
        tu_prompts = [tok.apply_chat_template(
            [{"role":"user","content":e["prompt"]}] if isinstance(e["prompt"], str) else e["prompt"],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for e in tu]
        tu_out = [o.outputs[0].text for o in llm.generate(tu_prompts, sp)]
        import re
        tu_ok = sum(1 for o in tu_out if re.search(r'Action Input:\s*\{', o))
        print(f"enable_thinking={thinking}: tooluse action-input present {tu_ok}/{n}")

if __name__ == "__main__":
    probe()
```

- [ ] **Step 4: Run the probe**

Run: `CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_base_parseable.py 2>&1 | tee results/fig3/base_parseable.log`
Expected: at least one `enable_thinking` setting gives mostly-parseable output for both tasks. **Record which setting to use.** If `enable_thinking=False` is needed, note that the eval scripts (which don't pass the flag) may need a wrapper — see Task 6. Inspect the tooluse eval prompt field name (`prompt` vs message list) and fix the probe accordingly if it errors.

- [ ] **Step 5: Commit the probe (not the venv)**

```bash
git add scripts/check_base_parseable.py
git commit -m "test: add base Qwen3-8B eval-parseability probe"
```

---

### Task 2: Add LoRA + init-from-checkpoint support to the SDFT path (`main.py`)

**Files:**
- Modify: `main.py`
- Test: manual smoke run (below)

**Interfaces:**
- Consumes: `DistilTrainer(..., peft_config=…)` (already supported, `distil_trainer.py:260`).
- Produces: `main.py` now accepts `--peft` (enable LoRA), `--lora_r`, `--lora_alpha`, `--init_model_path` (weights to start from; defaults to `--model_name`; tokenizer still from `--model_name`), and writes a LoRA adapter to `--output_dir`.

- [ ] **Step 1: Add CLI args**

In `main.py::parse_args`, add:
```python
parser.add_argument("--peft", action="store_true", help="Use LoRA")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--init_model_path", type=str, default=None,
                    help="Path to init weights (merged checkpoint for stage 2). Defaults to --model_name.")
```

- [ ] **Step 2: Load from init_model_path and build peft_config**

In `__main__`, change the model/teacher loading to use `init_path = args.init_model_path or args.model_name` for `from_pretrained` weights, keep tokenizer from `args.model_name`, and construct:
```python
peft_config = None
if args.peft:
    from peft import LoraConfig
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
```
Pass `peft_config=peft_config` to `DistilTrainer(...)`. Note: the teacher/ref_model must remain the frozen full model (do NOT wrap teacher in LoRA) — the trainer already treats `ref_model` separately.

- [ ] **Step 3: Add lora hyperparams to logging tags (optional) and set `report_to`**

Keep `report_to="wandb"` but allow `WANDB_MODE=offline`; no code change needed if env var is set at launch.

- [ ] **Step 4: Smoke-run a few steps**

Run (tiny, just to prove it trains + saves an adapter):
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run python main.py \
  --dataset_name science --model_name Qwen/Qwen3-8B --peft \
  --output_dir ckpt/_smoke_sdft --learning_rate 1e-4 --num_train_epochs 1 \
  2>&1 | tee results/fig3/_smoke_sdft.log
```
Let it run ~10–20 optimizer steps, then stop it. Expected: loss logged, no crash, `ckpt/_smoke_sdft/` contains `adapter_model.safetensors`. If OOM: lower `vllm_gpu_memory_utilization` in `main.py` config or per-device settings. Record the working config.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: add LoRA + init-from-checkpoint support to SDFT main"
```

---

### Task 3: SFT baseline script (`sft_main.py`)

**Files:**
- Create: `sft_main.py`

**Interfaces:**
- Consumes: repo dataset loaders' field conventions — science `output_text`, tooluse `golden_response` (list), and prompts as in `main.py::load_*_dataset`.
- Produces: `sft_main.py` with the same CLI shape as `main.py` (`--dataset_name {science,tooluse}`, `--model_name`, `--init_model_path`, `--output_dir`, `--learning_rate`, `--num_train_epochs`, `--peft`, lora args), training a LoRA adapter via TRL `SFTTrainer` on prompt→golden-completion chat examples, writing the adapter to `--output_dir`.

- [ ] **Step 1: Write `sft_main.py`**

Build chat-formatted examples (`messages` = [user prompt, assistant golden]) and train with `SFTTrainer`:
```python
import argparse, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", choices=["science","tooluse"], required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-8B")
    p.add_argument("--init_model_path", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--peft", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def build_dataset(name, seed):
    if name == "science":
        ds = load_from_disk("data/science_data/train_data")
        def fmt(e):
            return {"messages": [e["messages"][0], e["messages"][1],
                    {"role":"assistant","content": e["output_text"]}]}
        # messages[0]=system?, messages[1]=user — mirror main.py science loader (uses messages[0],[1])
        cols = ds.column_names
        return ds.map(fmt, remove_columns=cols).shuffle(seed=seed)
    else:
        ds = load_from_disk("data/tooluse_data/train_data")
        def fmt(e):
            return {"messages": [{"role":"user","content": e["prompt"]},
                    {"role":"assistant","content": "\n".join(e["golden_response"])}]}
        cols = ds.column_names
        return ds.map(fmt, remove_columns=cols).shuffle(seed=seed)

if __name__ == "__main__":
    a = parse_args()
    init = a.init_model_path or a.model_name
    model = AutoModelForCausalLM.from_pretrained(init, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    ds = build_dataset(a.dataset_name, a.seed)
    peft_config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM") if a.peft else None
    cfg = SFTConfig(
        output_dir=a.output_dir, learning_rate=a.learning_rate,
        num_train_epochs=a.num_train_epochs,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        warmup_steps=10, lr_scheduler_type="cosine", weight_decay=0.0,
        max_grad_norm=1.0, bf16=True, logging_steps=1, save_strategy="no",
        seed=a.seed, report_to="none", max_length=3072, packing=False,
        assistant_only_loss=True,  # train only on the golden completion tokens
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=peft_config)
    trainer.train()
    trainer.save_model(a.output_dir)
```
Note: confirm the science `train_data` schema (`messages`, `output_text`) matches `main.py::load_science_dataset` — it reads `example["messages"][0]`, `example["messages"][1]`, `example["output_text"]`. If `assistant_only_loss` is unsupported in trl 0.24, drop it and rely on chat-template masking, or set a response template.

- [ ] **Step 2: Verify dataset schema before training**

Run:
```bash
uv run python -c "
from datasets import load_from_disk
s=load_from_disk('data/science_data/train_data'); print('science cols', s.column_names); print(s[0].keys())
t=load_from_disk('data/tooluse_data/train_data'); print('tooluse cols', t.column_names); print({k:type(t[0][k]).__name__ for k in t[0]})
"
```
Expected: science has `messages` + `output_text`; tooluse has `prompt` + `golden_response` (list). Fix `build_dataset` if fields differ.

- [ ] **Step 3: Smoke-run SFT a few steps**

Run:
```bash
CUDA_VISIBLE_DEVICES=1 uv run python sft_main.py --dataset_name science --peft \
  --output_dir ckpt/_smoke_sft --num_train_epochs 1 \
  2>&1 | tee results/fig3/_smoke_sft.log
```
Stop after a few steps. Expected: loss decreases, `ckpt/_smoke_sft/adapter_model.safetensors` written.

- [ ] **Step 4: Commit**

```bash
git add sft_main.py
git commit -m "feat: add LoRA SFT baseline script"
```

---

### Task 4: LoRA merge helper (`merge_lora.py`)

**Files:**
- Create: `merge_lora.py`

**Interfaces:**
- Consumes: an adapter dir (from Task 2/3) + a base/init model path.
- Produces: `merge_lora.py --base <path> --adapter <dir> --out <dir>` writing a full merged model + tokenizer to `<dir>` (loadable by the vLLM eval scripts via `--model_path <dir>`).

- [ ] **Step 1: Write `merge_lora.py`**

```python
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base/init model path")
    p.add_argument("--adapter", required=True, help="LoRA adapter dir")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=None, help="Defaults to --base")
    a = p.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.tokenizer or a.base).save_pretrained(a.out)
    print(f"Merged model saved to {a.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test merge on the smoke adapter**

Run:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python merge_lora.py \
  --base Qwen/Qwen3-8B --adapter ckpt/_smoke_sdft --out ckpt/_smoke_merged
uv run python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('ckpt/_smoke_merged'); print('merged loads OK')"
```
Expected: merged dir has `model-*.safetensors` + `config.json` + tokenizer; loads without error.

- [ ] **Step 3: Commit**

```bash
git add merge_lora.py
git commit -m "feat: add LoRA merge helper"
```

---

### Task 5: Optional stage-1 LR check (Science only)

**Files:**
- Create: `results/fig3/lr_check.md` (record)

**Interfaces:**
- Consumes: Tasks 2–4. Produces: chosen LR for both arms (best Science accuracy), recorded.

- [ ] **Step 1: Train Science SDFT at two LRs, in parallel on 2 GPUs**

Run (background, one per GPU):
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run python main.py --dataset_name science \
  --model_name Qwen/Qwen3-8B --peft --learning_rate 5e-5 --num_train_epochs 2 \
  --output_dir ckpt/lr5e5_sdft_science
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline uv run python main.py --dataset_name science \
  --model_name Qwen/Qwen3-8B --peft --learning_rate 1e-4 --num_train_epochs 2 \
  --output_dir ckpt/lr1e4_sdft_science
```

- [ ] **Step 2: Merge + eval Science for both**

For each: `merge_lora.py` then `eval_science.py --model_path <merged> --output_dir results/fig3/lrcheck_<lr>_science`.

- [ ] **Step 3: Pick LR and record**

Write the two Science accuracies + chosen LR to `results/fig3/lr_check.md`. Use the winner for the full run (Task 6). If both similar, use 1e-4.

- [ ] **Step 4: Commit the record**

```bash
git add results/fig3/lr_check.md
git commit -m "chore: record stage-1 LR check result"
```

> Note: this task may be skipped (use LR 1e-4 directly) to save compute; if skipped, say so in the record file.

---

### Task 6: Full sequential training — both arms (`run_fig3.sh`)

**Files:**
- Create: `run_fig3.sh`

**Interfaces:**
- Consumes: Tasks 2–4 + chosen LR. Produces 4 merged checkpoints:
  `ckpt/sdft_science`, `ckpt/sdft_science_tooluse`, `ckpt/sft_science`, `ckpt/sft_science_tooluse`.

- [ ] **Step 1: Write `run_fig3.sh`**

Encode the dependency graph: within each arm, stage 2 needs stage 1's merged model; the two arms are independent. Use `CUDA_VISIBLE_DEVICES` to place the two arms on different GPUs and run them concurrently (`&` + `wait`), OR document that the orchestrator (subagent-driven executor) will dispatch the two arms as parallel subagents. Each stage = train adapter → merge → (adapter dir kept for provenance).

Structure (LR is a variable; `$D` = device):
```bash
#!/usr/bin/env bash
set -euo pipefail
LR=${LR:-1e-4}
BASE=Qwen/Qwen3-8B
export WANDB_MODE=offline
run_sdft () { # dataset init_path out_adapter out_merged device
  CUDA_VISIBLE_DEVICES=$5 uv run python main.py --dataset_name $1 --model_name $BASE \
    --init_model_path $2 --peft --learning_rate $LR --num_train_epochs 2 --output_dir $3
  CUDA_VISIBLE_DEVICES=$5 uv run python merge_lora.py --base $2 --adapter $3 --out $4
}
run_sft () { # dataset init_path out_adapter out_merged device
  CUDA_VISIBLE_DEVICES=$5 uv run python sft_main.py --dataset_name $1 --model_name $BASE \
    --init_model_path $2 --peft --learning_rate $LR --num_train_epochs 1 --output_dir $3
  CUDA_VISIBLE_DEVICES=$5 uv run python merge_lora.py --base $2 --adapter $3 --out $4
}
# SDFT arm on GPU 0; SFT arm on GPU 1 (run the two arms in parallel via subagents)
run_sdft science $BASE ckpt/sdft_science_adapter ckpt/sdft_science 0
run_sdft tooluse ckpt/sdft_science ckpt/sdft_science_tooluse_adapter ckpt/sdft_science_tooluse 0
run_sft  science $BASE ckpt/sft_science_adapter ckpt/sft_science 1
run_sft  tooluse ckpt/sft_science ckpt/sft_science_tooluse_adapter ckpt/sft_science_tooluse 1
```
Note: `merge_lora.py --base` must be the SAME `init_path` the adapter was trained on (stage-2 base = stage-1 merged model), else the merge is applied to the wrong weights.

- [ ] **Step 2: Launch both arms (parallel, background)**

Dispatch as two parallel subagents (per user CLAUDE.md), each running its arm's two stages sequentially. SDFT arm on GPU 0, SFT arm on GPU 1. Monitor via `BashOutput` / log tail; do not poll with `sleep`.

- [ ] **Step 3: Verify all 4 merged checkpoints exist and load**

Run:
```bash
for d in ckpt/sdft_science ckpt/sdft_science_tooluse ckpt/sft_science ckpt/sft_science_tooluse; do
  uv run python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$d'); print('OK $d')"
done
```
Expected: 4 × OK.

- [ ] **Step 4: Commit the orchestrator**

```bash
git add run_fig3.sh
git commit -m "feat: add Figure 3 sequential-training orchestrator"
```

---

### Task 7: Evaluation grid (`eval_grid.sh`) → `accuracy.json`

**Files:**
- Create: `eval_grid.sh`, `collect_accuracy.py`

**Interfaces:**
- Consumes: 4 merged checkpoints + base model. Produces `results/fig3/accuracy.json`:
  `{ "<checkpoint>": {"science": <float>, "tooluse": <float>}, ... }` for the 5 checkpoints.

- [ ] **Step 1: Write `eval_grid.sh`**

For each checkpoint in `{Qwen/Qwen3-8B(as "base"), sdft_science, sdft_science_tooluse, sft_science, sft_science_tooluse}` and each task in `{science, tooluse}`, run the matching eval script with a distinct `--output_dir results/fig3/eval/<ckpt>/<task>` (the scripts write a fixed `eval_results.json`). Place evals across GPUs 0–7 to parallelize.
```bash
#!/usr/bin/env bash
set -euo pipefail
declare -A CKPT=( [base]=Qwen/Qwen3-8B [sdft_science]=ckpt/sdft_science \
  [sdft_science_tooluse]=ckpt/sdft_science_tooluse [sft_science]=ckpt/sft_science \
  [sft_science_tooluse]=ckpt/sft_science_tooluse )
g=0
for name in "${!CKPT[@]}"; do
  path=${CKPT[$name]}
  CUDA_VISIBLE_DEVICES=$((g%8)) uv run python eval_science.py --model_path "$path" \
    --output_dir results/fig3/eval/$name/science; g=$((g+1))
  CUDA_VISIBLE_DEVICES=$((g%8)) uv run python eval_tooluse.py --model_path "$path" \
    --output_dir results/fig3/eval/$name/tooluse; g=$((g+1))
done
```
If Task 1 found `enable_thinking=False` is required and the eval scripts hardcode the template, add a minimal env-var or wrapper (e.g., a `EVAL_ENABLE_THINKING=0` read inside a tiny sitecustomize, OR — simpler — set the tokenizer's chat_template default) documented here; keep the eval `.py` files unedited per constraints by wrapping rather than editing. If wrapping proves impossible without editing, note the deviation and get sign-off.

- [ ] **Step 2: Run the grid**

Run `bash eval_grid.sh 2>&1 | tee results/fig3/eval_grid.log`. Expected: 10 `eval_results.json` files under `results/fig3/eval/*/*/`.

- [ ] **Step 3: Write `collect_accuracy.py`**

```python
import json, os
CKPTS = ["base","sdft_science","sdft_science_tooluse","sft_science","sft_science_tooluse"]
TASKS = ["science","tooluse"]
out = {}
for c in CKPTS:
    out[c] = {}
    for t in TASKS:
        p = f"results/fig3/eval/{c}/{t}/eval_results.json"
        out[c][t] = json.load(open(p))["accuracy"]
json.dump(out, open("results/fig3/accuracy.json","w"), indent=2)
print(json.dumps(out, indent=2))
```
Run it. Expected: `results/fig3/accuracy.json` with 10 numbers printed.

- [ ] **Step 4: Commit**

```bash
git add eval_grid.sh collect_accuracy.py results/fig3/accuracy.json results/fig3/eval_grid.log
git commit -m "feat: add eval grid + accuracy collection; add Fig3 accuracies"
```

---

### Task 8: Normalize + plot (`plot_fig3.py`)

**Files:**
- Create: `plot_fig3.py`

**Interfaces:**
- Consumes: `results/fig3/accuracy.json`. Produces `results/fig3/figure3_reproduction.png` and a printed normalized table.

- [ ] **Step 1: Write `plot_fig3.py`**

Stage index: base=0, after-Science=1, after-ToolUse=2. Per task, per method, the accuracy at each stage:
- SDFT: stage0=base, stage1=sdft_science, stage2=sdft_science_tooluse
- SFT: stage0=base, stage1=sft_science, stage2=sft_science_tooluse

Normalize per task: `norm = (acc - base) / (maxall - base)` where `maxall` = max over both methods & all stages for that task; guard `maxall==base`.
```python
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = json.load(open("results/fig3/accuracy.json"))
def series(method, task):
    return [A["base"][task], A[f"{method}_science"][task], A[f"{method}_science_tooluse"][task]]
tasks = ["science","tooluse"]
fig, axes = plt.subplots(1, 2, figsize=(11,4))
for ax, task in zip(axes, tasks):
    base = A["base"][task]
    raw = {m: series(m, task) for m in ["sdft","sft"]}
    maxall = max(max(v) for v in raw.values())
    denom = (maxall - base) or 1.0
    for m, style in [("sdft","-o"),("sft","--s")]:
        norm = [(x-base)/denom for x in raw[m]]
        ax.plot([0,1,2], norm, style, label=m.upper())
    ax.set_title(task); ax.set_xlabel("training stage"); ax.set_ylabel("normalized accuracy")
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["base","+Science","+ToolUse"]); ax.legend(); ax.grid(alpha=.3)
fig.suptitle("Figure 3 reproduction (2-task, Qwen3-8B, LoRA)")
fig.tight_layout(); fig.savefig("results/fig3/figure3_reproduction.png", dpi=150)
# also print the raw + normalized table
for task in tasks:
    print(task, {m: series(m, task) for m in ["sdft","sft"]})
```

- [ ] **Step 2: Run and eyeball the figure**

Run `uv run python plot_fig3.py`. Open/inspect `results/fig3/figure3_reproduction.png`. Expected qualitative pattern: SDFT Science stays high at stage 2; SFT Science drops at stage 2 (forgetting); both gain Tool Use at stage 2. **If the pattern does not hold, report the actual numbers plainly — do not force the narrative.**

- [ ] **Step 3: Commit**

```bash
git add plot_fig3.py results/fig3/figure3_reproduction.png
git commit -m "feat: add Figure 3 normalization + plot; add reproduced figure"
```

---

### Task 9: Wrap-up — README note + results summary

**Files:**
- Create: `results/fig3/README.md`
- Modify: repo `README.md` (optional short pointer)

**Interfaces:**
- Consumes: everything. Produces a short human-readable summary of the reproduction, deviations from the paper, and how to re-run.

- [ ] **Step 1: Write `results/fig3/README.md`**

Include: the 10 raw accuracies, the normalized table, the figure, the exact commands to reproduce (env → run_fig3.sh → eval_grid.sh → collect_accuracy.py → plot_fig3.py), and the explicit deviations from the paper (2-task not 3, Qwen3-8B not Qwen3.5-9B/Qwen2.5-7B, LoRA not full FT, 1 seed, chosen LR, enable_thinking setting).

- [ ] **Step 2: Commit**

```bash
git add results/fig3/README.md
git commit -m "docs: add Figure 3 reproduction results + summary"
```

---

## Self-Review

- **Spec coverage:** env+parseability (T1), SDFT LoRA + sequential init (T2, T6), SFT baseline (T3, T6), merge-between-stages (T4, T6), eval grid on 5 checkpoints (T7), normalization + 2-panel plot (T8), deviations/summary (T9), LR check (T5). All spec sections mapped.
- **Placeholders:** none — every code step has concrete code; verification steps have exact commands + expected output.
- **Type/name consistency:** checkpoint names (`sdft_science`, `sdft_science_tooluse`, `sft_science`, `sft_science_tooluse`, `base`) are identical across T6/T7/T8; `accuracy.json` shape defined in T7 and consumed in T8; `--init_model_path`/`--peft` defined in T2/T3 and used in T6.
- **Known open items to resolve during execution (not placeholders):** (a) exact science `train_data` schema for SFT — verified in T3 step 2 before use; (b) `enable_thinking` handling in eval scripts without editing them — T1 records the setting, T7 wraps; if impossible without an edit, flag for sign-off; (c) `assistant_only_loss` availability in trl 0.24 — T3 step 1 has a fallback.
