# Data-pipeline / prompt / length comparison: OURS (Qwen3-8B) vs THEIRS (Qwen3.5-9B)

**Question restated:** for both SDFT-on-ARC-AGI-1 codebases, what exactly is the teacher prompt vs student prompt, is thinking on/off at render, what is the teacher conditioned on, what are the prompt/completion caps and how are over-length rows handled, what length-growth analysis exists, and how is a truncated-before-JSON answer scored at eval.

Paths: **OURS** = `/home/xwang3/Projects/Self-Distillation` (abbrev. `O/`). **THEIRS** = `/tmp/claude-23749/-home-xwang3-Projects-Self-Distillation/299f8f15-91ed-45bc-ac4e-bbc910d3f288/scratchpad/know-vec/sdft` (abbrev. `T/`).

---

## Answer (3 sentences)

Our ARC teacher prompt is **byte-identical to the reference "demonstration" template that THEIRS deliberately abandoned** — it ends `"Now answer with a response of your own, including the thinking process."` while our tokenizer patch forces `enable_thinking` off (`<think>\n\n</think>` emitted at the generation prompt), which is exactly the mismatch their `arc1-gold.yaml` documents as the mechanism behind a 3× completion-length explosion. Second, our teacher is conditioned on a **full worked CoT demonstration** (p50 15,369 chars) vs their **bare answer grid** (p50 312 chars), and our 10240 cap **left-truncates 161/381 = 42.3% of teacher prompts** (silently dropping the question) whereas theirs **drops** over-length rows with a 1%-loss guard at a cap of 12288 chosen so 0 rows drop. Third, our completion budgets are larger (train 4096, eval 12288 vs their 3072/3072) and we have **no code or comment anywhere analysing completion-length growth**, while they have a measured step-by-step table.

---

## Verified facts

### 1. Teacher and student prompt text

**OURS — teacher template** (`O/main.py:152-169`, `load_arc_dataset`), verbatim:
```python
    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")
        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][0]['content'],
                    output_text=example['output_text'],
                )},
            ],
        }
```
- **It DOES end with `, including the thinking process.`** — `O/main.py:159`.
- Note the leading `\n` (template starts with a newline before `$orig_content`) and that it is a **single user turn, no system message** — `O/main.py:164-168`; docstring `O/main.py:145-147`: *"Arm A: the teacher is conditioned on question + full gold demonstration; the student sees the question alone."*
- The same template text is used for tooluse (`O/main.py:55-62`), science (`:81-88`), medical (`:116-123`).

**OURS — student prompt** = `example["messages"]`, a single user turn built by `O/prep_arc.py:31-39 / 55-59`:
```python
INSTRUCTION = (
    "Infer the transformation demonstrated by the ARC examples and apply it to every "
    "test input. Return exactly one JSON object with no extra keys, using the schema "
    '{"outputs":[<output grid for test 0>, ...]}.\n'
    "ARC task:\n"
)
...
    payload = {"train": task["train"], "test_inputs": [t["input"] for t in task["test"]]}
    return INSTRUCTION + json.dumps(payload, separators=(",", ":"))
```
(`O/prep_arc.py:23-28` and `:36-39`) — i.e. grids as **compact JSON**, all test inputs in one prompt.

**THEIRS — three named templates** (`T/src/sdft/data.py:33-99`), verbatim:
```python
TEACHER_TEMPLATE = """{prompt}

This is an example for a response to the question:
{privileged}

Now answer with a response of your own, including the thinking process.
"""                                            # data.py:33-39
...
WORKED_ANSWER_TEMPLATE = """{prompt}

This is an example for a response to the question:
{privileged}

Now answer with a response of your own.
"""                                            # data.py:87-93
...
TEMPLATES = {"demonstration": TEACHER_TEMPLATE,
             "worked_answer": WORKED_ANSWER_TEMPLATE,
             "insight": INSIGHT_TEMPLATE}      # data.py:97-99
```
`T/src/sdft/data.py:26-29`: *"From `vendor/self-distillation/main.py`, verbatim, for both the tool-use and the science datasets."*

**Which one arc1-gold selects: `worked_answer`** (no thinking clause) — `T/recipes/arc1-gold.yaml:124`: `  template: worked_answer`. The arm file `T/recipes/arc1-gold-lora16.yaml:16-17` (`extends: [arc1-gold.yaml]`) overrides only `trainee`.

**THEIRS — student prompt** = the row's `prompt` field rendered alone (`T/src/sdft/data.py:164-165`). Row 1 of `T/data/arc/train_rows_gold.jsonl` (`$ python3 -c "…json.loads(open(...).readline())"`): keys `['problem_id', 'prompt', 'target']`, `prompt` = 994 chars, digit-row grids, ending:
```
Final input:
707
707
770

Reply with only JSON in this form: {"output": [[0,1],[2,3]]}
The value of "output" is the output grid as a list of rows, and each row is a list of integers from 0 to 9.
```
Grid rendering: `T/src/sdft/tasks/arc.py:72-74` `"""Digit rows, no separators. `[[1,2],[3,4]]` -> `12\n34`."""`; one item per test input (`T/src/sdft/tasks/arc.py:31-37`).

### 2. Thinking mode at render time

**OURS: thinking DISABLED, by a chat-template patch that makes non-thinking the default.**
- `O/nothinking.py:23-24`:
  ```python
  _OLD = "{%- if enable_thinking is defined and enable_thinking is false %}"
  _NEW = "{%- if enable_thinking is not defined or enable_thinking is false %}"
  ```
- Applied in the training entrypoint: `O/main.py:207-209` `tokenizer = AutoTokenizer.from_pretrained(args.model_name)` / `from nothinking import patch_tokenizer` / `patch_tokenizer(tokenizer)`; also `O/sft_main.py:97-99` and `O/merge_lora.py` (`patch_tokenizer(tok)` then `tok.save_pretrained(a.out)`).
- The resulting template on disk, `O/ckpt/base/chat_template.jinja:84-89`:
  ```jinja
  {%- if add_generation_prompt %}
      {{- '<|im_start|>assistant\n' }}
      {%- if enable_thinking is not defined or enable_thinking is false %}
          {{- '<think>\n\n</think>\n\n' }}
      {%- endif %}
  {%- endif %}
  ```
- The trainer renders both prompts with `maybe_apply_chat_template(...)` and **no** `enable_thinking` argument (`O/distil_trainer.py:1444-1446` student, `:1463-1465` teacher), so the patched default (thinking OFF, empty think block appended) applies.

⇒ **OURS has exactly the mismatch: the teacher prompt asks for "the thinking process" while the render closes the think block empty.** (Fact = the two cited code facts; the *harm* is their inference, quoted next.)

**THEIRS: thinking DISABLED explicitly at render** — `T/src/sdft/data.py:132-133`:
```python
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
```
docstring `:127-129`: *"Thinking mode is off: a template that opens a reasoning block would put the completion inside it and the tokens the loss scores would no longer be the tokens the model is judged on."*

**They document the mismatch as a bug they removed** — `T/src/sdft/data.py:60-86`:
```
#: The demonstration framing WITHOUT the request for a thinking process.
#:
#: MEASURED, NOT PREFERRED. Under `demonstration` the gold arm's completions grew
#: from 63 to 246 tokens in twelve steps and 85% of on-policy samples hit the
#: token budget, while `insight` held 61 tokens flat for 150 steps on the same
#: model, rate and budget. ...
#: WHY THE ONE SENTENCE DOES IT. `render()` sets `enable_thinking=False`, so the
#: prompt the teacher scores against ends with the think block already opened,
#: closed and empty:
#:     Now answer with a response of your own, including the thinking process.
#:     assistant
#:     <think>
#:
#:     </think>
#: The instruction cannot be satisfied where it is meant to be. Under forward KL
#: the teacher scores the STUDENT's tokens, so at the position after a complete
#: answer the teacher still has an unmet instruction, puts low mass on the end
#: token, and pulls the student away from stopping. Then it ratchets: a truncated
#: sample carries no end token anywhere, so no position teaches stopping at all.
```

### 3. What the teacher is conditioned on

| | field | content | size |
|---|---|---|---|
| OURS | `output_text` ← `g["demonstration"]` (`O/prep_arc.py:58`) | **full worked CoT solution** ending in the final JSON | 381 rows, demo chars p50 **15,369**, max **46,214** (`$ python3 -c` over `O/data/arc_data/gold/gold_knowledge.jsonl`) |
| THEIRS | `target` (`T/recipes/arc1-gold.yaml:93` `privileged_key: target`) | **answer grid JSON only** | 416 rows, target chars p50 **312.0**, max **2772** (`$ python3 -c` over `T/data/arc/train_rows_gold.jsonl`) |

- OURS row-1 demonstration head: `"To solve this ARC task, we need to analyze the transformation rules derived from the training examples…### Step 1: Analyze the Training Examples…"`, 18,259 chars, tail is a fenced `{"outputs":[[...]]}`.
- THEIRS row-1 target, verbatim start: `{"output": [[7, 0, 7, 0, 0, 0, 7, 0, 7], …` (273 chars, nothing else).
- THEIRS recipe comment `T/recipes/arc1-gold.yaml:91-93`: *"THE GOLD. The rows carry `target`, the benchmark's own output grid, in exactly the shape sdft.tasks.arc.parse_grid reads."*
- THEIRS `target_ids` (the off-policy SFT target) = `row["target"]` else the privileged field — `T/src/sdft/data.py:185-186`.
- OURS SFT arm uses the same `output_text` (gold demonstration) as the completion — `O/sft_main.py:76-80`.

### 4. Length caps and over-length handling

**OURS**
- Defaults in the runner: `O/run_arc_armA.sh:27-29`
  ```bash
  MAXPROMPT=${MAXPROMPT:-10240}   # fits question + gold demonstration for ~p95 of tasks;
                                  # keeps SFT logits (seq x 151k vocab) under the 48GB cap
  MAXCOMP=${MAXCOMP:-4096}        # student on-policy generation budget
  ```
- Argparse help, `O/main.py:39-45`: *"Truncation cap for BOTH the student prompt and the (much longer, left-truncated) teacher_prompt. For ARC set this large enough to fit question+demonstration (~12k tokens) or the teacher loses the demonstrated task grids."*
- Applied as **truncation, not dropping, to BOTH sides, left-side** — `O/distil_trainer.py:1447-1477`:
  ```python
          if self.use_vllm:
              self.processing_class.truncation_side = "left"
          student_inputs = self.processing_class(
              text=prompts_text, ... max_length=self.max_prompt_length, truncation=True, ...)
          ...
          teacher_inputs = self.processing_class(
              text=teacher_prompts_text, ... max_length=self.max_prompt_length, truncation=True, ...)
          ...
          if self.use_vllm:
              self.processing_class.truncation_side = "right"
  ```
  `use_vllm = True` is set in `O/main.py:223`.
- **Measured on the real 381-row train set with the real tokenizer** (`$ ./.venv/bin/python -c "…AutoTokenizer.from_pretrained('ckpt/base') … apply_chat_template …"`, teacher text built from the exact `O/main.py` Template):
  ```
  n rows 381
  student p50 1428 p99 10301 max 16300
  teacher p50 8627 p99 22924 max 29686
  cap 8192:  teacher over = 198 (51.97%)  student over = 7
  cap 10240: teacher over = 161 (42.26%)  student over = 4
  cap 12288: teacher over = 119 (31.23%)  student over = 3
  cap 16384: teacher over = 45 (11.81%)   student over = 0
  gold demo token p50 6897 max 17336
  ```
- No `mask_truncated_completions` is set in `O/main.py` (config block `O/main.py:221-256`); default is `False` — `O/distil_config.py:583-584` `mask_truncated_completions: bool = field(default=False, …)`. So truncated completions ARE in the loss.
- vLLM colocate rollout passes `"max_tokens": self.max_completion_length, "truncate_prompt_tokens": self.max_prompt_length` (`O/distil_trainer.py:1238-1239`); `max_model_len = self.max_prompt_length + self.max_completion_length` (`O/distil_trainer.py:565-566`).
- Rollouts come from the **student** prompt: `generation_prompts = teacher_prompts if self.generate_from_teacher else prompts` (`O/distil_trainer.py:1433`) and `generate_from_teacher` defaults `False` (`O/distil_config.py:492-493`), not set in `O/main.py`.

**THEIRS**
- `T/recipes/arc1-gold.yaml:140-141`: `  max_prompt_tokens: 12288` / `  max_dropped_frac: 0.01`, with the sizing measurement at `:127-139`:
  ```
  # 12288, MEASURED ON THE TEACHER SIDE, which is what build_pairs checks
  # (sdft/data.py:168) because it is the longer of the two. ...
  #     student  p50   884            max  8512
  #     teacher  p50  1163  p99 7779  max 11244
  #     cap  8192 -> drops 4 rows (0.96%)
  #     cap  8704 -> drops 4 rows (0.96%)     <- the student-sized guess
  #     cap 12288 -> drops 0 rows
  ```
- Over-length rows are **DROPPED, never truncated** — `T/src/sdft/data.py:168-170`:
  ```python
        if len(teacher_ids) > max_prompt_tokens:
            dropped_long += 1
            continue
  ```
  with a loud guard `T/src/sdft/data.py:192-198` (`raise ValueError(f"{dropped}/{seen} rows ({dropped / seen:.1%}) were dropped, over the …limit…")`) and docstring `:149-153`: *"The length cap is checked on the TEACHER side, which is the longer one, so a row that survives runs on both."*
- Train sampling budget: `T/recipes/arc1-gold.yaml:86` `  max_new_tokens: 3072`, comment `:80-86`: *"3072, because an ARC answer reaches 2704 tokens and 42 of the 416 targets pass 1024. A budget below the answer turns a right answer into a format failure… IT IS ALSO THE MAIN COST OF A STEP."*, `n_generations: 4` (`:79`), `temperature: 1.0` (`:87`), sampler `kind: hf` (`:78`).
- Truncated on-policy samples are also kept in the loss (no masking); the trainer only records the rate — `T/src/sdft/loop.py:152-155`.

Row counts: OURS 381 (`$ python3 …gold_knowledge.jsonl` → `n 381`); THEIRS 416 (`$ wc -l T/data/arc/train_rows_gold.jsonl` → `416`).

### 5. Documented completion-length growth / truncation-over-steps

**THEIRS — yes, a measured table**, `T/recipes/arc1-gold.yaml:95-124` (verbatim excerpt):
```
  # `worked_answer`, AND THE FIRST RUN OF THIS RECIPE IS WHY.
  #
  # It began as `demonstration`, the reference's verbatim wording... That wording asks the teacher
  # for "the thinking process" while render() disables thinking, so the prompt
  # arrives with the think block already opened, closed and empty... It ratchets.
  #
  # MEASURED HERE, on both arms, in runs/archive/2026-08-13-arc1-v1-3072:
  #
  #     steps    lora16 tokens / truncated    memory40 tokens / truncated
  #      1-5          542 / 35%                     535 / 30%
  #      6-10        1208 / 40%                     822 / 30%
  #     11-15        1803 / 55%                    2366 / 75%
  #     16-20        1337 / 40%                    1608 / 50%
  #
  # Completions tripled in fifteen steps and step time doubled with them, 110s
  # to 220s, which put the epoch at about 25 hours instead of 5. The ToolUse
  # gold arm did the same thing -- 63 to 246 tokens in twelve steps, 85%
  # truncated (runs/archive/2026-08-12-gold-v1).
  #
  # THE TWO TEMPLATES DIFFER BY ONE CLAUSE (sdft/data.py:33 against :87):
  # ", including the thinking process." That clause is the whole mechanism.
  #
  # THE COST IS COMPARABILITY... `template: demonstration` is the one-line way back.
```
Same finding restated at `T/recipes/tooluse-sdft-gold.yaml:45-56` and `T/src/sdft/data.py:60-86` (quoted in §2). Live instrumentation: `T/src/sdft/loop.py:150-155`:
```python
    stats["mean_completion_len"] = float(completion_mask.sum(-1).float().mean())
    # The share of samples the budget cut off rather than the model ending. A
    # rising number here means the loss is being computed on truncated
    # reasoning, and accuracy would fall for a reason that is not the method.
    stats["truncated_frac"] = float((completion_mask[:, -1] == 1).float().mean())
```

**OURS — no such analysis exists.** Searched two ways:
- `$ grep -rn -i "truncat" --include=*.py --include=*.sh --include=*.md .` (repo root, excluding `.venv`) → only: `O/sft_main.py:28` ("…raise this so the trailing answer JSON isn't right-truncated."), `O/eval_arc.py:32`, `O/run_arc_armA.sh:10`, TRL boilerplate in `O/distil_config.py`, plus files inside sibling worktrees.
- `$ grep -rn -i "length\|truncat\|grow" README.md docs/ mlflow_arc_logger.py` → only `mlflow_arc_logger.py:125-127` logging the cap values as params.

Metrics *are* emitted per step by the trainer (`O/distil_trainer.py:1395-1409`: `completions/mean_length`, `completions/max_length`, `completions/clipped_ratio`, `completions/mean_terminated_length`), and `is_truncated` is computed as `ids[-1] not in [eos, pad]` (`:1401`) — but nothing in the repo reads or analyses them, and `WANDB_MODE=offline` is exported by `O/run_arc_armA.sh:32`.

### 6. Eval-time budget and how a truncated-before-JSON answer scores

**OURS** (`O/eval_arc.py`)
- Budget: `O/eval_arc.py:30-33`
  ```python
    p.add_argument("--max_new_tokens", type=int, default=12288,
                   help="Generous budget: SFT imitates the long gold demonstrations (~up to 11.5k "
                        "tokens) and gets truncated before its final JSON at smaller caps. Models "
                        "that emit EOS early (base, SDFT) finish fast regardless, so this is fair.")
  ```
  `--temperature` default `0.0` (`:34`), `--max_model_len` default `20480` (`:35`), vLLM `LLM(...)` (`:166-172`). `run_arc_armA.sh:86-88` calls `eval_arc.py` with no `--max_new_tokens`, so 12288 applies.
- Scoring a truncated (unparseable) answer: `O/eval_arc.py:193-199`
  ```python
    for resp, orc in zip(responses, oracle):
        parsed = extract_outputs(resp)
        if parsed is None:
            parse_fail += 1
            scores.append(0)
        else:
            scores.append(score_task(parsed, orc))
  ```
  ⇒ **counted as WRONG (0) in `accuracy`, and separately tallied in `parse_failed`** (`:213-217`). There is a lenient recovery path first (`:83-97`, regex-reassemble rows after the last `"outputs"`), so a partially-truncated answer can still parse. **No truncation/finish-reason flag is recorded** — `$ grep -n "finish_reason\|truncat\|stop_reason" eval_arc.py` returned only the line-32 help text; the saved record is `{"task_id", "response", "correct"}` (`:224-227`).
- Metric: strict per-task all-or-nothing over the 400-task ARC evaluation split (`O/eval_arc.py:1-9`, `score_task` `:122-127`), prompts from `data/arc_data/eval_data` built by `O/prep_arc.py:64-82`.

**THEIRS**
- Budget: `T/recipes/arc1-gold.yaml:229` `  max_new_tokens: 3072`, `:230` `  temperature: 0.0      # greedy: the curve is what is read`, `:233` `  batch_size: 4`, `:190` `  limit: 100`, `:195` `  every: 0`. The 3072-vs-6144 re-measurement is recorded verbatim at `:196-228`:
  ```
  # 3072. IT WAS RAISED TO 6144 AND THE MEASUREMENT SENT IT BACK.
  # The argument for raising it was that 3072 clipped real reasoning: on the
  # frozen base, 8 of 100 items hit the cap and none of the 8 emitted a
  # parseable grid, so they scored as format failures rather than as wrong
  # answers. That looked like the budget hiding capability.
  # IT WAS NOT. Re-measured at 6144, greedily, on the same frozen model:
  #     truncated at 3072    8 of 100      at 6144    8 of 100
  #     the SAME 8 items, and all 8 ran the full 6144 tokens
  #     recovered by doubling the budget   0
  #     the other 92 completions           byte-identical (greedy is exact)
  #     mean_tokens          799.76 -> 1045.52, which is 8 x 3072 / 100 exactly
  ```
- Scoring: `T/src/sdft/tasks/arc.py:142-164` `grade()` → `Outcome(correct=format_valid and not feedback, format_valid=format_valid, …)`; a non-parsing completion yields `feedback = [{"type": "format", …}]` and `correct=False`. **Format failure is separated from wrong answer** by design — `T/src/sdft/tasks/arc.py:39-45`: *"a completion that emits a plausible grid wrapped in an explanation is a DIFFERENT failure from one that emits the wrong grid, and `format_valid` keeps them apart."*
- Truncation is recorded **per sequence**: `T/src/sdft/evaluate.py:120` `"tokens": len(ids), "truncated": not stopped,` with `_trim` (`:147-158`) returning `stopped=False` when no terminator was emitted; summary keys `{part}/accuracy`, `{part}/format_valid`, `{part}/truncated_frac`, `{part}/mean_tokens` (`T/src/sdft/evaluate.py:164-169`).

---

## Inferences (labeled — not directly asserted by the code)

1. **Our run reproduces the exact failure mode theirs measured and removed.** Facts: our teacher template ends with the thinking clause (`O/main.py:159`), and our render emits a closed empty think block (`O/ckpt/base/chat_template.jinja:86-88` + `O/distil_trainer.py:1463-1465` passing no `enable_thinking`). Their `T/src/sdft/data.py:72-84` describes precisely this configuration and attributes a 3× length growth + 85% truncation to it. **Inference:** our SDFT arm should show rising `completions/mean_length` and `completions/clipped_ratio`. Not verified on our runs — I did not read any of our training logs.
2. **Our 42.3% left-truncation removes the QUESTION, not the demonstration.** Facts: `truncation_side = "left"` (`O/distil_trainer.py:1448`) keeps the *last* `max_length` tokens; our teacher text order is question → "This is an example…" → demonstration → "Now answer…". **Inference:** for the 161 over-cap rows the teacher's context begins mid-demonstration with the ARC task grids gone — the opposite of what `O/run_arc_armA.sh:10` intends ("or the SDFT teacher loses the demonstrated task grids"). This makes the teacher a next-token continuation of a headless CoT for ~42% of rows, and the student/teacher then disagree about what problem is being solved.
3. **Their cap is effectively lossless; ours is a silent 42% corruption.** Their design drops rows and raises if >1% are lost (`T/src/sdft/data.py:192-198`); ours truncates with no counter, no log line, and no test.
4. **Length pressure is structurally larger on our side even before the thinking clause**: our teacher is conditioned on a 6,897-token-median CoT and our SFT arm imitates that same CoT, whereas theirs conditions on a ~312-char grid. A forward-KL teacher holding a long CoT in context plausibly favours long student completions.
5. **Our eval budget (12288) hides length blow-up rather than exposing it**: with no `truncated`/finish-reason recorded (`O/eval_arc.py:224-227`) a model that has learned not to stop shows up only as `parse_failed`, indistinguishable from a badly-formatted short answer.

---

## Unknowns / caveats

- **I did not verify runtime behaviour of either training loop** — no logs, no MLflow queries, no runs. All length-growth claims about OUR run are inferences.
- **`T/runs/` does not exist in this snapshot** (`$ ls runs` → `No such file or directory`), so the archived measurements quoted from `arc1-gold.yaml` / `data.py` are *their comments*, not artifacts I could re-verify.
- **vLLM's `truncate_prompt_tokens` slicing side is unverified.** `O/distil_trainer.py:1238-1239` passes it, but `$ grep -rn "truncate_prompt_tokens" .venv/…/vllm/inputs/ .venv/…/vllm/v1/` found only plumbing in `v1/engine/async_llm.py` and `sampling_params.py`, not the slice. I don't know whether it keeps head or tail. It affects only rollout prompts (4/381 student prompts over 10240), not the teacher-side truncation, which is the HF-tokenizer path I did verify.
- **Version pinning:** vLLM read from `O/.venv/lib/python3.12/site-packages/vllm` (exact version not queried). Token measurements used the tokenizer at `O/ckpt/base` (patched Qwen3-8B); their p50/max figures are quoted from their comments, measured with a Qwen3.5-9B tokenizer — the two token-count columns are **not** directly comparable.
- **Checkout note:** I read `/home/xwang3/Projects/Self-Distillation/*` (main checkout). The current git worktree `.claude/worktrees/cheeky-snacking-duckling` has `eval_arc.py`, `run_arc_armA.sh`, `prep_arc.py`, `nothinking.py`, `distil_trainer.py` **byte-identical** (`$ cmp -s` → SAME); `main.py` differs by exactly one hunk (`$ diff` → only the vLLM `Qwen3_5ForCausalLM` `ModelRegistry.register_model` block at lines 179-191). **The teacher Template and the DistilConfig block are identical in both.**
- I did not audit `T/src/sdft/loss.py`, `teacher.py`, or our loss internals — the brief was data/prompt/length only.

---

## Methods (commands run)

```
ls -la /home/xwang3/Projects/Self-Distillation/
ls -la <THEIRS>/ ; find <THEIRS>/src -type f ; ls -la <THEIRS>/recipes
Read: O/main.py, O/prep_arc.py, O/nothinking.py, O/run_arc_armA.sh, O/eval_arc.py, O/sft_main.py
Read: O/distil_trainer.py (1325-1373, 1378-1417, 1415-1514)
Read: T/src/sdft/data.py, T/recipes/arc1-gold.yaml, T/src/sdft/tasks/arc.py,
      T/src/sdft/evaluate.py (80-179), T/src/sdft/loop.py (100-179, 350-375),
      T/scripts/train_sdft.py (255-294)
cd O && grep -n "max_prompt_length|truncation_side|max_completion_length|teacher_prompt" distil_trainer.py
cd O && grep -rn -i "truncat" --include=*.py --include=*.sh --include=*.md .
cd O && grep -rn -i "length|truncat|grow" README.md docs/ mlflow_arc_logger.py
cd O && grep -n "mask_truncated_completions" -A6 distil_config.py
cd O && grep -n "generate_from_teacher" distil_config.py distil_trainer.py
cd O && grep -n "finish_reason|truncat|stop_reason" eval_arc.py
cd O && grep -n "enable_thinking" -A4 -B2 ckpt/base/chat_template.jinja
cd O && ./.venv/bin/python -c "<tokenize 381 rows: student vs teacher chat-templated lengths, cap sweep>"
cd O && python3 -c "<demonstration char stats over gold_knowledge.jsonl>"
cd T && python3 -c "<print row keys/prompt/target of train_rows_gold.jsonl line 1>"
cd T && wc -l data/arc/train_rows_gold.jsonl ; python3 -c "<target char stats>"
cd T && grep -n -i "thinking process" DECISIONS.md src/sdft/*.py recipes/*.yaml
cd T && cat recipes/arc1-gold-lora16.yaml ; sed -n 40,75p recipes/tooluse-sdft-gold.yaml ; ls runs
cd T && grep -n "template|max_prompt_tokens|privileged_key|max_new_tokens|max_dropped" src/sdft/recipe.py
cd T && grep -rln "build_pairs" . --include=*.py
for f in main.py eval_arc.py run_arc_armA.sh prep_arc.py nothinking.py distil_trainer.py; do cmp -s <main> <worktree>; done ; diff main.py <worktree>/main.py
```

---

## Prompt/length differences that would change the LOSS/LENGTH CURVE (ranked)

1. **The `, including the thinking process.` clause combined with thinking disabled at render.** OURS has it (`O/main.py:159` + `O/ckpt/base/chat_template.jinja:86-88`); THEIRS removed it (`T/recipes/arc1-gold.yaml:124` → `worked_answer`, `T/src/sdft/data.py:87-93`) after measuring completions tripling in 15 steps and 35→55% truncation (`T/recipes/arc1-gold.yaml:105-116`). This is a one-line, forward-KL-mediated pressure against emitting EOS — the single highest-leverage difference. **Fix = change our Template to drop that clause (or pass `enable_thinking=True`), not both.**
2. **Teacher conditioning: 6,897-token-median worked CoT (ours) vs ~312-char answer grid (theirs).** `O/prep_arc.py:58` (`output_text = g["demonstration"]`, p50 15,369 chars) vs `T/recipes/arc1-gold.yaml:93` (`privileged_key: target`, p50 312 chars). A teacher holding a long CoT scores long student continuations as likely; a teacher holding only the grid does not.
3. **Silent left-truncation of 42.3% of teacher prompts at cap 10240** (`O/distil_trainer.py:1448-1474`; measured 161/381) — the teacher loses the question for those rows, so its distribution over the student's tokens is conditioned on the wrong context. THEIRS drops instead, sized for 0 drops (`T/recipes/arc1-gold.yaml:127-141`, `T/src/sdft/data.py:168-170`). Any loss curve of ours mixes two teacher regimes.
4. **Train generation budget 4096 (ours, `O/run_arc_armA.sh:29`) vs 3072 (theirs, `T/recipes/arc1-gold.yaml:86`), with truncated completions unmasked on both sides** (`O/distil_config.py:583-584` default `False`; `T/src/sdft/loop.py:150-155`). A larger budget defers the clip but makes each step cost more and lets length grow further before the ratchet is visible.
5. **`num_generations = 1` (ours, `O/main.py:245`) vs `n_generations: 4` (theirs, `T/recipes/arc1-gold.yaml:79`)** and effective batch 32 prompts/step (`O/run_arc_armA.sh:31`) vs 1 prompt × 4 samples/step (`T/recipes/arc1-gold.yaml:153-155`). Different gradient noise and a much coarser length curve on our side (points per epoch), independent of the mechanism above.
6. **Eval budget/protocol: 12288 greedy, per-task all-or-nothing over 400 tasks, truncation not recorded** (`O/eval_arc.py:30-33`, `:193-199`, `:224-227`) vs **3072 greedy, per-item over 418, `format_valid` + `truncated_frac` recorded, limit 100** (`T/recipes/arc1-gold.yaml:171-233`, `T/src/sdft/evaluate.py:120,164-169`). Ours cannot distinguish "learned not to stop" from "answered wrongly"; theirs can, and their headroom re-measurement showed 0 items recovered by doubling the budget.
7. **Prompt encoding itself: compact JSON grids with all test inputs in one prompt (ours, `O/prep_arc.py:36-39`) vs digit-row grids, one item per test input (theirs, `T/src/sdft/tasks/arc.py:72-74, 91-109`).** Student prompt p50 1428 tokens (ours, measured) vs 884 (theirs, their comment) and the answer format differs (`{"outputs": [grids]}` vs `{"output": [[…]]}`) — so completion-length numbers are not comparable across the two repos even after fixing 1-3.