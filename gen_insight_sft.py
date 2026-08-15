"""Generate NON-THINKING insight-conditioned rollouts as the SFT-insight training target.

The shipped `best_rollout` (arc-train93-split/insight_knowledge.jsonl) was produced with
thinking ON (it was the ICL insight-verification rollout): ~24k tokens median, 85/93 carry a
<think> block. The repo trains/evals thinking OFF, so those cannot be the SFT target. This
regenerates, per the user's protocol:

  prompt  = system + [ repo render_question  +  "a useful insight: <knowledge_text>"  + footer ]
  gen     = Qwen3.5-9B (the text base the students init from), enable_thinking=FALSE,
            best-of-8, T=0.7 top-p 0.95, 4096-token budget
  select  = shortest CORRECT (exact-match every oracle test grid); else shortest that stopped;
            else shortest overall  (last-attempt-regardless: a task always yields a row)

The SFT-insight TRAINING row is question-only (the insight is a generation-time crutch the
student won't have at eval): messages=[system, user=render_question], output_text=<rollout>.
Byte-identical question rendering to the gold cell (prep_arc.render_question) + the same system
message, so gold vs insight differ ONLY in the knowledge source, not prompt/thinking/format.
"""
import argparse
import json
import os
import sys

from datasets import Dataset

# repo prompt rendering (matches golden_knowledge.jsonl `question` byte-for-byte) + system msg
from prep_arc import render_question, load_task, SYSTEM
# validated insight-injection strings + parser/scorer from the insight pipeline
sys.path.insert(0, os.path.expanduser("~/insight-knowledge-arc"))
import pipeline as P  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--insight_jsonl", default=os.path.expanduser("~/arc-train93-split/insight_knowledge.jsonl"))
    p.add_argument("--data_root", default="data/arc_agi_1")
    p.add_argument("--model", default="ckpt/qwen35_9b_text_v2")
    p.add_argument("--out", default="data/arc_data/sft_insight_data", help="HF dataset dir for SFT")
    p.add_argument("--audit_out", default="data/arc_data/sft_insight_audit.jsonl")
    p.add_argument("--n", type=int, default=8, help="samples per task (best-of-n)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--max_model_len", type=int, default=12288)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


DERIVE_MIN = 200  # chars of text before the answer object to count a rollout as "deriving"


def _derivation_chars(text):
    """Chars before the first {"outputs" object — how much the rollout reasons on paper."""
    i = text.find('{"outputs"')
    return len(text) if i == -1 else i


def select_rollout(samples, oracle):
    """samples: list of (text, finish_reason). Return (chosen_text, chosen_correct, finish, n_correct).

    Reasoning-first (the corpus's own selection rule): among candidates prefer ones that actually
    DERIVE, then the shortest (student-affordable). Priority: correct+deriving -> correct ->
    deriving -> shortest overall. A task always yields exactly one row (last-attempt-regardless).
    Bare-answer rollouts are the LAST resort — training on them reproduces the answer-only collapse."""
    scored = []
    for text, finish in samples:
        parsed = P.parse_outputs(text)
        ok = bool(parsed is not None and P.exact_match(parsed, oracle)[0])
        scored.append({"len": len(text), "ok": ok, "finish": finish, "text": text,
                       "derives": _derivation_chars(text) >= DERIVE_MIN})
    n_correct = sum(s["ok"] for s in scored)

    def shortest(cands):
        return sorted(cands, key=lambda s: s["len"])[0] if cands else None

    for cands in ([s for s in scored if s["ok"] and s["derives"]],
                  [s for s in scored if s["ok"]],
                  [s for s in scored if s["derives"]],
                  scored):
        p = shortest(cands)
        if p:
            return p["text"], p["ok"], p["finish"], n_correct


def main():
    args = parse_args()
    rows_in = [json.loads(l) for l in open(args.insight_jsonl)]
    print(f"{len(rows_in)} insight rows from {args.insight_jsonl}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=False, trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    stop_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"),
                            tok.convert_tokens_to_ids("<|endoftext|>")) if i is not None and i >= 0]
    print(f"stop_token_ids={stop_ids}")

    def render(messages):
        try:
            return tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompts, meta = [], []
    for r in rows_in:
        tid = r["task_id"]
        task = load_task(args.data_root, "training", tid)
        oracle = [t["output"] for t in task["test"]]
        question = render_question(task)
        # Elicit a WRITTEN derivation (like the 27B gold), not a bare answer: the pipeline's
        # verify footer ("...return one JSON object and nothing else") suppresses reasoning, which
        # in non-thinking mode collapses to answer-only rollouts. Ask it to reason THEN answer.
        hint_user = (question + "\n\nA useful insight for solving this task:\n" + r["knowledge_text"]
                     + "\n\nUsing the insight, reason through the transformation step by step, "
                     'then give your final answer as one JSON object {"outputs":[<grid for test 0>, ...]}.')
        gen_msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": hint_user}]
        text = render(gen_msgs)
        plen = len(tok(text, add_special_tokens=False).input_ids)
        budget = min(args.max_tokens, args.max_model_len - plen - 64)
        if budget < 256:
            print(f"[skip] {tid}: prompt {plen} tok leaves budget {budget}")
            meta.append({"task_id": tid, "oracle": oracle, "question": question,
                         "has_good_insight": r.get("has_good_insight"), "skip": True})
            continue
        prompts.append((text, budget))
        meta.append({"task_id": tid, "oracle": oracle, "question": question,
                     "has_good_insight": r.get("has_good_insight"), "skip": False,
                     "prompt_idx": len(prompts) - 1, "budget": budget})

    # one SamplingParams per prompt (budgets differ), n samples each
    sps = [SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                          max_tokens=b, seed=args.seed, stop_token_ids=stop_ids or None)
           for _, b in prompts]
    print(f"generating {len(prompts)} tasks x n={args.n} (non-thinking, T={args.temperature})...")
    outs = llm.generate([t for t, _ in prompts], sps)

    ds_rows, audit, n_ok = [], [], 0
    for m in meta:
        tid = m["task_id"]
        if m["skip"]:
            audit.append({"task_id": tid, "chosen_correct": False, "n_correct": 0,
                          "finish": "skipped", "n_samples": 0})
            # still emit a training row? no target -> drop from training set
            continue
        o = outs[m["prompt_idx"]]
        samples = [(c.text, c.finish_reason) for c in o.outputs]
        chosen, ok, finish, n_correct = select_rollout(samples, m["oracle"])
        n_ok += int(ok)
        has_think = ("<think>" in chosen or "</think>" in chosen)
        ds_rows.append({
            # Training row is nosys by default (matches prep_arc._msgs); SYSTEM stays only as the
            # generation-time crutch above, not in the trained prompt.
            "messages": [{"role": "user", "content": m["question"]}],
            "output_text": chosen,
            "task_id": tid,
        })
        audit.append({"task_id": tid, "chosen_correct": ok, "n_correct": n_correct,
                      "finish": finish, "n_samples": len(samples),
                      "chosen_chars": len(chosen), "has_think": has_think,
                      "chosen_derives": _derivation_chars(chosen) >= DERIVE_MIN,
                      "has_good_insight": m["has_good_insight"]})

    Dataset.from_list(ds_rows).save_to_disk(args.out)
    with open(args.audit_out, "w") as f:
        for a in audit:
            f.write(json.dumps(a) + "\n")
    print("=" * 60)
    print(f"wrote {len(ds_rows)} SFT-insight rows -> {args.out}")
    print(f"correct rollout (insight helped): {n_ok}/{len(ds_rows)}")
    print(f"with <think> leak: {sum(a.get('has_think') for a in audit)} (should be ~0 non-thinking)")
    print(f"audit -> {args.audit_out}")


if __name__ == "__main__":
    main()
