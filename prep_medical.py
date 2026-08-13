"""Build the Medical Skill-Learning task from the public HuatuoGPT-o1 datasets, in the same
on-disk format as this repo's science/tooluse tasks.

- TRAIN  = FreedomIntelligence/medical-o1-reasoning-SFT (stage 1): Question, Complex_CoT, Response
           -> messages=[system,user(Question)] + output_text="<reasoning>{CoT}</reasoning><answer>{Response}</answer>"
- EVAL   = FreedomIntelligence/medical-o1-verifiable-problem (stage 2): open-ended question + short ground-truth
           -> prompt=[system,user(Question)] + answer=Ground-True Answer

Writes data/medical_data/{train_data,eval_data}. Grading (in eval_medical.py) is normalized
containment of the short ground-truth answer inside the model's <answer> — no LLM judge / API key.
"""
import argparse
from datasets import load_dataset

# Shared system prompt: reasoning + a CONCISE final answer, mirroring the science task's structure
# so the same <answer> extraction works and short verifiable answers are gradable by containment.
MED_SYS = (
    "You are a medical expert answering clinical reasoning questions. "
    "Think through the problem, then give your final answer. Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n"
    "In <answer>, state the single most likely answer as concisely as possible "
    "(a diagnosis, term, or short phrase), and nothing else."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=2674, help="match science train size for comparable compute")
    p.add_argument("--n_eval", type=int, default=507, help="match science eval size")
    p.add_argument("--max_answer_chars", type=int, default=100,
                   help="drop eval items whose ground-truth answer is longer than this — containment "
                        "grading is only meaningful for short verifiable answers (~97%% are <=100 chars)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    # TRAIN
    tr = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    tr = tr.shuffle(seed=a.seed).select(range(min(a.n_train, len(tr))))

    def fmt_train(e):
        out = f"<reasoning>\n{e['Complex_CoT'].strip()}\n</reasoning>\n<answer>\n{e['Response'].strip()}\n</answer>"
        return {
            "messages": [
                {"role": "system", "content": MED_SYS},
                {"role": "user", "content": e["Question"].strip()},
            ],
            "output_text": out,
        }

    tr = tr.map(fmt_train, remove_columns=tr.column_names)
    tr.save_to_disk("data/medical_data/train_data")
    print(f"train: {len(tr)} examples -> data/medical_data/train_data")
    print("  sample output_text[:200]:", repr(tr[0]["output_text"][:200]))

    # EVAL (different question set -> naturally disjoint from train)
    ev = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem", split="train")
    ev = ev.filter(lambda e: len(e["Ground-True Answer"].strip()) <= a.max_answer_chars)
    ev = ev.shuffle(seed=a.seed).select(range(min(a.n_eval, len(ev))))

    def fmt_eval(e):
        return {
            "prompt": [
                {"role": "system", "content": MED_SYS},
                {"role": "user", "content": e["Open-ended Verifiable Question"].strip()},
            ],
            "answer": e["Ground-True Answer"].strip(),
        }

    ev = ev.map(fmt_eval, remove_columns=ev.column_names)
    ev.save_to_disk("data/medical_data/eval_data")
    print(f"eval: {len(ev)} examples -> data/medical_data/eval_data")
    print("  sample answer:", repr(ev[0]["answer"]))


if __name__ == "__main__":
    main()
