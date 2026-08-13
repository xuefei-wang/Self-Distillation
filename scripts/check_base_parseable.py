"""Probe: does base Qwen/Qwen3-8B produce outputs the repo's eval extractors can parse,
and which `enable_thinking` chat-template setting should evals use?

Loads the vLLM engine once and reuses it across both enable_thinking settings and
both tasks (science, tooluse) to save time.

Run:
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 uv run python scripts/check_base_parseable.py
"""
import os
import re
import sys

# repo root (parent of this scripts/ dir) so `eval_science` is importable
# regardless of the invocation cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# reuse the repo's own extractors so the probe matches eval exactly
from eval_science import extract_xml_answer


def probe(model="Qwen/Qwen3-8B", n=8):
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.6,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
    )

    sci = Dataset.load_from_disk("data/science_data/eval_data").select(range(n))
    tu = Dataset.load_from_disk("data/tooluse_data/eval_data").select(range(n))

    sp = SamplingParams(temperature=0.0, max_tokens=2048)

    results = {}
    for thinking in (True, False):
        # science: prompt is already a list of chat messages
        sci_prompts = [
            tok.apply_chat_template(
                e["prompt"], tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking,
            )
            for e in sci
        ]
        sci_out = [o.outputs[0].text for o in llm.generate(sci_prompts, sp)]
        sci_ok = sum(1 for o in sci_out if extract_xml_answer(o) != "")
        sci_exact = sum(
            1 for o, e in zip(sci_out, sci)
            if extract_xml_answer(o) == e["answer"]
        )
        print(f"enable_thinking={thinking}: science parseable {sci_ok}/{n} (exact-match {sci_exact}/{n})")

        # tooluse: prompt is a plain string
        tu_prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": e["prompt"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking,
            )
            for e in tu
        ]
        tu_out = [o.outputs[0].text for o in llm.generate(tu_prompts, sp)]
        tu_ok = sum(1 for o in tu_out if re.search(r"Action Input:\s*\{", o))
        print(f"enable_thinking={thinking}: tooluse action-input present {tu_ok}/{n}")

        results[thinking] = {"sci_ok": sci_ok, "sci_exact": sci_exact, "tu_ok": tu_ok}

    # recommendation: higher parseability for both tasks; tie -> prefer non-thinking
    def score(t):
        r = results[t]
        return r["sci_ok"] + r["tu_ok"]

    best = max(results, key=score)
    if score(True) == score(False):
        best = False  # tie -> prefer paper-like non-thinking behavior
    print(f"RECOMMENDATION: enable_thinking={best} "
          f"(science {results[best]['sci_ok']}/{n}, tooluse {results[best]['tu_ok']}/{n})")


if __name__ == "__main__":
    probe()
