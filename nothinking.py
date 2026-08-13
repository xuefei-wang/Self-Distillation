"""Force Qwen3 non-thinking mode by DEFAULT via a one-line chat-template patch.

Why: the paper used Qwen2.5-7B-Instruct, which has no thinking mode. We use Qwen3-8B,
whose chat template only suppresses the native <think> block when `enable_thinking` is
*explicitly* passed as False. The unedited eval scripts (eval_science.py, eval_tooluse.py)
and the TRL trainer call `apply_chat_template` WITHOUT that arg, so they would default to
thinking ON — inconsistent between training rollouts, the teacher, and eval, and empirically
worse (probe: science exact-match 1/8 thinking-on vs 5/8 thinking-off).

Fix: patch the template's guard so the empty `<think></think>` block is emitted whenever
`enable_thinking` is not explicitly True. This makes non-thinking the default everywhere the
tokenizer is used, while still allowing an explicit enable_thinking=True to override.

Apply `patch_tokenizer(tok)` after loading a tokenizer in every training/merge script; the
patched template is then saved with the tokenizer so unedited eval scripts inherit it.
Run `python nothinking.py --out ckpt/base` to materialize a base-model eval dir (weights
symlinked from the HF cache + patched tokenizer) so base-model eval is also non-thinking.
"""
import argparse
import os

# Qwen3 chat-template guard: original only fires on explicit `enable_thinking is false`.
_OLD = "{%- if enable_thinking is defined and enable_thinking is false %}"
_NEW = "{%- if enable_thinking is not defined or enable_thinking is false %}"


def patch_tokenizer(tokenizer):
    """Set non-thinking as the default in the tokenizer's chat_template. Idempotent.

    Raises if the expected guard is absent (so a future template change fails loudly
    instead of silently leaving thinking on)."""
    ct = tokenizer.chat_template
    if ct is None:
        raise ValueError("tokenizer has no chat_template to patch")
    if _NEW in ct:
        return tokenizer  # already patched
    if _OLD not in ct:
        raise ValueError(
            "expected Qwen3 enable_thinking guard not found in chat_template; "
            "template may have changed — re-inspect before trusting non-thinking default"
        )
    tokenizer.chat_template = ct.replace(_OLD, _NEW)
    return tokenizer


def _make_base_dir(model_name, out_dir):
    """Materialize an eval dir: symlink the base weights from the HF cache + write a
    patched tokenizer, so eval of the base model is also non-thinking without a 16GB copy."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    src = snapshot_download(model_name)
    os.makedirs(out_dir, exist_ok=True)
    # symlink weight/config shards; tokenizer files are overwritten below with the patched ones
    tok_files = {
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    }
    for fn in os.listdir(src):
        if fn in tok_files:
            continue
        dst = os.path.join(out_dir, fn)
        if not os.path.exists(dst):
            os.symlink(os.path.join(src, fn), dst)
    tok = patch_tokenizer(AutoTokenizer.from_pretrained(model_name))
    tok.save_pretrained(out_dir)  # writes patched chat_template into out_dir
    print(f"Base eval dir written to {out_dir} (weights symlinked, tokenizer patched non-thinking)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-8B")
    p.add_argument("--out", required=True, help="Output base-model eval dir")
    a = p.parse_args()
    _make_base_dir(a.model_name, a.out)
