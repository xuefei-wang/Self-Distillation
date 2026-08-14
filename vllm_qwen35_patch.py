"""Register a text-only Qwen3.5 causal-LM architecture with vLLM.

Qwen3.5-9B ships as a VLM (Qwen3_5ForConditionalGeneration). A text-only checkpoint
(built by build_text_ckpt.py) is still serialized by transformers in the VLM-canonical
layout — on-disk safetensors keys are `model.language_model.*` — and transformers remaps
them to `model.*` when loading the text `Qwen3_5ForCausalLM`.

vLLM gained a working NATIVE text `Qwen3_5ForCausalLM` (weight mapper + M-RoPE + hybrid
mamba KV-cache) in >=0.27 / PR #50210. On that stack — which is the pinned one now
(vLLM 0.27.1) — `register()` sees the arch already supported and early-returns, so the
native class handles the load and this subclass never runs. The subclass is kept only as a
FALLBACK for older stacks (<0.27), whose stock text class was a `pass` stub with no
`hf_to_vllm_mapper` and so could not load `model.language_model.*` off disk. It adds the
prefix mapper `model.language_model.` -> `model.`, fixing both the initial checkpoint load and
the on-policy weight-sync (the sync sends already-clean `model.*` names, a no-op for the
mapper). Registered by string path so it re-imports in the colocate vLLM worker subprocess.
No-op if vLLM lacks the base class.

Note: `rekey_text_checkpoint` below is likewise inert on transformers 5.14 (the merge/save
already writes bare `model.*` keys); it is retained as a safety net for layouts that still
serialize the nested `model.language_model.*` form.

See vLLM #36275 / TRL #5269 for the upstream issue.
"""

try:
    import torch
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM as _Qwen35TextBase
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
    from vllm.model_executor.models.interfaces import SupportsMRoPE

    class Qwen3_5TextForCausalLM(_Qwen35TextBase, SupportsMRoPE):
        # Strip the VLM `language_model` nesting so weights land on vLLM's bare Qwen3_5Model
        # (`layers.*`, `embed_tokens`, `norm`). `lm_head.weight` is left untouched.
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "model."}
        )
        supports_mrope = True

        def load_weights(self, weights):
            loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
            return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        def get_mrope_input_positions(self, input_tokens, mm_features):
            # Qwen3.5 uses interleaved M-RoPE even for text; vLLM 0.25.1's stock text class
            # doesn't implement the interface. This pipeline is ALWAYS text (no image tokens),
            # so the 3 (T/H/W) position rows are just sequential token positions and the delta
            # is 0 — identical to what the VLM path computes for a purely-text sequence.
            length = len(input_tokens)
            positions = torch.arange(length, dtype=torch.long).unsqueeze(0).expand(3, -1).contiguous()
            return positions, 0

except Exception:  # vLLM missing / older stack without the qwen3_5 text base
    Qwen3_5TextForCausalLM = None


def rekey_text_checkpoint(out_dir):
    """Strip the VLM-canonical `model.language_model.` nesting from a saved checkpoint's
    safetensors so vLLM's native text Qwen3_5ForCausalLM loads it (transformers writes the
    nested layout even for the text model). No-op for non-Qwen3.5 checkpoints (no such keys)."""
    import glob
    import json
    import os
    from safetensors.torch import load_file, save_file
    changed = False
    for shard in glob.glob(os.path.join(out_dir, "*.safetensors")):
        sd = load_file(shard)
        if any("model.language_model." in k for k in sd):
            save_file({k.replace("model.language_model.", "model."): v for k, v in sd.items()},
                      shard, metadata={"format": "pt"})
            changed = True
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    if changed and os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        idx["weight_map"] = {k.replace("model.language_model.", "model."): v
                             for k, v in idx["weight_map"].items()}
        with open(index_path, "w") as f:
            json.dump(idx, f, indent=2)
    return changed


def register():
    """Register the text arch with vLLM if available and not already present. Idempotent."""
    try:
        from vllm import ModelRegistry
        if Qwen3_5TextForCausalLM is None:
            return
        if "Qwen3_5ForCausalLM" in ModelRegistry.get_supported_archs():
            return
        ModelRegistry.register_model(
            "Qwen3_5ForCausalLM",
            "vllm_qwen35_patch:Qwen3_5TextForCausalLM",
        )
    except Exception:
        pass
