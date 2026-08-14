"""Materialize a text-only Qwen3.5-9B checkpoint (vision tower dropped).

Qwen/Qwen3.5-9B ships as a VLM (Qwen3_5ForConditionalGeneration). AutoModelForCausalLM already
instantiates the text-only Qwen3_5ForCausalLM (vision weights are simply unused), so we re-save it
with the promoted *text* config and architectures=['Qwen3_5ForCausalLM']. Both transformers and
vLLM (after registering the class) then load it as a plain causal LM — no vision memory, matching
param names for the SDFT weight-sync.
"""
import sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from nothinking import patch_tokenizer

SRC = "Qwen/Qwen3.5-9B"
OUT = "ckpt/qwen35_9b_text_v2"

comp = AutoConfig.from_pretrained(SRC)
text_cfg = comp.text_config
text_cfg.architectures = ["Qwen3_5ForCausalLM"]

print("loading (text weights only; vision unused)...")
import torch
m = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16)
print("loaded class:", type(m).__name__)
names = [n for n, _ in m.named_parameters()]
print("param name sample:", names[:3])
print("any vision params in text model:", any(n.startswith("visual") for n in names))

m.config = text_cfg
m.save_pretrained(OUT, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(SRC)
patch_tokenizer(tok)
tok.save_pretrained(OUT)

# transformers serializes the text weights in the VLM-canonical layout (`model.language_model.*`)
# and remaps them back on load. vLLM (native Qwen3_5ForCausalLM, >=0.27) instead expects the bare
# text layout `model.*`, so re-key the saved safetensors to strip the `.language_model` nesting.
# Both transformers and vLLM then load this checkpoint cleanly.
import glob
from safetensors.torch import load_file, save_file
for shard in glob.glob(os.path.join(OUT, "*.safetensors")):
    sd = load_file(shard)
    if any("model.language_model." in k for k in sd):
        rekeyed = {k.replace("model.language_model.", "model."): v for k, v in sd.items()}
        save_file(rekeyed, shard, metadata={"format": "pt"})
index_path = os.path.join(OUT, "model.safetensors.index.json")
if os.path.exists(index_path):  # multi-shard: re-key the weight_map too
    with open(index_path) as f:
        idx = json.load(f)
    idx["weight_map"] = {k.replace("model.language_model.", "model."): v
                         for k, v in idx["weight_map"].items()}
    with open(index_path, "w") as f:
        json.dump(idx, f, indent=2)
print("saved text-only checkpoint ->", OUT)
# report saved config arch
saved = AutoConfig.from_pretrained(OUT)
print("saved config: model_type=%s architectures=%s layers=%s" % (
    saved.model_type, saved.architectures, getattr(saved, "num_hidden_layers", "?")))
