import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from nothinking import patch_tokenizer

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
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.base)
    patch_tokenizer(tok)
    tok.save_pretrained(a.out)
    # Qwen3.5 text checkpoints serialize with the VLM-canonical `model.language_model.*` layout;
    # re-key to bare `model.*` so vLLM's native text arch can load the merged model at eval.
    # No-op for other models (no such keys).
    from vllm_qwen35_patch import rekey_text_checkpoint
    if rekey_text_checkpoint(a.out):
        print("re-keyed merged safetensors (model.language_model.* -> model.*)")
    print(f"Merged model saved to {a.out}")

if __name__ == "__main__":
    main()
