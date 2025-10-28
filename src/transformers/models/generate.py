# gen_local.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import transformers.models.powen3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Path to local model repo (with config/weights/code)")
    ap.add_argument("-p", "--prompt", default="Hello!", help="User prompt")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--device-map", default="auto", help='"auto" uses accelerate; or "cpu", "cuda", "mps", etc.')
    args = ap.parse_args()

    dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    tok = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,   # enables custom tokenizer code in repo
        # local_files_only=True,
    )
    # Avoid pad-token warnings
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # Load model + custom modeling code
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,   # loads local modeling_*.py
        # local_files_only=True,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    model.eval()

    text = args.prompt
    inputs = tok(text, return_tensors="pt")
    # Send tensors to the same device as the model's first parameter
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Print only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, prompt_len:]
    print(tok.decode(generated, skip_special_tokens=True))

if __name__ == "__main__":
    main()
