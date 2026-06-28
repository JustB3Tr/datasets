#!/usr/bin/env python3
"""
Merge a saved LoRA adapter back into the base model and export a standalone model.

Usage:
    python merge_adapter.py \
        --adapter ./output/final_adapter \
        --base Qwen/Qwen2.5-VL-3B-Instruct \
        --output ./output/merged_model
"""

import argparse
import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to saved LoRA adapter directory")
    parser.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model repo or path")
    parser.add_argument("--output", required=True, help="Where to save the merged model")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading base model: {args.base}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter weights ...")
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving merged model to: {args.output}")
    model.save_pretrained(args.output, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.adapter, trust_remote_code=True)
    processor.save_pretrained(args.output)

    print("Done! Merged model saved.")


if __name__ == "__main__":
    main()
