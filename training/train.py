#!/usr/bin/env python3
"""
Interactive SFT LoRA/QLoRA training script for Qwen2.5-VL-3B.
Run:  python train.py
      python train.py --config my_config.json   (skip prompts, load saved config)
      python train.py --save-config              (save answered config then train)
"""

import json
import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List


# ── pretty printing ──────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

def banner(text: str, style: str = "bold cyan"):
    if RICH:
        console.print(Panel(text, style=style, expand=False))
    else:
        print(f"\n{'='*60}\n  {text}\n{'='*60}")

def info(text: str):
    if RICH:
        console.print(f"[bold green]>[/bold green] {text}")
    else:
        print(f"  > {text}")

def warn(text: str):
    if RICH:
        console.print(f"[bold yellow]![/bold yellow] {text}")
    else:
        print(f"  ! {text}")

def section(text: str):
    if RICH:
        console.rule(f"[bold]{text}[/bold]")
    else:
        print(f"\n--- {text} ---")

def ask(prompt: str, default=None, choices: Optional[List[str]] = None) -> str:
    if RICH:
        kwargs = {}
        if default is not None:
            kwargs["default"] = str(default)
        if choices:
            kwargs["choices"] = choices
        return Prompt.ask(f"[cyan]{prompt}[/cyan]", **kwargs)
    else:
        suffix = f" [{default}]" if default is not None else ""
        if choices:
            suffix += f" ({'/'.join(choices)})"
        raw = input(f"  {prompt}{suffix}: ").strip()
        return raw if raw else (str(default) if default is not None else "")

def ask_int(prompt: str, default: int) -> int:
    if RICH:
        return IntPrompt.ask(f"[cyan]{prompt}[/cyan]", default=default)
    else:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return int(raw) if raw else default

def ask_float(prompt: str, default: float) -> float:
    if RICH:
        return FloatPrompt.ask(f"[cyan]{prompt}[/cyan]", default=default)
    else:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return float(raw) if raw else default

def ask_bool(prompt: str, default: bool = True) -> bool:
    if RICH:
        return Confirm.ask(f"[cyan]{prompt}[/cyan]", default=default)
    else:
        suffix = f" [{'Y/n' if default else 'y/N'}]"
        raw = input(f"  {prompt}{suffix}: ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes", "true", "1")


# ── config dataclass ─────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    use_qlora: bool = False
    quant_bits: int = 4          # 4 or 8, only used when use_qlora=True
    torch_dtype: str = "bfloat16"

    # Dataset
    dataset_path: str = ""
    dataset_format: str = "jsonl"   # jsonl | hf_hub | csv
    dataset_split: str = "train"
    val_split_ratio: float = 0.05
    has_images: bool = True
    messages_column: str = "messages"
    max_seq_len: int = 2048

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_bias: str = "none"

    # Training
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optimizer: str = "adamw_torch"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0

    # Logging / saving
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "none"    # none | wandb | tensorboard
    run_name: str = "qwen25vl-sft-lora"
    resume_from_checkpoint: Optional[str] = None
    disable_tqdm: bool = True    # avoid per-batch progress-bar spam (esp. during eval on Colab)


# ── interactive config builder ───────────────────────────────────────────────

def build_config_interactively() -> TrainingConfig:
    cfg = TrainingConfig()

    banner("Qwen2.5-VL-3B  ·  Interactive SFT LoRA/QLoRA Trainer")

    # ── model ────────────────────────────────────────────────────────────────
    section("Model")
    cfg.model_name = ask(
        "Base model (HF repo or local path)",
        default=cfg.model_name,
    )

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            info(f"Detected GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb >= 6:
                info("Qwen2.5-VL-3B in bf16 should fit without quantization.")
            else:
                info("VRAM may be tight — consider enabling QLoRA below.")
        else:
            warn("No CUDA GPU detected. Training will use CPU (very slow).")
    except Exception:
        pass
    cfg.use_qlora = ask_bool(
        "Use QLoRA (4-bit quantization)? Saves VRAM but slightly slower",
        default=False,
    )
    if cfg.use_qlora:
        cfg.quant_bits = int(ask("Quantization bits", default="4", choices=["4", "8"]))

    cfg.torch_dtype = ask(
        "Compute dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )

    # ── dataset ──────────────────────────────────────────────────────────────
    section("Dataset")
    info("Supported formats:")
    info("  jsonl  – local .jsonl file; each line is a JSON object")
    info("  hf_hub – Hugging Face Hub dataset (e.g. username/my-dataset)")
    info("  csv    – local .csv file")

    cfg.dataset_format = ask(
        "Dataset format", default="jsonl", choices=["jsonl", "hf_hub", "csv"]
    )
    cfg.dataset_path = ask("Dataset path or HF Hub ID", default=cfg.dataset_path or "")
    while not cfg.dataset_path:
        warn("Dataset path cannot be empty.")
        cfg.dataset_path = ask("Dataset path or HF Hub ID")

    if cfg.dataset_format == "hf_hub":
        cfg.dataset_split = ask("Dataset split", default="train")

    cfg.messages_column = ask(
        "Column name containing messages/conversations", default="messages"
    )
    cfg.has_images = ask_bool(
        "Does your dataset contain images? (images embedded inside messages as {type:image})",
        default=True,
    )

    cfg.val_split_ratio = ask_float(
        "Validation split ratio (0 to disable)", default=0.05
    )
    cfg.max_seq_len = ask_int("Max sequence length (tokens)", default=2048)

    # ── lora ─────────────────────────────────────────────────────────────────
    section("LoRA Hyperparameters")

    cfg.lora_r = ask_int("LoRA rank (r)  – higher = more capacity, more VRAM", default=16)
    cfg.lora_alpha = ask_int("LoRA alpha – usually 2× rank", default=cfg.lora_r * 2)
    cfg.lora_dropout = ask_float("LoRA dropout", default=0.05)

    info(f"Default target modules: {cfg.lora_target_modules}")
    custom_targets = ask_bool("Customise target modules?", default=False)
    if custom_targets:
        raw = ask(
            "Enter comma-separated module names",
            default=",".join(cfg.lora_target_modules),
        )
        cfg.lora_target_modules = [m.strip() for m in raw.split(",") if m.strip()]

    cfg.lora_bias = ask(
        "LoRA bias mode", default="none", choices=["none", "all", "lora_only"]
    )

    # ── training ─────────────────────────────────────────────────────────────
    section("Training Hyperparameters")

    run_name = ask("Run name (used as output subfolder)", default="run1")
    cfg.run_name = run_name
    cfg.output_dir = f"./output/{run_name}"
    info(f"Output will be saved to: {cfg.output_dir}")
    cfg.num_train_epochs = ask_int("Number of epochs", default=3)
    cfg.per_device_train_batch_size = ask_int(
        "Batch size per device (keep 1 for 8 GB VRAM)", default=1
    )
    cfg.gradient_accumulation_steps = ask_int(
        "Gradient accumulation steps (effective batch = batch × accum)", default=8
    )
    cfg.learning_rate = ask_float("Learning rate", default=2e-4)
    cfg.lr_scheduler_type = ask(
        "LR scheduler",
        default="cosine",
        choices=["cosine", "linear", "constant", "cosine_with_restarts"],
    )
    cfg.warmup_ratio = ask_float("Warmup ratio", default=0.05)
    cfg.weight_decay = ask_float("Weight decay", default=0.01)
    cfg.optimizer = ask(
        "Optimizer",
        default="adamw_torch",
        choices=["adamw_torch", "adamw_8bit", "paged_adamw_8bit", "paged_adamw_32bit"],
    )
    cfg.max_grad_norm = ask_float("Max gradient norm", default=1.0)
    cfg.gradient_checkpointing = ask_bool(
        "Enable gradient checkpointing? (saves VRAM, ~20% slower)", default=True
    )

    # ── logging / saving ─────────────────────────────────────────────────────
    section("Logging & Saving")

    cfg.logging_steps = ask_int("Log every N steps", default=10)
    cfg.save_steps = ask_int("Save checkpoint every N steps", default=200)
    cfg.eval_steps = ask_int("Evaluate every N steps", default=200)
    cfg.save_total_limit = ask_int("Max checkpoints to keep", default=3)
    cfg.report_to = ask(
        "Report metrics to",
        default="none",
        choices=["none", "wandb", "tensorboard"],
    )
    # run_name already set from the output subfolder name above

    cfg.disable_tqdm = ask_bool(
        "Disable progress bars? (recommended on Colab — avoids per-batch eval spam)",
        default=True,
    )

    resume = ask_bool("Resume from a checkpoint?", default=False)
    if resume:
        cfg.resume_from_checkpoint = ask("Checkpoint path")

    return cfg


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(cfg: TrainingConfig):
    section("Configuration Summary")
    if RICH:
        t = Table(show_header=True, header_style="bold magenta")
        t.add_column("Setting", style="cyan")
        t.add_column("Value")
        rows = [
            ("Run name", cfg.run_name),
            ("Model", cfg.model_name),
            ("QLoRA", f"{'Yes (' + str(cfg.quant_bits) + '-bit)' if cfg.use_qlora else 'No (bf16 LoRA)'}"),
            ("Dataset", cfg.dataset_path),
            ("Format", cfg.dataset_format),
            ("Messages col", cfg.messages_column),
            ("Images in messages", str(cfg.has_images)),
            ("Max seq len", str(cfg.max_seq_len)),
            ("LoRA r / alpha", f"{cfg.lora_r} / {cfg.lora_alpha}"),
            ("LoRA targets", ", ".join(cfg.lora_target_modules)),
            ("Output dir", cfg.output_dir),
            ("Epochs", str(cfg.num_train_epochs)),
            ("Batch size", str(cfg.per_device_train_batch_size)),
            ("Grad accum", str(cfg.gradient_accumulation_steps)),
            ("Effective batch", str(cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)),
            ("Learning rate", str(cfg.learning_rate)),
            ("LR scheduler", cfg.lr_scheduler_type),
            ("Optimizer", cfg.optimizer),
            ("Grad checkpoint", str(cfg.gradient_checkpointing)),
            ("Report to", cfg.report_to),
        ]
        for k, v in rows:
            t.add_row(k, v)
        console.print(t)
    else:
        for k, v in asdict(cfg).items():
            print(f"  {k}: {v}")


# ── dataset loading ──────────────────────────────────────────────────────────

def load_dataset_from_config(cfg: TrainingConfig):
    from datasets import load_dataset

    info(f"Loading dataset from: {cfg.dataset_path}")

    if cfg.dataset_format == "jsonl":
        path = Path(cfg.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        ds = load_dataset("json", data_files=str(path), split="train")

    elif cfg.dataset_format == "csv":
        path = Path(cfg.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        ds = load_dataset("csv", data_files=str(path), split="train")

    elif cfg.dataset_format == "hf_hub":
        ds = load_dataset(cfg.dataset_path, split=cfg.dataset_split)

    else:
        raise ValueError(f"Unknown dataset format: {cfg.dataset_format}")

    info(f"Loaded {len(ds)} samples.")
    return ds


def split_dataset(ds, val_ratio: float):
    if val_ratio <= 0:
        return ds, None
    split = ds.train_test_split(test_size=val_ratio, seed=42)
    info(f"Train: {len(split['train'])}  |  Val: {len(split['test'])}")
    return split["train"], split["test"]


# ── collator for Qwen2.5-VL ──────────────────────────────────────────────────

def make_collator(processor, cfg: TrainingConfig):
    """Returns a data collator that handles text-only or multimodal samples."""
    import torch
    from qwen_vl_utils import process_vision_info

    has_images = cfg.has_images

    def collate(batch):
        texts = []
        image_inputs_list = []
        video_inputs_list = []

        for sample in batch:
            messages = sample[cfg.messages_column]
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Could not parse messages column '{cfg.messages_column}' as JSON. "
                        f"Check your dataset format (CSV quoting issues are common). "
                        f"Original error: {e}"
                    ) from e

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            if has_images:
                image_inputs, video_inputs = process_vision_info(messages)
                image_inputs_list.append(image_inputs)
                video_inputs_list.append(video_inputs)

        if has_images:
            # Flatten lists
            flat_images = [img for imgs in image_inputs_list if imgs for img in imgs] or None
            flat_videos = [vid for vids in video_inputs_list if vids for vid in vids] or None
            inputs = processor(
                text=texts,
                images=flat_images,
                videos=flat_videos,
                padding=True,
                truncation=True,
                max_length=cfg.max_seq_len,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=cfg.max_seq_len,
                return_tensors="pt",
            )

        # Mask padding tokens in labels
        labels = inputs["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        inputs["labels"] = labels
        return inputs

    return collate


# ── fast path: pre-tokenized text-only training ──────────────────────────────

def tokenize_text_dataset(ds, processor, cfg: TrainingConfig):
    """Pre-tokenize a text-only dataset once so each step only pads tensors
    instead of re-running apply_chat_template + the tokenizer every batch."""
    tokenizer = getattr(processor, "tokenizer", processor)

    def _tok(sample):
        messages = sample[cfg.messages_column]
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse messages column '{cfg.messages_column}' as JSON. "
                    f"Check your dataset format (CSV quoting issues are common). "
                    f"Original error: {e}"
                ) from e
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(text, truncation=True, max_length=cfg.max_seq_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    return ds.map(_tok, remove_columns=ds.column_names, desc="Tokenizing")


def make_padding_collator(processor):
    """Collator for an already-tokenized dataset: just pad and build labels."""
    tokenizer = getattr(processor, "tokenizer", processor)

    def collate(batch):
        features = {
            "input_ids": [b["input_ids"] for b in batch],
            "attention_mask": [b["attention_mask"] for b in batch],
        }
        padded = tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100
        padded["labels"] = labels
        return padded

    return collate


# ── model + LoRA setup ───────────────────────────────────────────────────────

def load_model_and_processor(cfg: TrainingConfig):
    import torch
    from transformers import (
        AutoProcessor, AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    is_vl = cfg.has_images or "vl" in cfg.model_name.lower()

    info(f"Loading processor from {cfg.model_name} ...")
    if is_vl:
        processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
    else:
        processor = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        processor.pad_token = processor.pad_token or processor.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.torch_dtype, torch.bfloat16)

    ModelClass = Qwen2_5_VLForConditionalGeneration if is_vl else AutoModelForCausalLM

    if cfg.use_qlora:
        info(f"Loading model in {cfg.quant_bits}-bit QLoRA mode ...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=(cfg.quant_bits == 4),
            load_in_8bit=(cfg.quant_bits == 8),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = ModelClass.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_cfg,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        info(f"Loading model in {cfg.torch_dtype} LoRA mode ...")
        model = ModelClass.from_pretrained(
            cfg.model_name,
            torch_dtype=torch_dtype,
            device_map={"": 0},
            trust_remote_code=True,
        )

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()

    return model, processor


def apply_lora(model, cfg: TrainingConfig):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    if cfg.use_qlora:
        info("Preparing model for k-bit training ...")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias=cfg.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── training ─────────────────────────────────────────────────────────────────

def run_training(cfg: TrainingConfig):
    import torch
    from transformers import TrainingArguments, Trainer

    # Faster matmuls on Ampere/Ada GPUs (e.g. RTX 4060) at no quality cost for bf16.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── load model ────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(cfg)
    model = apply_lora(model, cfg)

    # ── load dataset ──────────────────────────────────────────────────────
    ds = load_dataset_from_config(cfg)
    train_ds, val_ds = split_dataset(ds, cfg.val_split_ratio)

    # ── data collator ─────────────────────────────────────────────────────
    # Text-only: pre-tokenize once (visible progress bar) and pad per batch —
    # far faster per step than templating + tokenizing live in the collator.
    # Multimodal: keep the live collator that processes images each batch.
    if cfg.has_images:
        collator = make_collator(processor, cfg)
    else:
        info("Pre-tokenizing dataset (one-time, replaces per-step tokenization) ...")
        train_ds = tokenize_text_dataset(train_ds, processor, cfg)
        if val_ds is not None:
            val_ds = tokenize_text_dataset(val_ds, processor, cfg)
        collator = make_padding_collator(processor)

    # ── training args ─────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)

    total_steps = (len(train_ds) // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)) * cfg.num_train_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    if cfg.gradient_checkpointing:
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        optim=cfg.optimizer,
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=(cfg.torch_dtype == "bfloat16"),
        fp16=(cfg.torch_dtype == "float16"),
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if val_ds else None,
        eval_strategy="steps" if val_ds else "no",
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to if cfg.report_to != "none" else [],
        run_name=cfg.run_name if cfg.report_to != "none" else None,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
        disable_tqdm=cfg.disable_tqdm,
    )

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    info("Starting training...")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    info("Saving final adapter weights ...")
    final_dir = os.path.join(cfg.output_dir, "final_adapter")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    info(f"Adapter saved to: {final_dir}")

    info("Done! To merge adapter into base model, run:  python merge_adapter.py")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive SFT LoRA/QLoRA trainer for Qwen2.5-VL")
    parser.add_argument("--config", type=str, help="Path to a saved JSON config (skips prompts)")
    parser.add_argument("--save-config", action="store_true", help="Save config to JSON before training")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            data = json.load(f)
        cfg = TrainingConfig(**data)
        info(f"Loaded config from {args.config}")
        print_summary(cfg)
    else:
        cfg = build_config_interactively()
        print_summary(cfg)
        proceed = ask_bool("\nLooks good — start training?", default=True)
        if not proceed:
            info("Aborted.")
            sys.exit(0)

    if args.save_config or (not args.config and ask_bool("Save this config to JSON for reuse?", default=True)):
        cfg_path = os.path.join(cfg.output_dir, "train_config.json")
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(cfg_path, "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        info(f"Config saved to {cfg_path}")

    run_training(cfg)


if __name__ == "__main__":
    main()
