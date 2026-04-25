"""
Fine-tune a safety-aligned LLaMA model on GSM8K.
Output is saved to finetuned_models/{output_folder}/ and is directly
compatible with run_safedelta.py for the Safe Delta pipeline.

Example (base model):
python llama2/finetuning_gsm8k.py \
    --model_name kmseong/llama2_7b-Safety-FT-lr3e-5 \
    --output_folder gsm8k-llama2-7b-safeft \
    --lr 3e-5 --epochs 3

Example (instruct model):
python llama2/finetuning_gsm8k.py \
    --model_name kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_folder gsm8k-llama2-7b-chat-safeft \
    --lr 5e-5 --epochs 3

Then apply Safe Delta:
python llama2/run_safedelta.py \
    --model_name_align kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --model_name_ft finetuned_models/gsm8k-llama2-7b-chat-safeft \
    --scale 0.1 \
    --safe_data_path ./llama2/safedelta/data/circuit_breakers_train.json \
    --upload_name kmseong/llama2-7b-chat-safedelta-scale0.1
"""

import argparse
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def is_instruct_model(model_ref: str) -> bool:
    return any(tag in str(model_ref).lower() for tag in ("instruct", "chat"))


def tokenize_gsm8k_example(
    question: str,
    answer_text: str,
    tokenizer,
    max_length: int,
    model_ref: str,
) -> Dict[str, List[int]]:
    """
    SFT tokenization for a single GSM8K example.
    - instruct / chat models  → apply_chat_template
    - base models             → plain "Question: ...\nAnswer:" prefix
    Loss is masked on the prompt portion (labels = -100).
    """
    question = str(question).strip()
    answer_text = str(answer_text).strip()

    if is_instruct_model(model_ref):
        # ── instruct branch ──────────────────────────────────────────────
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
    else:
        # ── base model branch ────────────────────────────────────────────
        prompt_text = f"Question: {question}\nAnswer:"
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        remain = max(1, max_length - len(prompt_ids))
        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=True,
            max_length=remain,
        )["input_ids"]
        # append EOS if it fits
        if (
            tokenizer.eos_token_id is not None
            and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id)
            and len(prompt_ids) + len(answer_ids) < max_length
        ):
            answer_ids = answer_ids + [tokenizer.eos_token_id]

        full_ids = (prompt_ids + answer_ids)[:max_length]

    labels = full_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


@dataclass
class PaddingCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a safety-aligned model on GSM8K (SafeDelta pipeline)"
    )
    # model
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model ID or local path (safety-aligned model)")
    # output — saved under finetuned_models/{output_folder}
    p.add_argument("--output_folder", type=str, default=None,
                   help="Sub-folder name under finetuned_models/ (auto-generated if omitted)")
    # GSM8K data
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    p.add_argument("--dataset_subset", type=str, default="main")
    p.add_argument("--num_train_samples", type=int, default=7473)
    p.add_argument("--cache_dir", type=str, default="./cache")
    # training hyper-parameters
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--report_to", type=str, default="none")
    return p.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # resolve model path
    if args.model_name.startswith(("./", "/", "../")):
        model_path = os.path.abspath(args.model_name)
    else:
        model_path = args.model_name  # HF Hub ID

    # output directory  (finetuned_models is the SafeDelta convention)
    if args.output_folder is None:
        slug = os.path.basename(model_path.rstrip("/")).replace("/", "_")
        args.output_folder = f"gsm8k-{slug}"
    output_dir = os.path.join("finetuned_models", args.output_folder)
    os.makedirs(output_dir, exist_ok=True)

    use_chat = is_instruct_model(model_path)
    print(f"[Info] model        : {model_path}")
    print(f"[Info] output_dir   : {output_dir}")
    print(f"[Info] instruct mode: {use_chat}  (chat template: {'yes' if use_chat else 'no'})")

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ───────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"[Info] model params : {total / 1e9:.2f}B")

    # ── Dataset ─────────────────────────────────────────────────────────
    train_ds = load_dataset(
        args.dataset_name,
        args.dataset_subset,
        split="train",
        cache_dir=args.cache_dir,
    )
    if args.num_train_samples and args.num_train_samples < len(train_ds):
        train_ds = train_ds.select(range(args.num_train_samples))
    print(f"[Info] train samples: {len(train_ds)}")

    def preprocess(ex):
        return tokenize_gsm8k_example(
            ex["question"], ex["answer"], tokenizer, args.max_length, model_path
        )

    train_tok = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        num_proc=max(1, args.num_workers),
        desc="Tokenising GSM8K",
    )

    # ── Trainer ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="no",
        report_to=args.report_to,
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        tokenizer=tokenizer,
        data_collator=PaddingCollator(tokenizer),
    )

    print("[Info] Starting training...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Info] Fine-tuned model saved to: {output_dir}")

    # save config for reference
    config_snapshot = {
        "model_name": model_path,
        "output_dir": output_dir,
        "dataset": args.dataset_name,
        "num_train_samples": len(train_tok),
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "instruct_mode": use_chat,
    }
    with open(os.path.join(output_dir, "finetune_config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("\n[Done] Next step — apply Safe Delta:")
    print(f"  python run_safedelta.py \\")
    print(f"    --model_name_align '{model_path}' \\")
    print(f"    --model_name_ft '{output_dir}' \\")
    print(f"    --scale 0.1")


if __name__ == "__main__":
    main()
