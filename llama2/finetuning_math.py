"""
Fine-tune a safety-aligned LLaMA model on Hendrycks MATH.
Output is saved to finetuned_models/{output_folder}/ and is directly
compatible with run_safedelta.py for the Safe Delta pipeline.

Example (base model):
    python llama2/finetuning_math.py \
        --model_name kmseong/llama2_7b-Safety-FT-lr3e-5 \
        --output_folder math-llama2-7b-safeft \
        --lr 3e-5 --epochs 3

Example (instruct/chat model):
python llama2/finetuning_math.py \
    --model_name kmseong/llama3_2_3b-instruct-SSFT-lr5e-5 \
    --output_folder math-llama3_2_3b-instruct-safeft \
    --lr 3e-5 --epochs 3

Subset of subjects / levels:
    python llama2/finetuning_math.py \
        --model_name kmseong/llama3_2_3b-instruct-SSFT-lr5e-5 \
        --output_folder math-llama3_2_3b-instruct-algebra \
        --math_subjects Algebra \
        --math_levels "Level 1,Level 2,Level 3"

Then apply Safe Delta:
python llama2/run_safedelta.py \
    --model_name_align kmseong/llama3_2_3b-instruct-SSFT-lr5e-5 \
    --model_name_ft kmseong/llama3_2_3b_instruct_MATH_SSFT_lr3e-5 \
    --scale 2 \
    --safe_data_path ./llama2/safedelta/data/circuit_breakers_train.json \
    --upload_name kmseong/llama3_2_3b-instruct-math-safedelta-scale2
"""

import argparse
import os
import re
import json
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -----------------------------------------------------------------------
# Subject / Level helpers
# -----------------------------------------------------------------------

SUBJECT_TO_CONFIG = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}
VALID_LEVELS = {f"Level {i}" for i in range(1, 6)}


def normalize_csv_arg(raw_value: str) -> str:
    value = str(raw_value).strip()
    if len(value) >= 2 and (
        (value[0] == '"' and value[-1] == '"')
        or (value[0] == "'" and value[-1] == "'")
    ):
        value = value[1:-1].strip()
    return value


# -----------------------------------------------------------------------
# Answer extraction helpers (from Hendrycks MATH solutions)
# -----------------------------------------------------------------------

def last_boxed_only_string(text: str):
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if s is None:
        raise ValueError("remove_boxed received None")
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left):]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    left = "\\fbox{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def extract_final_answer_from_solution(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        raise ValueError(f"No boxed answer in: {solution[:200]!r}")
    return remove_boxed(boxed).strip()


_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_solution_for_reasoning(solution: str, final_answer: str) -> str:
    text = solution.strip()
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        text = text.replace(boxed, final_answer)
    for token in ("$", "\\[", "\\]", "\\(", "\\)", "\\boxed", "\\fbox"):
        text = text.replace(token, "")
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def build_target(solution: str, rng: random.Random, train_on_mixed_formats: bool) -> str:
    final_answer = extract_final_answer_from_solution(solution)
    rationale = clean_solution_for_reasoning(solution, final_answer)

    long_target = f"{rationale}\nFinal Answer: ${final_answer}$"
    short_target = f"Final Answer: ${final_answer}$"
    minimal_target = f"${final_answer}$"

    if not train_on_mixed_formats:
        return long_target

    draw = rng.random()
    if draw < 0.70:
        return long_target
    if draw < 0.90:
        return short_target
    return minimal_target


# -----------------------------------------------------------------------
# Tokenization
# -----------------------------------------------------------------------

def is_instruct_model(model_ref: str) -> bool:
    return any(tag in str(model_ref).lower() for tag in ("instruct", "chat"))


def tokenize_math_example(
    problem: str,
    target_text: str,
    tokenizer,
    max_length: int,
    model_ref: str,
) -> Dict[str, List[int]]:
    """
    SFT tokenization for a single MATH example.
    - instruct / chat models  → apply_chat_template
    - base models             → plain "Question: ...\nAnswer:" prefix
    Loss is masked on the prompt portion (labels = -100).
    """
    problem = str(problem).strip()
    target_text = str(target_text).strip()

    if is_instruct_model(model_ref):
        # ── instruct branch ──────────────────────────────────────────────
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": target_text},
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
        prompt_text = f"Question: {problem}\nAnswer:"
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        remain = max(1, max_length - len(prompt_ids))
        answer_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=remain,
        )["input_ids"]
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
        description="Fine-tune a safety-aligned model on Hendrycks MATH (SafeDelta pipeline)"
    )
    # model
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model ID or local path (safety-aligned model)")
    # output — saved under finetuned_models/{output_folder}
    p.add_argument("--output_folder", type=str, default=None,
                   help="Sub-folder name under finetuned_models/ (auto-generated if omitted)")
    # MATH dataset
    p.add_argument("--math_dataset_path", type=str, default="EleutherAI/hendrycks_math")
    p.add_argument("--math_subjects", type=str, default="all",
                   help="Comma-separated subjects or 'all'. "
                        "Valid: Algebra, Counting & Probability, Geometry, "
                        "Intermediate Algebra, Number Theory, Prealgebra, Precalculus")
    p.add_argument("--math_levels", type=str, default="all",
                   help="Comma-separated levels or 'all'. e.g. 'Level 1,Level 2'")
    p.add_argument("--num_train_samples", type=int, default=0,
                   help="Max training samples (0 = use all)")
    p.add_argument("--train_on_mixed_formats", action="store_true", default=False,
                   help="Mix long/short/minimal answer formats during training")
    p.add_argument("--cache_dir", type=str, default="./cache")
    # training hyper-parameters
    p.add_argument("--lr", type=float, default=1e-5)
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

    # output directory (finetuned_models is the SafeDelta convention)
    if args.output_folder is None:
        slug = os.path.basename(model_path.rstrip("/")).replace("/", "_")
        args.output_folder = f"math-{slug}"
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
    subjects_arg = normalize_csv_arg(args.math_subjects)
    if subjects_arg.lower() == "all":
        subjects = list(SUBJECT_TO_CONFIG.keys())
    else:
        subjects = [normalize_csv_arg(s) for s in subjects_arg.split(",") if normalize_csv_arg(s)]

    print(f"[Info] subjects     : {subjects}")

    datasets_per_subject = []
    for subject in subjects:
        config_name = SUBJECT_TO_CONFIG[subject]
        ds = load_dataset(
            args.math_dataset_path,
            config_name,
            split="train",
            cache_dir=args.cache_dir,
        )
        ds = ds.map(lambda ex, s=subject: {"type": s})
        datasets_per_subject.append(ds)
    train_ds = concatenate_datasets(datasets_per_subject)

    # level filtering
    levels_arg = normalize_csv_arg(args.math_levels)
    if levels_arg.lower() != "all":
        allowed_levels = set()
        for item in levels_arg.split(","):
            item = normalize_csv_arg(item)
            if not item:
                continue
            lvl = item if item.startswith("Level ") else f"Level {int(item)}"
            if lvl not in VALID_LEVELS:
                raise ValueError(f"Invalid math level: {item!r}")
            allowed_levels.add(lvl)
        train_ds = train_ds.filter(lambda ex: ex.get("level") in allowed_levels)
        print(f"[Info] levels       : {sorted(allowed_levels)}")
    else:
        print(f"[Info] levels       : all")

    train_ds = train_ds.shuffle(seed=args.seed)
    if args.num_train_samples and args.num_train_samples > 0:
        train_ds = train_ds.select(range(min(args.num_train_samples, len(train_ds))))
    print(f"[Info] train samples: {len(train_ds)}")

    # ── Tokenize ────────────────────────────────────────────────────────
    def preprocess(ex, idx: int):
        problem = ex.get("problem", "").strip()
        solution = ex.get("solution", "").strip()
        rng = random.Random(args.seed + idx)
        try:
            target_text = build_target(solution, rng, args.train_on_mixed_formats)
        except ValueError:
            # fallback: use raw solution if boxed answer not found
            target_text = solution
        return tokenize_math_example(problem, target_text, tokenizer, args.max_length, model_path)

    train_tok = train_ds.map(
        preprocess,
        with_indices=True,
        remove_columns=train_ds.column_names,
        num_proc=max(1, args.num_workers),
        desc="Tokenising Hendrycks MATH",
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

    config_snapshot = {
        "model_name": model_path,
        "output_dir": output_dir,
        "dataset": args.math_dataset_path,
        "math_subjects": subjects,
        "math_levels": args.math_levels,
        "num_train_samples": len(train_tok),
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "instruct_mode": use_chat,
        "train_on_mixed_formats": args.train_on_mixed_formats,
    }
    with open(os.path.join(output_dir, "finetune_config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("\n[Done] Next step — apply Safe Delta:")
    print(f"  python llama2/run_safedelta.py \\")
    print(f"    --model_name_align '{model_path}' \\")
    print(f"    --model_name_ft '{output_dir}' \\")
    print(f"    --scale 0.1 \\")
    print(f"    --safe_data_path ./llama2/safedelta/data/circuit_breakers_train.json")


if __name__ == "__main__":
    main()
