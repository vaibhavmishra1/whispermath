from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from common import load_config, load_input_output_dataset, maybe_limit, merge_config, split_by_type


CONFIG_KEYS = [
    "dataset_id",
    "model_name",
    "output_dir",
    "seed",
    "max_source_length",
    "max_target_length",
    "validation_ratio",
    "test_ratio",
    "max_train_samples",
    "max_eval_samples",
    "num_train_epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "fp16",
    "bf16",
    "gradient_checkpointing",
    "optim",
    "dataloader_num_workers",
    "predict_with_generate",
    "generation_max_length",
    "generation_num_beams",
]


def device_hint() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def preprocess_batch(batch: Dict[str, List[str]], tokenizer: Any, max_source_length: int, max_target_length: int) -> Dict[str, Any]:
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["output_text"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def training_args_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"

    kwargs = {
        "output_dir": config["output_dir"],
        "overwrite_output_dir": True,
        "seed": int(config["seed"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "logging_steps": int(config["logging_steps"]),
        "save_steps": int(config["save_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "predict_with_generate": bool(config["predict_with_generate"]),
        "generation_max_length": int(config["generation_max_length"]),
        "generation_num_beams": int(config["generation_num_beams"]),
        "report_to": "none",
        "remove_unused_columns": True,
        "fp16": bool(config["fp16"]),
        "bf16": bool(config["bf16"]),
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "optim": config["optim"],
        "dataloader_num_workers": int(config["dataloader_num_workers"]),
    }

    kwargs[eval_key] = "steps"
    kwargs["eval_steps"] = int(config["eval_steps"])
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a ByT5 model for WhisperMath input_text -> output_text.")
    parser.add_argument("--config", type=Path, default=Path("configs/byt5_small.yaml"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--model-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-source-length", type=int)
    parser.add_argument("--max-target-length", type=int)
    parser.add_argument("--validation-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--optim")
    parser.add_argument("--dataloader-num-workers", type=int)
    parser.add_argument("--predict-with-generate", action="store_true")
    parser.add_argument("--generation-max-length", type=int)
    parser.add_argument("--generation-num-beams", type=int)
    args = parser.parse_args()

    config = merge_config(load_config(args.config), args, CONFIG_KEYS)
    config.setdefault("dataset_id", "vibhuiitj/whispermath-input-output")
    config.setdefault("model_name", "google/byt5-small")
    config.setdefault("output_dir", "models/byt5-small-whispermath")
    config.setdefault("seed", 7)
    config.setdefault("max_source_length", 512)
    config.setdefault("max_target_length", 512)
    config.setdefault("validation_ratio", 0.01)
    config.setdefault("test_ratio", 0.01)
    config.setdefault("num_train_epochs", 1)
    config.setdefault("learning_rate", 3e-4)
    config.setdefault("weight_decay", 0.01)
    config.setdefault("warmup_ratio", 0.03)
    config.setdefault("per_device_train_batch_size", 1)
    config.setdefault("per_device_eval_batch_size", 1)
    config.setdefault("gradient_accumulation_steps", 16)
    config.setdefault("logging_steps", 25)
    config.setdefault("eval_steps", 500)
    config.setdefault("save_steps", 500)
    config.setdefault("save_total_limit", 2)
    config.setdefault("fp16", False)
    config.setdefault("bf16", False)
    config.setdefault("gradient_checkpointing", False)
    config.setdefault("optim", "adamw_torch")
    config.setdefault("dataloader_num_workers", 0)
    config.setdefault("predict_with_generate", False)
    config.setdefault("generation_max_length", 512)
    config.setdefault("generation_num_beams", 1)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Device hint: {device_hint()}")
    print(f"Loading dataset: {config['dataset_id']}")
    raw_dataset = load_input_output_dataset(config["dataset_id"])
    splits = split_by_type(
        raw_dataset,
        validation_ratio=float(config["validation_ratio"]),
        test_ratio=float(config["test_ratio"]),
        seed=int(config["seed"]),
    )
    splits["train"] = maybe_limit(splits["train"], config.get("max_train_samples"), int(config["seed"]))
    splits["validation"] = maybe_limit(splits["validation"], config.get("max_eval_samples"), int(config["seed"]))

    print(splits)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    if bool(config["gradient_checkpointing"]):
        model.config.use_cache = False

    tokenized = splits.map(
        lambda batch: preprocess_batch(
            batch,
            tokenizer=tokenizer,
            max_source_length=int(config["max_source_length"]),
            max_target_length=int(config["max_target_length"]),
        ),
        batched=True,
        remove_columns=splits["train"].column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs(config))

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": collator,
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Saved model to {config['output_dir']}")


if __name__ == "__main__":
    main()
