from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


TYPES = ("latex", "mixed", "normal")


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_config(config: Dict[str, Any], args: Any, keys: Iterable[str]) -> Dict[str, Any]:
    merged = dict(config)
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    return merged


def get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def load_input_output_dataset(dataset_id: str, split: str = "train") -> Dataset:
    dataset = load_dataset(dataset_id, split=split, token=get_hf_token())
    required = {"input_text", "output_text", "type"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    dataset = dataset.filter(
        lambda row: bool(str(row["input_text"]).strip()) and bool(str(row["output_text"]).strip())
    )
    return dataset


def split_by_type(dataset: Dataset, validation_ratio: float, test_ratio: float, seed: int) -> DatasetDict:
    train_parts = []
    validation_parts = []
    test_parts = []

    for row_type in TYPES:
        subset = dataset.filter(lambda row, row_type=row_type: row["type"] == row_type)
        if len(subset) == 0:
            continue

        holdout_ratio = validation_ratio + test_ratio
        if holdout_ratio <= 0:
            train_parts.append(subset)
            continue

        first = subset.train_test_split(test_size=holdout_ratio, seed=seed)
        holdout = first["test"]
        train_parts.append(first["train"])

        relative_test_ratio = test_ratio / holdout_ratio if holdout_ratio else 0
        second = holdout.train_test_split(test_size=relative_test_ratio, seed=seed)
        validation_parts.append(second["train"])
        test_parts.append(second["test"])

    return DatasetDict(
        {
            "train": concatenate_datasets(train_parts).shuffle(seed=seed),
            "validation": concatenate_datasets(validation_parts).shuffle(seed=seed),
            "test": concatenate_datasets(test_parts).shuffle(seed=seed),
        }
    )


def maybe_limit(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if not limit or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def character_error_rate(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not ref:
        return 0.0 if not pred else 1.0

    previous = list(range(len(pred) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current = [i]
        for j, pred_char in enumerate(pred, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (ref_char != pred_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1] / len(ref)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

