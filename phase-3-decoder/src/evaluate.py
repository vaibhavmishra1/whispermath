from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import character_error_rate, load_input_output_dataset, maybe_limit, normalize_text, split_by_type, write_jsonl


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_prediction(model: Any, tokenizer: Any, input_text: str, device: torch.device, max_source_length: int, max_new_tokens: int, num_beams: int) -> str:
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_source_length,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    summary: Dict[str, Any] = {"total": len(rows), "by_type": {}}
    for row_type, type_rows in by_type.items():
        exact = sum(1 for row in type_rows if normalize_text(row["prediction"]) == normalize_text(row["output_text"]))
        avg_cer = sum(row["cer"] for row in type_rows) / len(type_rows)
        summary["by_type"][row_type] = {
            "count": len(type_rows),
            "exact_match": exact / len(type_rows),
            "avg_character_error_rate": avg_cer,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a WhisperMath decoder model on held-out rows.")
    parser.add_argument("--dataset-id", default="vibhuiitj/whispermath-input-output")
    parser.add_argument("--model", default="models/byt5-small-whispermath")
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/eval_predictions.jsonl"))
    parser.add_argument("--output-summary", type=Path, default=Path("outputs/eval_summary.json"))
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    args = parser.parse_args()

    device = select_device()
    dataset = load_input_output_dataset(args.dataset_id)
    splits = split_by_type(dataset, args.validation_ratio, args.test_ratio, args.seed)
    test_dataset: Dataset = maybe_limit(splits["test"], args.max_samples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    rows = []
    for row in tqdm(test_dataset, desc="evaluate"):
        prediction = generate_prediction(
            model,
            tokenizer,
            row["input_text"],
            device,
            args.max_source_length,
            args.max_new_tokens,
            args.num_beams,
        )
        rows.append(
            {
                "id": row["id"],
                "type": row["type"],
                "input_text": row["input_text"],
                "output_text": row["output_text"],
                "prediction": prediction,
                "cer": character_error_rate(prediction, row["output_text"]),
            }
        )

    write_jsonl(args.output_jsonl, rows)
    summary = summarize(rows)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(__import__("json").dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
