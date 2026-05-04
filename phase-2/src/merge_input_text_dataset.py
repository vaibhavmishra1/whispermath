from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset
from tqdm import tqdm


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def extract_response_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]

    pieces: list[str] = []
    for output_item in body.get("output", []):
        for content in output_item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                pieces.append(content["text"])
    return "\n".join(pieces).strip()


def parse_json_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def parse_batch_item(item: dict[str, Any]) -> tuple[Optional[dict[str, str]], Optional[dict[str, Any]]]:
    custom_id = item["custom_id"]
    if item.get("error"):
        return None, {"custom_id": custom_id, "error_type": "batch_item_error", "error": item.get("error")}

    response = item.get("response") or {}
    body = response.get("body") or {}
    text = extract_response_text(body)

    try:
        parsed = parse_json_text(text)
    except Exception as exc:
        return None, {
            "custom_id": custom_id,
            "error_type": "json_parse_error",
            "error": str(exc),
            "response_text": text,
        }

    input_text = str(parsed.get("input_text", "")).strip()
    if not input_text:
        return None, {
            "custom_id": custom_id,
            "error_type": "missing_input_text",
            "response_text": text,
        }

    return {
        "input_text": input_text,
        "input_style": str(parsed.get("input_style", "")).strip(),
        "input_generation_notes": str(parsed.get("notes", "")).strip(),
    }, None


def load_batch_results(output_dir: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    results: dict[str, dict[str, str]] = {}
    errors: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.output.jsonl")):
        for item in tqdm(load_jsonl(path), desc=path.name):
            parsed, error = parse_batch_item(item)
            if parsed:
                results[item["custom_id"]] = parsed
            if error:
                error["batch_output_file"] = path.name
                errors.append(error)
    return results, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge OpenAI Batch outputs into a final input_text/output_text training dataset.")
    parser.add_argument("--source", type=Path, default=Path("data/compiled/raw_output_dataset_balanced_50k.jsonl"))
    parser.add_argument("--batch-output-dir", type=Path, default=Path("data/batches/input_text_outputs"))
    parser.add_argument("--output", type=Path, default=Path("data/compiled/input_output_dataset_150k.jsonl"))
    parser.add_argument("--errors-output", type=Path, default=Path("data/compiled/input_output_dataset_150k.merge_errors.jsonl"))
    parser.add_argument("--validate-hf", action="store_true")
    args = parser.parse_args()

    batch_results, parse_errors = load_batch_results(args.batch_output_dir)
    counts: Counter[str] = Counter()
    missing: list[str] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in tqdm(load_jsonl(args.source), desc="merge"):
            row_type = row["type"]
            if row_type == "normal":
                merged = {
                    **row,
                    "input_text": row["output_text"],
                    "input_style": "identity",
                    "input_generation_notes": "normal text copied directly",
                }
            else:
                generated = batch_results.get(row["id"])
                if not generated or not generated["input_text"]:
                    missing.append(row["id"])
                    continue
                merged = {**row, **generated}

            handle.write(json.dumps(merged, ensure_ascii=False, default=str) + "\n")
            counts[row_type] += 1

    summary = {
        "source": str(args.source),
        "batch_output_dir": str(args.batch_output_dir),
        "output": str(args.output),
        "counts": dict(counts),
        "total": sum(counts.values()),
        "batch_result_count": len(batch_results),
        "parse_error_count": len(parse_errors),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "errors_output": str(args.errors_output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    args.errors_output.parent.mkdir(parents=True, exist_ok=True)
    with args.errors_output.open("w", encoding="utf-8") as handle:
        for error in parse_errors:
            handle.write(json.dumps(error, ensure_ascii=False, default=str) + "\n")
        for custom_id in missing:
            handle.write(json.dumps({"custom_id": custom_id, "error_type": "missing_batch_result"}, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))

    if args.validate_hf:
        dataset = Dataset.from_json(str(args.output))
        print(dataset)
        print(dataset.features)


if __name__ == "__main__":
    main()
