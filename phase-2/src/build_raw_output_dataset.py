from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from datasets import Image, load_dataset
from tqdm import tqdm

from sample_latex import clean_latex, is_candidate


LATEX_DATASET = "OleehyO/latex-formulas"
LATEX_CONFIG = "cleaned_formulas"
LATEX_SPLIT = "train"
LATEX_COLUMN = "latex_formula"

MIXED_DATASET = "math-ai/StackMathQA"
MIXED_CONFIG = "stackmathqa100k"
MIXED_SPLIT = "train"

NORMAL_DATASET = "Salesforce/wikitext"
NORMAL_CONFIG = "wikitext-103-raw-v1"
NORMAL_SPLIT = "train"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, default=str)


def next_id(prefix: str, index: int) -> str:
    return f"{prefix}-{index:08d}"


def maybe_limit_reached(count: int, limit: Optional[int]) -> bool:
    return limit is not None and count >= limit


def iter_latex_rows(limit: Optional[int], max_chars: int, filter_profile: str) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(
        LATEX_DATASET,
        LATEX_CONFIG,
        split=LATEX_SPLIT,
        streaming=True,
    )
    if "image" in dataset.features:
        dataset = dataset.cast_column("image", Image(decode=False))

    seen: set[str] = set()
    emitted = 0
    for source_row_index, row in enumerate(dataset):
        original_latex = str(row.get(LATEX_COLUMN, ""))
        latex = clean_latex(original_latex)
        if not latex or latex in seen:
            continue
        seen.add(latex)

        if filter_profile != "none":
            if not is_candidate(latex, min_chars=3, max_chars=max_chars, profile=filter_profile):
                continue

        yield {
            "id": next_id("latex", emitted),
            "output_text": latex,
            "type": "latex",
            "source_dataset": LATEX_DATASET,
            "source_config": LATEX_CONFIG,
            "source_split": LATEX_SPLIT,
            "source_row_index": source_row_index,
            "source_field": LATEX_COLUMN,
            "metadata_json": metadata_json({
                "original_latex": original_latex,
            }),
        }
        emitted += 1
        if maybe_limit_reached(emitted, limit):
            break


def iter_mixed_rows(limit: int, max_chars: int) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(
        MIXED_DATASET,
        MIXED_CONFIG,
        split=MIXED_SPLIT,
        streaming=True,
    )

    emitted = 0
    seen: set[str] = set()
    for source_row_index, row in enumerate(dataset):
        for source_field in ("Q", "A"):
            text = normalize_text(str(row.get(source_field, "")))
            if not text or text in seen:
                continue
            if len(text) > max_chars:
                continue
            seen.add(text)

            yield {
                "id": next_id("mixed", emitted),
                "output_text": text,
                "type": "mixed",
                "source_dataset": MIXED_DATASET,
                "source_config": MIXED_CONFIG,
                "source_split": MIXED_SPLIT,
                "source_row_index": source_row_index,
                "source_field": source_field,
                "metadata_json": metadata_json({
                    "meta": row.get("meta"),
                }),
            }
            emitted += 1
            if maybe_limit_reached(emitted, limit):
                return


def iter_normal_rows(limit: int, min_chars: int, max_chars: int) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(
        NORMAL_DATASET,
        NORMAL_CONFIG,
        split=NORMAL_SPLIT,
        streaming=True,
    )

    emitted = 0
    seen: set[str] = set()
    for source_row_index, row in enumerate(dataset):
        text = normalize_text(str(row.get("text", "")))
        if not text or text in seen:
            continue
        if text.startswith("=") and text.endswith("="):
            continue
        if not (min_chars <= len(text) <= max_chars):
            continue
        seen.add(text)

        yield {
            "id": next_id("normal", emitted),
            "output_text": text,
            "type": "normal",
            "source_dataset": NORMAL_DATASET,
            "source_config": NORMAL_CONFIG,
            "source_split": NORMAL_SPLIT,
            "source_row_index": source_row_index,
            "source_field": "text",
            "metadata_json": metadata_json({}),
        }
        emitted += 1
        if maybe_limit_reached(emitted, limit):
            break


def write_rows(output: Path, rows: Iterable[dict[str, Any]], label: str, total: Optional[int]) -> int:
    count = 0
    for row in tqdm(rows, total=total, desc=label):
        append_jsonl(output, row)
        count += 1
    return count


def parse_latex_limit(value: str) -> Optional[int]:
    if value.lower() in {"all", "none", "full"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("limit must be non-negative or 'all'")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw output_text dataset rows from LaTeX, mixed math, and normal text sources.")
    parser.add_argument("--output", type=Path, default=Path("data/compiled/raw_output_dataset.jsonl"))
    parser.add_argument("--latex-limit", type=parse_latex_limit, default=None, help="Use 'all' for all filtered latex rows.")
    parser.add_argument("--mixed-limit", type=int, default=50_000)
    parser.add_argument("--normal-limit", type=int, default=50_000)
    parser.add_argument("--latex-max-chars", type=int, default=120)
    parser.add_argument("--latex-filter", choices=["none", "simple", "broad"], default="none")
    parser.add_argument("--mixed-max-chars", type=int, default=2_500)
    parser.add_argument("--normal-min-chars", type=int, default=30)
    parser.add_argument("--normal-max-chars", type=int, default=1_200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    counts = {
        "latex": write_rows(
            args.output,
            iter_latex_rows(args.latex_limit, args.latex_max_chars, args.latex_filter),
            "latex",
            args.latex_limit,
        ),
        "mixed": write_rows(
            args.output,
            iter_mixed_rows(args.mixed_limit, args.mixed_max_chars),
            "mixed",
            args.mixed_limit,
        ),
        "normal": write_rows(
            args.output,
            iter_normal_rows(args.normal_limit, args.normal_min_chars, args.normal_max_chars),
            "normal",
            args.normal_limit,
        ),
    }

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"output": str(args.output), "counts": counts}, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote dataset: {args.output}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
