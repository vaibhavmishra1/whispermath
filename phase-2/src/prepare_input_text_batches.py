from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_MODEL = "gpt-5.4-nano"
LATEX_STYLES = ("casual", "explicit", "teacher", "formal", "compact")

SYSTEM_PROMPT = """You create input_text for a spoken-math training dataset.

Return only JSON with this shape:
{"input_text":"...","input_style":"...","notes":"..."}

Rules:
- Generate exactly one input_text.
- Do not solve, simplify, or change the meaning.
- For type=latex, verbalize the formula as a person would say it aloud.
- For type=mixed, keep the surrounding natural language and verbalize math/LaTeX notation inside it.
- For type=latex, follow the requested style.
- Use lowercase for spoken math unless a proper noun needs capitalization.
- Avoid adding explanations.
"""


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def make_user_prompt(row: dict[str, Any], style: str) -> str:
    return (
        "Create input_text for this dataset row.\n\n"
        f"type: {row['type']}\n"
        f"requested_style: {style}\n"
        f"source_dataset: {row['source_dataset']}\n"
        "output_text:\n"
        "```text\n"
        f"{row['output_text']}\n"
        "```\n\n"
        "Return JSON only."
    )


def make_request(row: dict[str, Any], model: str, style: str, max_output_tokens: int, structured: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(row, style)},
        ],
        "max_output_tokens": max_output_tokens,
    }

    if structured:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "spoken_input_text",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "input_text": {"type": "string"},
                        "input_style": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["input_text", "input_style", "notes"],
                },
                "strict": True,
            }
        }

    return {
        "custom_id": row["id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def open_next_chunk(output_dir: Path, chunk_index: int):
    path = output_dir / f"requests_{chunk_index:05d}.jsonl"
    return path, path.open("w", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenAI Batch API request JSONL files for generating input_text.")
    parser.add_argument("--input", type=Path, default=Path("data/compiled/raw_output_dataset_balanced_50k.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/batches/input_text_requests"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--requests-per-file", type=int, default=10_000)
    parser.add_argument("--limit-requests", type=int, help="Optional cap for smoke testing.")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--no-structured-output", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    chunks: list[dict[str, Any]] = []
    request_count = 0
    chunk_index = 0
    current_path, current_handle = open_next_chunk(args.output_dir, chunk_index)
    current_count = 0

    try:
        for row in tqdm(load_jsonl(args.input), desc="prepare"):
            row_type = row.get("type")
            if row_type == "normal":
                counts[row_type] += 1
                continue
            if row_type not in {"latex", "mixed"}:
                continue

            style = random.choice(LATEX_STYLES) if row_type == "latex" else "natural_mixed"
            request = make_request(
                row=row,
                model=args.model,
                style=style,
                max_output_tokens=args.max_output_tokens,
                structured=not args.no_structured_output,
            )
            current_handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            counts[row_type] += 1
            style_counts[style] += 1
            request_count += 1
            current_count += 1

            if args.limit_requests and request_count >= args.limit_requests:
                break

            if current_count >= args.requests_per_file:
                current_handle.close()
                chunks.append({"path": str(current_path), "requests": current_count})
                chunk_index += 1
                current_path, current_handle = open_next_chunk(args.output_dir, chunk_index)
                current_count = 0
    finally:
        current_handle.close()

    if current_count:
        chunks.append({"path": str(current_path), "requests": current_count})
    elif current_path.exists():
        current_path.unlink()

    manifest = {
        "source_dataset": str(args.input),
        "model": args.model,
        "endpoint": "/v1/responses",
        "request_count": request_count,
        "requests_per_file": args.requests_per_file,
        "counts": dict(counts),
        "latex_style_counts": dict(style_counts),
        "chunks": chunks,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    compact_summary = {
        "source_dataset": manifest["source_dataset"],
        "model": manifest["model"],
        "endpoint": manifest["endpoint"],
        "request_count": manifest["request_count"],
        "chunk_count": len(chunks),
        "requests_per_file": manifest["requests_per_file"],
        "counts": manifest["counts"],
        "input_style_counts": manifest["latex_style_counts"],
        "manifest_path": str(manifest_path),
    }
    print(json.dumps(compact_summary, indent=2))


if __name__ == "__main__":
    main()
