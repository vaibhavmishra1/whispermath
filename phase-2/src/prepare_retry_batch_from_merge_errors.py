from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prepare_input_text_batches import LATEX_STYLES, make_request


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create retry OpenAI Batch requests for rows that failed merge parsing.")
    parser.add_argument("--source", type=Path, default=Path("data/compiled/raw_output_dataset_balanced_50k.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("data/compiled/input_output_dataset_150k.merge_errors.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/batches/input_text_retry_requests/requests_retry.jsonl"))
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args()

    failed_ids = {row["custom_id"] for row in load_jsonl(args.errors) if row.get("custom_id")}
    source_rows: dict[str, dict[str, Any]] = {row["id"]: row for row in load_jsonl(args.source) if row.get("id") in failed_ids}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for custom_id in tqdm(sorted(failed_ids), desc="retry"):
            row = source_rows.get(custom_id)
            if not row or row.get("type") == "normal":
                continue
            style = LATEX_STYLES[hash(custom_id) % len(LATEX_STYLES)] if row["type"] == "latex" else "natural_mixed"
            request = make_request(
                row=row,
                model=args.model,
                style=style,
                max_output_tokens=args.max_output_tokens,
                structured=True,
            )
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1

    manifest = {
        "source": str(args.source),
        "errors": str(args.errors),
        "output": str(args.output),
        "retry_request_count": count,
    }
    manifest_path = args.output.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
