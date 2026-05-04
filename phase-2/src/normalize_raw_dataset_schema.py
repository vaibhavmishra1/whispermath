from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize compiled raw dataset JSONL to a stable Hugging Face schema.")
    parser.add_argument("--input", type=Path, default=Path("data/compiled/raw_output_dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/compiled/raw_output_dataset.normalized.jsonl"))
    parser.add_argument("--replace", action="store_true", help="Replace input with normalized output after writing.")
    args = parser.parse_args()

    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in tqdm(source, desc="normalize"):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.pop("metadata", None)
            if "metadata_json" not in row:
                row["metadata_json"] = json.dumps(metadata or {}, ensure_ascii=False, default=str)
            target.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1

    if args.replace:
        args.output.replace(args.input)

    print(f"Normalized {count} rows")
    print(f"Output: {args.input if args.replace else args.output}")


if __name__ == "__main__":
    main()
