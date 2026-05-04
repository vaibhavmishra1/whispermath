from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced raw dataset by limiting rows per type.")
    parser.add_argument("--input", type=Path, default=Path("data/compiled/raw_output_dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/compiled/raw_output_dataset_balanced_50k.jsonl"))
    parser.add_argument("--per-type", type=int, default=50_000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    selected: list[dict] = []

    with args.input.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="select"):
            if not line.strip():
                continue
            row = json.loads(line)
            row_type = row["type"]
            if counts[row_type] >= args.per_type:
                continue
            selected.append(row)
            counts[row_type] += 1

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in tqdm(selected, desc="write"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "per_type": args.per_type,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "counts": dict(counts),
        "total": sum(counts.values()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
