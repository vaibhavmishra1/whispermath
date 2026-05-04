from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print generated spoken math -> LaTeX pairs.")
    parser.add_argument("--input", type=Path, default=Path("data/generated/spoken_latex_pairs.jsonl"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    shown = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        print(f"spoken: {row['spoken']}")
        print(f"latex:  {row['latex']}")
        print(f"style:  {row.get('style', '')}")
        print()
        shown += 1
        if shown >= args.limit:
            break


if __name__ == "__main__":
    main()
