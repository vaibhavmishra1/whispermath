from __future__ import annotations

import json
from pathlib import Path

from math_parser import parse_math_text


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "examples" / "text_cases.jsonl"


def main() -> None:
    failures = 0

    for line_number, line in enumerate(CASES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        case = json.loads(line)
        result = parse_math_text(case["spoken"])
        expected = case["latex"]

        if result.latex != expected:
            failures += 1
            print(f"FAIL line {line_number}")
            print(f"  spoken:   {case['spoken']}")
            print(f"  expected: {expected}")
            print(f"  got:      {result.latex}")
        else:
            print(f"PASS {case['spoken']} -> {result.latex}")

    if failures:
        raise SystemExit(f"{failures} parser smoke test(s) failed")

    print("All parser smoke tests passed.")


if __name__ == "__main__":
    main()
