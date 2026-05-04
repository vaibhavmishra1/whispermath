from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import Image, load_dataset
from tqdm import tqdm


DEFAULT_DATASET = "OleehyO/latex-formulas"
DEFAULT_CONFIG = "cleaned_formulas"
DEFAULT_SPLIT = "train"
DEFAULT_COLUMN = "latex_formula"


BLOCKED_PATTERNS = [
    r"\\begin\{array\}",
    r"\\begin\{matrix\}",
    r"\\begin\{pmatrix\}",
    r"\\begin\{bmatrix\}",
    r"\\mbox",
    r"\\textrm",
    r"\\mathrm\{[A-Za-z ]{8,}\}",
    r"\\vec",
    r"\\mathbb",
    r"\\cal",
    r"\\operatorname",
    r"\\partial",
    r"\\nabla",
    r"\\cdots",
    r"\\ldots",
    r"\.\.\.",
    r"[,;]",
]

ALLOWED_COMMANDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "lambda",
    "mu",
    "nu",
    "pi",
    "rho",
    "sigma",
    "phi",
    "omega",
    "Delta",
    "Phi",
    "Omega",
    "frac",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "log",
    "ln",
    "lim",
    "int",
    "sum",
    "pm",
    "times",
    "cdot",
    "leq",
    "geq",
    "neq",
}


def clean_latex(latex: str) -> str:
    latex = latex.strip()
    latex = re.sub(r"^\\begin\{align\*\}", "", latex)
    latex = re.sub(r"\\end\{align\*\}$", "", latex)
    latex = re.sub(r"^\\begin\{equation\*\}", "", latex)
    latex = re.sub(r"\\end\{equation\*\}$", "", latex)
    latex = latex.replace(r"\displaystyle", "")
    latex = re.sub(r"\\left\b", "", latex)
    latex = re.sub(r"\\right\b", "", latex)
    latex = latex.replace(r"\,", " ")
    latex = latex.replace(r"\;", " ")
    latex = latex.replace(r"\ ", " ")
    latex = re.sub(r"\{\\rm\s+d\}", "d", latex)
    latex = re.sub(r"\s+", " ", latex).strip()
    return latex


def is_candidate(latex: str, min_chars: int, max_chars: int, profile: str) -> bool:
    if not (min_chars <= len(latex) <= max_chars):
        return False
    if any(re.search(pattern, latex) for pattern in BLOCKED_PATTERNS):
        return False
    if latex.count("{") != latex.count("}"):
        return False
    if latex.count("\\") > 18:
        return False
    if profile == "simple":
        commands = set(re.findall(r"\\([A-Za-z]+)", latex))
        if commands - ALLOWED_COMMANDS:
            return False
        if latex.count("_") > 4:
            return False
        if latex.count("^") > 6:
            return False
    return True


def iter_latex_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )
    if "image" in dataset.features:
        dataset = dataset.cast_column("image", Image(decode=False))

    seen: set[str] = set()
    for row_index, row in enumerate(dataset):
        original_latex = str(row.get(args.column, ""))
        latex = clean_latex(original_latex)
        if not latex or latex in seen:
            continue
        seen.add(latex)

        if not is_candidate(latex, args.min_chars, args.max_chars, args.profile):
            continue

        if args.keep_probability < 1.0 and random.random() > args.keep_probability:
            continue

        yield {
            "source_dataset": args.dataset,
            "source_config": args.config,
            "source_split": args.split,
            "source_column": args.column,
            "source_row_index": row_index,
            "original_latex": original_latex,
            "latex": latex,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample usable LaTeX expressions from a Hugging Face dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--column", default=DEFAULT_COLUMN)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--profile", choices=["simple", "broad"], default="simple")
    parser.add_argument("--keep-probability", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/raw/latex_samples.jsonl"))
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for item in tqdm(iter_latex_rows(args), total=args.limit):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.limit:
                break

    print(f"Wrote {count} LaTeX samples to {args.output}")


if __name__ == "__main__":
    main()
