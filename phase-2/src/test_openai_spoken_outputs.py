from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


DEFAULT_REPO_ID = "vibhuiitj/whispermath-raw-output"
DEFAULT_LOCAL_INPUT = Path("data/compiled/raw_output_dataset_balanced_50k.jsonl")
DEFAULT_PROMPT = Path("prompts/output_text_to_spoken_prompt.md")
DEFAULT_MODEL = "gpt-5.4-nano"
TYPES = ("latex", "mixed", "normal")


def get_secret(arg_value: Optional[str], env_names: list[str], prompt: str) -> Optional[str]:
    if arg_value:
        return arg_value
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    return None


def ask_secret(prompt: str) -> str:
    return getpass.getpass(prompt).strip()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_source_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if not args.use_hf:
        yield from load_jsonl(args.local_input)
        return

    hf_token = get_secret(args.hf_token, ["HF_TOKEN", "HUGGINGFACE_TOKEN"], "Hugging Face token: ")
    dataset = load_dataset(
        args.repo_id,
        split=args.split,
        streaming=True,
        token=hf_token,
    )
    for row in dataset:
        yield dict(row)


def sample_rows(rows: Iterable[dict[str, Any]], per_type: int, max_scan: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []

    for scanned, row in enumerate(rows, start=1):
        row_type = row.get("type")
        if row_type in TYPES and counts[row_type] < per_type:
            selected.append(row)
            counts[row_type] += 1

        if all(counts[row_type] >= per_type for row_type in TYPES):
            return selected

        if scanned >= max_scan:
            break

    missing = {row_type: per_type - counts[row_type] for row_type in TYPES if counts[row_type] < per_type}
    if missing:
        raise RuntimeError(f"Could not find enough rows by type. Missing: {missing}")

    return selected


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def generate_spoken_text(client: OpenAI, model: str, system_prompt: str, row: dict[str, Any]) -> dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Convert this dataset output text into one spoken version.\n\n"
                    f"type: {row['type']}\n"
                    f"source_dataset: {row['source_dataset']}\n"
                    "output_text:\n"
                    "```text\n"
                    f"{row['output_text']}\n"
                    "```\n\n"
                    "Return JSON only."
                ),
            },
        ],
    )
    parsed = parse_json_response(response.output_text)
    spoken_text = str(parsed.get("spoken_text", "")).strip()
    if not spoken_text:
        raise ValueError("OpenAI response did not include spoken_text")
    return {
        "spoken_text": spoken_text,
        "notes": str(parsed.get("notes", "")).strip(),
        "raw_response": response.output_text,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], model: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {row_type: [] for row_type in TYPES}
    for row in rows:
        grouped[row["type"]].append(row)

    lines = [
        "# OpenAI Spoken Text Sample",
        "",
        f"Model: `{model}`",
        "",
    ]
    for row_type in TYPES:
        lines.append(f"## {row_type}")
        lines.append("")
        for index, row in enumerate(grouped[row_type], start=1):
            lines.append(f"### {index}. `{row['id']}`")
            lines.append("")
            lines.append("Output text:")
            lines.append("")
            lines.append("```text")
            lines.append(row["output_text"])
            lines.append("```")
            lines.append("")
            lines.append("Spoken text:")
            lines.append("")
            lines.append("```text")
            lines.append(row.get("spoken_text", ""))
            lines.append("```")
            if row.get("notes"):
                lines.append("")
                lines.append(f"Notes: {row['notes']}")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OpenAI spoken-text samples from the raw WhisperMath output dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--local-input", type=Path, default=DEFAULT_LOCAL_INPUT)
    parser.add_argument("--use-hf", action="store_true", help="Sample from Hugging Face instead of the local JSONL file.")
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--max-scan", type=int, default=50_000)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--openai-api-key", help="Optional. Prefer OPENAI_API_KEY env var or hidden prompt.")
    parser.add_argument("--hf-token", help="Optional for private Hugging Face datasets. Prefer HF_TOKEN env var.")
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/generated/openai_spoken_sample_30.jsonl"))
    parser.add_argument("--output-md", type=Path, default=Path("data/generated/openai_spoken_sample_30.md"))
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="Only sample rows; do not call OpenAI.")
    args = parser.parse_args()

    rows = sample_rows(iter_source_rows(args), args.per_type, args.max_scan)
    print(f"Sampled {len(rows)} rows: {dict(Counter(row['type'] for row in rows))}")

    if args.dry_run:
        write_jsonl(args.output_jsonl, rows)
        print(f"Wrote sampled rows without OpenAI generations: {args.output_jsonl}")
        return

    api_key = get_secret(args.openai_api_key, ["OPENAI_API_KEY"], "OpenAI API key: ")
    if not api_key:
        api_key = ask_secret("OpenAI API key: ")

    client = OpenAI(api_key=api_key)
    system_prompt = args.prompt.read_text(encoding="utf-8")

    generated: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="openai"):
        result = generate_spoken_text(client, args.model, system_prompt, row)
        generated.append(
            {
                **row,
                "model": args.model,
                "spoken_text": result["spoken_text"],
                "notes": result["notes"],
            }
        )
        time.sleep(args.sleep)

    write_jsonl(args.output_jsonl, generated)
    write_markdown(args.output_md, generated, args.model)
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Wrote Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
