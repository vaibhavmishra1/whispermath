from __future__ import annotations

import argparse
import getpass
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_PROMPT_PATH = Path("prompts/latex_verbalizer_prompt.md")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def response_to_json(response: Any) -> dict[str, Any]:
    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return json.loads(text)


def verbalize_latex(client: OpenAI, model: str, system_prompt: str, latex: str) -> dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Latex:\n\n```latex\n{latex}\n```\n\nReturn JSON only.",
            },
        ],
    )
    return response_to_json(response)


def get_api_key(arg_value: Optional[str]) -> str:
    if arg_value:
        return arg_value
    env_value = os.getenv("OPENAI_API_KEY")
    if env_value:
        return env_value
    return getpass.getpass("OpenAI API key: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate spoken-math variants from LaTeX with OpenAI.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/latex_samples.jsonl"))
    parser.add_argument("--raw-output", type=Path, default=Path("data/generated/verbalized_raw.jsonl"))
    parser.add_argument("--pairs-output", type=Path, default=Path("data/generated/spoken_latex_pairs.jsonl"))
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", help="Optional. Prefer OPENAI_API_KEY or hidden prompt to avoid shell history.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    client = OpenAI(api_key=api_key)
    system_prompt = args.prompt.read_text(encoding="utf-8")
    source_rows = load_jsonl(args.input)[: args.limit]

    completed_latex: set[str] = set()
    if args.resume and args.raw_output.exists():
        for row in load_jsonl(args.raw_output):
            completed_latex.add(row.get("latex", ""))

    generated = 0
    for source in tqdm(source_rows):
        latex = source["latex"]
        if latex in completed_latex:
            continue

        try:
            result = verbalize_latex(client, args.model, system_prompt, latex)
            raw_item = {
                "latex": latex,
                "model": args.model,
                "source": source,
                "result": result,
            }
            append_jsonl(args.raw_output, raw_item)

            quality = result.get("quality", {})
            if quality.get("usable") is True:
                for variant_index, variant in enumerate(result.get("variants", [])):
                    spoken = str(variant.get("spoken", "")).strip()
                    if not spoken:
                        continue
                    append_jsonl(
                        args.pairs_output,
                        {
                            "spoken": spoken,
                            "latex": latex,
                            "style": variant.get("style", "unknown"),
                            "notes": variant.get("notes", ""),
                            "variant_index": variant_index,
                            "source_dataset": source.get("source_dataset"),
                            "source_row_index": source.get("source_row_index"),
                        },
                    )

            generated += 1
            time.sleep(args.sleep)
        except Exception as exc:
            append_jsonl(
                args.raw_output,
                {
                    "latex": latex,
                    "model": args.model,
                    "source": source,
                    "error": str(exc),
                },
            )

    print(f"Processed {generated} LaTeX expressions.")
    print(f"Raw generations: {args.raw_output}")
    print(f"Training pairs: {args.pairs_output}")


if __name__ == "__main__":
    main()
