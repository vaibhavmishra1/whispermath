from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


def get_api_key(arg_value: Optional[str]) -> str:
    if arg_value:
        return arg_value
    env_value = os.getenv("OPENAI_API_KEY")
    if env_value:
        return env_value
    return getpass.getpass("OpenAI API key: ").strip()


def write_file_content(client: OpenAI, file_id: str, output_path: Path) -> None:
    content = client.files.content(file_id)
    data = content.read()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download completed OpenAI Batch API output/error files.")
    parser.add_argument("--submitted", type=Path, default=Path("data/batches/input_text_requests/submitted_batches.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/batches/input_text_outputs"))
    parser.add_argument("--api-key", help="Optional. Prefer OPENAI_API_KEY env var.")
    args = parser.parse_args()

    client = OpenAI(api_key=get_api_key(args.api_key))
    manifest = json.loads(args.submitted.read_text(encoding="utf-8"))

    downloaded = []
    for item in tqdm(manifest["submitted_batches"], desc="download"):
        batch = client.batches.retrieve(item["batch_id"])
        if batch.status != "completed":
            print(f"Skipping {batch.id}: status={batch.status}")
            continue

        output_path = args.output_dir / f"{batch.id}.output.jsonl"
        error_path = args.output_dir / f"{batch.id}.errors.jsonl"

        if batch.output_file_id:
            write_file_content(client, batch.output_file_id, output_path)
        if batch.error_file_id:
            write_file_content(client, batch.error_file_id, error_path)

        downloaded.append(
            {
                "batch_id": batch.id,
                "output_file_id": batch.output_file_id,
                "output_path": str(output_path) if batch.output_file_id else None,
                "error_file_id": batch.error_file_id,
                "error_path": str(error_path) if batch.error_file_id else None,
            }
        )

    manifest["downloaded_outputs"] = downloaded
    args.submitted.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(downloaded)} completed batch outputs.")


if __name__ == "__main__":
    main()
