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


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload request JSONL files and submit OpenAI Batch API jobs.")
    parser.add_argument("--manifest", type=Path, default=Path("data/batches/input_text_requests/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/batches/input_text_requests/submitted_batches.json"))
    parser.add_argument("--api-key", help="Optional. Prefer OPENAI_API_KEY env var.")
    parser.add_argument("--description", default="WhisperMath input_text generation")
    args = parser.parse_args()

    client = OpenAI(api_key=get_api_key(args.api_key))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    submitted = []
    for chunk in tqdm(manifest["chunks"], desc="submit"):
        path = Path(chunk["path"])
        with path.open("rb") as handle:
            uploaded_file = client.files.create(file=handle, purpose="batch")

        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint=manifest["endpoint"],
            completion_window="24h",
            metadata={
                "description": args.description,
                "request_file": path.name,
                "model": manifest["model"],
            },
        )
        submitted.append(
            {
                "request_file": str(path),
                "requests": chunk["requests"],
                "input_file_id": uploaded_file.id,
                "batch_id": batch.id,
                "status": batch.status,
            }
        )

    output = {**manifest, "submitted_batches": submitted}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote submitted batch manifest: {args.output}")


if __name__ == "__main__":
    main()
