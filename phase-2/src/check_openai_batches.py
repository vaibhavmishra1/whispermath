from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI


def get_api_key(arg_value: Optional[str]) -> str:
    if arg_value:
        return arg_value
    env_value = os.getenv("OPENAI_API_KEY")
    if env_value:
        return env_value
    return getpass.getpass("OpenAI API key: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OpenAI Batch API job statuses.")
    parser.add_argument("--submitted", type=Path, default=Path("data/batches/input_text_requests/submitted_batches.json"))
    parser.add_argument("--api-key", help="Optional. Prefer OPENAI_API_KEY env var.")
    args = parser.parse_args()

    client = OpenAI(api_key=get_api_key(args.api_key))
    manifest = json.loads(args.submitted.read_text(encoding="utf-8"))

    statuses = []
    for item in manifest["submitted_batches"]:
        batch = client.batches.retrieve(item["batch_id"])
        status = {
            **item,
            "status": batch.status,
            "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
        statuses.append(status)
        print(json.dumps(status, indent=2))

    manifest["submitted_batches"] = statuses
    args.submitted.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
