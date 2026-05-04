from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset
from huggingface_hub import HfApi, login


DEFAULT_REPO_ID = "vibhuiitj/whispermath-raw-output"


def get_token(arg_value: Optional[str]) -> str:
    if arg_value:
        return arg_value
    env_value = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if env_value:
        return env_value
    return getpass.getpass("Hugging Face token: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the raw WhisperMath output dataset JSONL to Hugging Face.")
    parser.add_argument("--input", type=Path, default=Path("data/compiled/raw_output_dataset.jsonl"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--token", help="Optional. Prefer HF_TOKEN env var or hidden prompt.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload WhisperMath raw output dataset")
    args = parser.parse_args()

    token = get_token(args.token)
    login(token=token, add_to_git_credential=False)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    dataset = Dataset.from_json(str(args.input))
    dataset.push_to_hub(
        args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
    )

    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
