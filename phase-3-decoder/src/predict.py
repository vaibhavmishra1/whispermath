from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-off predictions with a WhisperMath decoder model.")
    parser.add_argument("text", nargs="*", help="Input text to normalize.")
    parser.add_argument("--model", default="models/byt5-small-whispermath")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.25)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    args = parser.parse_args()

    input_text = " ".join(args.text).strip()
    if not input_text:
        input_text = "x squared minus y squared equals four"

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=args.max_source_length,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            early_stopping=args.num_beams > 1,
        )

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
