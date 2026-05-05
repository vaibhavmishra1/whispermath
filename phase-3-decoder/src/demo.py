from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL = "vibhuiitj/byt5-small-whispermath-a100-checkpoint-2000"


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_latex(
    input_text: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_source_length,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=num_beams > 1,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive spoken-math text to LaTeX demo.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local path or Hugging Face model id.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu.")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.25)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Loading {args.model} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()
    print("Ready. Type spoken math text. Use :q, quit, or exit to stop.\n", flush=True)

    interactive = sys.stdin.isatty()
    while True:
        try:
            input_text = input("spoken> " if interactive else "")
        except EOFError:
            break

        input_text = input_text.strip()
        if not input_text:
            continue
        if input_text.lower() in {":q", "quit", "exit"}:
            break

        output_text = generate_latex(
            input_text=input_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        print(f"latex>  {output_text}\n", flush=True)


if __name__ == "__main__":
    main()
