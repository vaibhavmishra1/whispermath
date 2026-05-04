from __future__ import annotations

import argparse
import json
from pathlib import Path

from math_parser import parse_math_text
from transcribe_audio import transcribe_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 audio -> transcript -> LaTeX experiment.")
    parser.add_argument("audio", type=Path, help="Path to a recorded math audio file.")
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    transcription = transcribe_audio(args.audio, args.model, args.device, args.compute_type)
    parsed = parse_math_text(transcription["transcript"])

    result = {
        "audio": str(args.audio),
        "model": transcription["model"],
        "device": transcription["device"],
        "compute_type": transcription["compute_type"],
        "transcript": transcription["transcript"],
        "normalized_transcript": parsed.normalized,
        "latex": parsed.latex,
        "warnings": parsed.warnings,
        "segments": transcription["segments"],
    }

    print(f"Transcript: {result['transcript']}")
    print(f"LaTeX: {result['latex']}")
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
