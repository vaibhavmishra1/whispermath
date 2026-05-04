from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from faster_whisper import WhisperModel


DEFAULT_MODEL = "tiny.en"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


def transcribe_audio(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> dict:
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=False,
    )
    segment_list = [
        {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        }
        for segment in segments
    ]
    transcript = " ".join(segment["text"] for segment in segment_list).strip()

    return {
        "audio": str(audio_path),
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": info.language,
        "language_probability": info.language_probability,
        "transcript": transcript,
        "segments": segment_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one math audio file with faster-whisper.")
    parser.add_argument("audio", type=Path, help="Path to an audio file.")
    parser.add_argument("--model", default=os.getenv("WHISPERMATH_MODEL", DEFAULT_MODEL))
    parser.add_argument("--device", default=os.getenv("WHISPERMATH_DEVICE", DEFAULT_DEVICE))
    parser.add_argument("--compute-type", default=os.getenv("WHISPERMATH_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE))
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = parser.parse_args()

    result = transcribe_audio(args.audio, args.model, args.device, args.compute_type)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["transcript"])


if __name__ == "__main__":
    main()
