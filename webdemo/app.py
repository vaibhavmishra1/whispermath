from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

DEFAULT_WHISPER_MODEL = os.getenv("WHISPERMATH_WHISPER_MODEL", "small.en")
DEFAULT_WHISPER_DEVICE = os.getenv("WHISPERMATH_WHISPER_DEVICE", "cpu")
DEFAULT_WHISPER_COMPUTE_TYPE = os.getenv("WHISPERMATH_WHISPER_COMPUTE_TYPE", "int8")
DEFAULT_DECODER_MODEL = os.getenv(
    "WHISPERMATH_DECODER_MODEL",
    "vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724",
)


def select_decoder_device(device_name: str | None = None) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DemoModels:
    whisper: WhisperModel
    tokenizer: Any
    decoder: Any
    decoder_device: torch.device


class DecodeRequest(BaseModel):
    text: str
    num_beams: int = 4
    max_new_tokens: int = 256


def load_models() -> DemoModels:
    decoder_device = select_decoder_device(os.getenv("WHISPERMATH_DECODER_DEVICE", "auto"))
    print(
        f"Loading Whisper {DEFAULT_WHISPER_MODEL} "
        f"({DEFAULT_WHISPER_DEVICE}, {DEFAULT_WHISPER_COMPUTE_TYPE})...",
        flush=True,
    )
    whisper = WhisperModel(
        DEFAULT_WHISPER_MODEL,
        device=DEFAULT_WHISPER_DEVICE,
        compute_type=DEFAULT_WHISPER_COMPUTE_TYPE,
    )

    print(f"Loading decoder {DEFAULT_DECODER_MODEL} on {decoder_device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_DECODER_MODEL)
    decoder = AutoModelForSeq2SeqLM.from_pretrained(
        DEFAULT_DECODER_MODEL,
        low_cpu_mem_usage=True,
    ).to(decoder_device)
    decoder.eval()
    print("WhisperMath web demo is ready.", flush=True)

    return DemoModels(
        whisper=whisper,
        tokenizer=tokenizer,
        decoder=decoder,
        decoder_device=decoder_device,
    )


models: DemoModels | None = None

app = FastAPI(title="WhisperMath Web Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    global models
    models = load_models()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    decoder_device = str(models.decoder_device) if models else "not_loaded"
    return {
        "status": "ok" if models else "loading",
        "whisper_model": DEFAULT_WHISPER_MODEL,
        "decoder_model": DEFAULT_DECODER_MODEL,
        "decoder_device": decoder_device,
    }


def transcribe_audio(audio_path: Path) -> tuple[str, list[dict[str, float | str]]]:
    if models is None:
        raise RuntimeError("Models are still loading.")

    segments, _info = models.whisper.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,
    )
    segment_rows = [
        {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        }
        for segment in segments
    ]
    transcript = " ".join(row["text"] for row in segment_rows).strip()
    return transcript, segment_rows


def decode_math_text(
    transcript: str,
    max_source_length: int = 512,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    repetition_penalty: float = 1.25,
    no_repeat_ngram_size: int = 4,
) -> str:
    if models is None:
        raise RuntimeError("Models are still loading.")
    if not transcript:
        return ""

    encoded = models.tokenizer(
        transcript,
        return_tensors="pt",
        max_length=max_source_length,
        truncation=True,
    ).to(models.decoder_device)

    with torch.no_grad():
        output_ids = models.decoder.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=num_beams > 1,
        )
    return models.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def clamp_generation_args(num_beams: int, max_new_tokens: int) -> tuple[int, int]:
    return max(1, min(num_beams, 8)), max(32, min(max_new_tokens, 1024))


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    num_beams: int = Form(4),
    max_new_tokens: int = Form(256),
) -> dict[str, Any]:
    if models is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")

    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)
        handle.write(await audio.read())

    try:
        transcript, segments = transcribe_audio(temp_path)
        safe_num_beams, safe_max_new_tokens = clamp_generation_args(num_beams, max_new_tokens)
        math_text = decode_math_text(
            transcript,
            num_beams=safe_num_beams,
            max_new_tokens=safe_max_new_tokens,
        )
    except Exception as exc:  # pragma: no cover - returned to the demo UI.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "transcript": transcript,
        "math_text": math_text,
        "segments": segments,
        "whisper_model": DEFAULT_WHISPER_MODEL,
        "decoder_model": DEFAULT_DECODER_MODEL,
    }


@app.post("/api/decode")
def decode(request: DecodeRequest) -> dict[str, Any]:
    if models is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")

    transcript = request.text.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Text is required.")

    safe_num_beams, safe_max_new_tokens = clamp_generation_args(
        request.num_beams,
        request.max_new_tokens,
    )
    math_text = decode_math_text(
        transcript,
        num_beams=safe_num_beams,
        max_new_tokens=safe_max_new_tokens,
    )
    return {
        "transcript": transcript,
        "math_text": math_text,
        "decoder_model": DEFAULT_DECODER_MODEL,
    }
