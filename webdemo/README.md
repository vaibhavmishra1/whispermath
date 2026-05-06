---
title: WhisperMath
emoji: 🧮
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# WhisperMath Web Demo

WhisperMath is an interactive demo for converting spoken mathematical phrases into rendered math notation.

```text
browser microphone
  -> faster-whisper transcript
  -> ByT5 math decoder
  -> rendered KaTeX output
```

The demo has two useful modes:

- **Record audio**: speak a math expression in the browser.
- **Edit transcript**: correct Whisper's transcript and click **Decode Transcript** to test only the ByT5 decoder.

This separation is important because spoken-math errors can come from two different places:

- Whisper may hear the audio incorrectly.
- The ByT5 decoder may convert a correct transcript incorrectly.

## Live Demo

Hugging Face Space:

```text
https://huggingface.co/spaces/vibhuiitj/whispermath-webdemo
```

Direct app URL:

```text
https://vibhuiitj-whispermath-webdemo.hf.space
```

The public Space is configured for free CPU:

```text
Whisper: small.en
Decoder: vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724
Device: CPU
```

## Files

```text
webdemo/
  app.py              FastAPI backend
  static/index.html   Browser UI with recorder and KaTeX rendering
  requirements.txt    Python dependencies
  Dockerfile          Hugging Face Spaces Docker image
  .dockerignore       Files ignored by Docker build
  README.md           Space metadata and this guide
```

## How It Works

1. The browser records audio using `MediaRecorder`.
2. The frontend uploads the recording to `POST /api/transcribe`.
3. The backend saves the audio to a temporary file.
4. `faster-whisper` transcribes the audio into English text.
5. The transcript is passed to the ByT5 checkpoint.
6. The ByT5 output is returned as raw math/LaTeX-like text.
7. The frontend renders the output using KaTeX and also shows the raw model output for debugging.

## Local Setup

From this folder:

```bash
cd /Users/vaibhav/Desktop/beyond/whispermath/webdemo
```

You can use the existing Phase 3 virtualenv:

```bash
/Users/vaibhav/Desktop/beyond/whispermath/phase-3-decoder/.venv/bin/python -m pip install -r requirements.txt
```

Or create a fresh virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Locally

CPU-friendly default:

```bash
uvicorn app:app --host 127.0.0.1 --port 8766
```

Then open:

```text
http://127.0.0.1:8766
```

If using the Phase 3 virtualenv directly:

```bash
/Users/vaibhav/Desktop/beyond/whispermath/phase-3-decoder/.venv/bin/python \
  -m uvicorn app:app --host 127.0.0.1 --port 8766
```

## Model Configuration

The app is controlled with environment variables.

```bash
export WHISPERMATH_WHISPER_MODEL=small.en
export WHISPERMATH_WHISPER_DEVICE=cpu
export WHISPERMATH_WHISPER_COMPUTE_TYPE=int8
export WHISPERMATH_DECODER_MODEL=vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724
export WHISPERMATH_DECODER_DEVICE=auto
```

Whisper model options:

```text
tiny.en    Fastest, weakest transcription
base.en    Better quality, still fairly light
small.en   Good CPU default for the public Space
medium.en  Better transcription, slower and heavier
```

For local testing with medium Whisper:

```bash
WHISPERMATH_WHISPER_MODEL=medium.en \
/Users/vaibhav/Desktop/beyond/whispermath/phase-3-decoder/.venv/bin/python \
  -m uvicorn app:app --host 127.0.0.1 --port 8766
```

For the free Hugging Face CPU Space, keep:

```bash
WHISPERMATH_WHISPER_MODEL=small.en
WHISPERMATH_DECODER_DEVICE=cpu
```

## API Endpoints

### Health

```bash
curl http://127.0.0.1:8766/api/health
```

Example:

```json
{
  "status": "ok",
  "whisper_model": "small.en",
  "decoder_model": "vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724",
  "decoder_device": "cpu"
}
```

### Decode Text Only

Use this when you want to test the ByT5 decoder without audio:

```bash
curl -X POST http://127.0.0.1:8766/api/decode \
  -H "Content-Type: application/json" \
  -d '{"text":"x squared minus y squared equals four","num_beams":1,"max_new_tokens":128}'
```

Example response:

```json
{
  "transcript": "x squared minus y squared equals four",
  "math_text": "x^2-y^2=4",
  "decoder_model": "vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724"
}
```

### Transcribe Audio

```bash
curl -X POST http://127.0.0.1:8766/api/transcribe \
  -F audio=@/path/to/audio.wav \
  -F num_beams=4 \
  -F max_new_tokens=256
```

Example response:

```json
{
  "transcript": "integral from zero to pi of sine x dx.",
  "math_text": "\\int_0^\\pi \\sin x dx.",
  "segments": [
    {
      "start": 0.0,
      "end": 2.72,
      "text": "integral from zero to pi of sine x dx."
    }
  ],
  "whisper_model": "small.en",
  "decoder_model": "vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724"
}
```

## Deploy To Hugging Face Spaces

This folder is ready for a Docker Space.

### 1. Login

Do not commit or paste tokens into files.

```bash
huggingface-cli login
```

Or use the Python API with your local authenticated session.

### 2. Create The Space

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="vibhuiitj/whispermath-webdemo",
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
)
```

### 3. Upload The Folder

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id="vibhuiitj/whispermath-webdemo",
    repo_type="space",
    folder_path="/Users/vaibhav/Desktop/beyond/whispermath/webdemo",
    path_in_repo=".",
    ignore_patterns=[
        "__pycache__/*",
        "*.pyc",
        ".DS_Store",
        ".venv/*",
        "audio/*",
        "outputs/*",
    ],
    commit_message="Deploy WhisperMath web demo",
)
```

### 4. Check Runtime

```python
from huggingface_hub import HfApi

runtime = HfApi().get_space_runtime("vibhuiitj/whispermath-webdemo")
print(runtime)
```

Expected final state:

```text
stage='RUNNING'
hardware='cpu-basic'
requested_hardware='cpu-basic'
```

### 5. Test The Deployed Space

Health:

```bash
curl https://vibhuiitj-whispermath-webdemo.hf.space/api/health
```

Text decode:

```bash
curl -L -X POST https://vibhuiitj-whispermath-webdemo.hf.space/api/decode \
  -H "Content-Type: application/json" \
  -d '{"text":"x squared minus y squared equals four","num_beams":1,"max_new_tokens":128}'
```

Expected:

```json
{
  "transcript": "x squared minus y squared equals four",
  "math_text": "x^2-y^2=4",
  "decoder_model": "vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724"
}
```

## Docker Notes

The Docker image:

- Uses `python:3.11-slim`
- Installs `libgomp1`, needed by some CPU inference dependencies
- Runs on port `7860`, the Hugging Face Spaces default
- Sets `WHISPERMATH_WHISPER_MODEL=small.en`
- Sets `WHISPERMATH_DECODER_DEVICE=cpu`
- Sets `HF_HUB_DISABLE_XET=1`

`HF_HUB_DISABLE_XET=1` is included because local testing showed large model downloads could get stuck with incomplete Xet-backed cache files.

## Troubleshooting

### Space Takes A Long Time To Start

The first start downloads:

- `Systran/faster-whisper-small.en`
- `vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724`

On free CPU, startup can take a few minutes.

### Audio Transcription Is Wrong

Try these in order:

1. Speak shorter phrases.
2. Use clearer operator words, for example `over` instead of `by`.
3. Check the editable transcript box.
4. Correct the transcript manually.
5. Click **Decode Transcript**.
6. Try `base.en`, `small.en`, or `medium.en` locally.

### ByT5 Output Is Wrong But Transcript Is Correct

That means the decoder needs more targeted training data. Common weak phrases include:

```text
by / divided by / over
whole square
derivative of ...
limit as ...
fraction with grouped numerator and denominator
```

Use the editable transcript box to collect failure cases.

### KaTeX Rendering Fails

The app still shows the raw ByT5 output under **Raw ByT5 Output**. If the raw output is malformed LaTeX-like text, KaTeX may render an error-colored expression or show fallback text.

### Medium Whisper Hangs During Download

If a local download leaves an incomplete cache file, run:

```bash
HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-medium.en", max_workers=1)
PY
```

Then restart:

```bash
HF_HUB_DISABLE_XET=1 WHISPERMATH_WHISPER_MODEL=medium.en \
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

## Security

Never commit Hugging Face tokens or API keys into this folder.

If a token is pasted into a chat or terminal history by mistake, revoke/rotate it from Hugging Face settings.
