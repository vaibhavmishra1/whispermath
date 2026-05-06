# WhisperMath

Speech-to-math demo using Whisper for transcription and a fine-tuned ByT5 decoder for math notation.

## Links

- Model: `vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724`
- Training data: `vibhuiitj/whispermath-input-output`
- Web demo Space: https://huggingface.co/spaces/vibhuiitj/whispermath-webdemo
- Direct demo: https://vibhuiitj-whispermath-webdemo.hf.space

## What It Does

```text
audio
  -> faster-whisper
  -> spoken transcript
  -> fine-tuned ByT5
  -> LaTeX-like math text
  -> KaTeX render
```

Example:

```text
audio/transcript: integral from zero to pi of sine x dx
model output:     \int_0^\pi \sin x dx
```

Whisper is not fine-tuned here. The fine-tuned model is ByT5, trained for:

```text
spoken math text -> math / LaTeX-like text
```

## Repo Layout

```text
phase-1/           Whisper transcription + early rule parser
phase-2/           Dataset construction
phase-3-decoder/   ByT5 training, eval, prediction scripts
webdemo/           FastAPI + browser demo, deployed to HF Spaces
```

## Dataset

Final dataset:

```text
vibhuiitj/whispermath-input-output
```

Schema:

```json
{
  "input_text": "x squared minus y squared equals four",
  "output_text": "x^2-y^2=4",
  "type": "latex"
}
```

Types:

```text
latex   formula-heavy rows
mixed   natural language with math
normal  normal text copied through as a control task
```

Split used during training/eval:

```text
train:      146,981
validation: 1,500
test:       1,500
```

## Model

Base:

```text
google/byt5-base
```

Fine-tuned checkpoint:

```text
vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724
```

ByT5 was used because it is byte-level and handles LaTeX characters like:

```text
\ { } _ ^
```

without adding custom tokenizer tokens.

## Training

Config:

```text
phase-3-decoder/configs/byt5_base_a100_80gb.yaml
```

Main settings:

```yaml
model_name: google/byt5-base
dataset_id: vibhuiitj/whispermath-input-output
num_train_epochs: 3
learning_rate: 5e-5
max_source_length: 512
max_target_length: 512
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
bf16: true
```

Run:

```bash
cd phase-3-decoder
python src/train_byt5.py --config configs/byt5_base_a100_80gb.yaml
```

## Inference

Text-only:

```bash
cd phase-3-decoder
python src/predict.py \
  --model vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724 \
  "x squared minus y squared equals four"
```

Expected:

```text
x^2-y^2=4
```

Web demo:

```bash
cd webdemo
WHISPERMATH_WHISPER_MODEL=small.en \
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

Open:

```text
http://127.0.0.1:8766
```

Use `medium.en` locally for better Whisper transcription:

```bash
WHISPERMATH_WHISPER_MODEL=medium.en python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

## Results

Evaluation:

```bash
cd phase-3-decoder
python src/evaluate.py \
  --model vibhuiitj/byt5-base-whispermath-a100-checkpoint-10724 \
  --max-samples 150 \
  --num-beams 1
```

Results on 150 held-out rows:

| Type | Count | Exact | CER |
|---|---:|---:|---:|
| normal | 55 | 18.2% | 0.468 |
| latex | 53 | 3.8% | 0.260 |
| mixed | 42 | 0.0% | 0.559 |

Comparable 30-row sample vs earlier smoke checkpoint:

| Type | Old CER | New CER |
|---|---:|---:|
| normal | 0.654 | 0.563 |
| latex | 0.656 | 0.222 |
| mixed | 0.660 | 0.553 |

The biggest improvement is on pure LaTeX/formula rows.

## Examples

Good:

```text
x squared minus y squared equals four
-> x^2-y^2=4
```

```text
integral from zero to pi of sine x dx
-> \int_0^\pi \sin x dx
```

Known weak cases:

```text
limit as x tends to zero of sine x by x
-> \lim_{x\to 0}\sin x\by x
```

```text
the derivative of x cubed plus five x squared minus seven
-> \frac{d}{x^3+5x^2-7}
```

## Notes

- `tiny.en` is too weak for spoken math.
- `small.en` is the best free-CPU default.
- `medium.en` gives better transcripts locally but is slower.
- The decoder still needs more data for fractions, derivatives, limits, grouped expressions, and phrases like `whole square`.
- The web UI includes an editable transcript box so Whisper errors can be corrected before re-decoding with ByT5.
