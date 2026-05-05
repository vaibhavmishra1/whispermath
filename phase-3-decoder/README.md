# WhisperMath Phase 3: Decoder Fine-Tuning

This phase trains a text-to-text decoder:

```text
input_text -> output_text
```

Dataset:

```text
vibhuiitj/whispermath-input-output
```

Model:

```text
google/byt5-small first, then google/byt5-base for the larger run
```

ByT5 is a good first model because it is byte-level, so LaTeX characters like `\`, `{`, `}`, `_`, and `^` do not need a special tokenizer.

## Setup

```bash
cd /Users/vaibhav/Desktop/beyond/whispermath/phase-3-decoder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke Train

Run a tiny training job first:

```bash
python src/train_byt5.py \
  --max-train-samples 1000 \
  --max-eval-samples 100 \
  --num-train-epochs 0.1 \
  --eval-steps 50 \
  --save-steps 50 \
  --output-dir models/byt5-smoke
```

Then test:

```bash
python src/predict.py \
  --model models/byt5-smoke \
  "x squared minus y squared equals four"
```

## Full First Run

```bash
python src/train_byt5.py --config configs/byt5_small.yaml
```

The default config is conservative for an Apple Silicon laptop:

```text
batch size: 1
gradient accumulation: 16
max source length: 512
max target length: 512
epochs: 1
```

## Full CUDA Run

On a CUDA GPU machine, install a CUDA-enabled PyTorch build, then run:

```bash
cd /Users/vaibhav/Desktop/beyond/whispermath/phase-3-decoder
source .venv/bin/activate
python src/check_gpu.py
python src/train_byt5.py --config configs/byt5_small_cuda_full.yaml
```

The CUDA config uses:

```text
epochs: 2
per-device train batch size: 4
gradient accumulation: 4
effective batch size: 16
fp16: true
max source/target length: 512
```

If the GPU runs out of memory:

```bash
python src/train_byt5.py \
  --config configs/byt5_small_cuda_full.yaml \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing
```

If your GPU supports bf16 well, prefer:

```bash
python src/train_byt5.py \
  --config configs/byt5_small_cuda_full.yaml \
  --bf16
```

## A100 80GB Run

For a single A100 80GB, use the dedicated config:

```bash
python src/train_byt5.py --config configs/byt5_small_a100_80gb.yaml
```

This uses:

```text
epochs: 3
learning rate: 1e-4
bf16: true
per-device batch size: 16
gradient accumulation: 2
effective batch size: 32
max source/target length: 512
```

If memory usage is low, try:

```bash
python src/train_byt5.py \
  --config configs/byt5_small_a100_80gb.yaml \
  --per-device-train-batch-size 32 \
  --gradient-accumulation-steps 1
```

## A100 80GB Bigger Run

If `byt5-small` saturates but still fails real prompts, run the same dataset with `google/byt5-base`:

```bash
cd /Users/vaibhav/Desktop/beyond/whispermath/phase-3-decoder
source .venv/bin/activate
python src/check_gpu.py
python src/train_byt5.py --config configs/byt5_base_a100_80gb.yaml
```

This uses:

```text
model: google/byt5-base
epochs: 3
learning rate: 5e-5
bf16: true
per-device batch size: 8
gradient accumulation: 4
effective batch size: 32
max source/target length: 512
```

If it runs out of memory:

```bash
python src/train_byt5.py \
  --config configs/byt5_base_a100_80gb.yaml \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing
```

If memory usage is low, try:

```bash
python src/train_byt5.py \
  --config configs/byt5_base_a100_80gb.yaml \
  --per-device-train-batch-size 16 \
  --gradient-accumulation-steps 2
```

## Evaluate

```bash
python src/evaluate.py \
  --model models/byt5-small-whispermath \
  --max-samples 300
```

Outputs:

```text
outputs/eval_predictions.jsonl
outputs/eval_summary.json
```

Metrics are grouped by:

```text
latex
mixed
normal
```

## Manual Predictions

```bash
python src/predict.py \
  --model models/byt5-small-whispermath \
  "can you solve x squared plus three x plus two equals zero"
```

Expected style:

```text
can you solve x^2 + 3x + 2 = 0
```

## Interactive Demo

Run the uploaded checkpoint and keep typing spoken-math text:

```bash
cd /Users/vaibhav/Desktop/beyond/whispermath/phase-3-decoder
source .venv/bin/activate
python src/demo.py \
  --model vibhuiitj/byt5-small-whispermath-a100-checkpoint-2000
```

The demo defaults to beam search plus repetition controls so failed generations are less likely to get stuck repeating one token.

Example:

```text
spoken> x squared minus y squared equals four
latex>  x^2-y^2=4

spoken> show me the next step
latex>  Show me the next step
```

Use `:q`, `quit`, or `exit` to stop the demo.
