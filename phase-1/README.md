# WhisperMath Phase 1 Experiment

This folder is the first practical experiment for:

```text
math audio -> Whisper transcript -> LaTeX-style equation text
```

The goal is to quickly learn:

- How well a local Whisper model understands spoken math.
- Which math phrases Whisper gets wrong.
- How much can be solved with a parser before fine-tuning.
- What dataset we need to collect next.

## Chosen Model

Phase 1 uses:

```text
faster-whisper + tiny.en + CPU + int8
```

Reason:

- This device is an Apple M2 Pro.
- `tiny.en` is small and should run locally without GPU setup.
- `faster-whisper` avoids needing a separate system `ffmpeg` install in most cases.
- CPU `int8` is the most reliable first setting.

If transcription quality is weak, try:

```text
base.en
```

or:

```text
small.en
```

Those will be slower but usually more accurate.

## Folder Structure

```text
phase-1/
  audio/                  # Put your recorded audio files here
  outputs/                # JSON experiment outputs
  examples/text_cases.jsonl
  src/math_parser.py
  src/transcribe_audio.py
  src/run_audio.py
  src/smoke_test_parser.py
  requirements.txt
```

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model will be downloaded the first time you run transcription.

## Step 1: Test The Parser Without Audio

```bash
python src/smoke_test_parser.py
```

You can also try one phrase:

```bash
python src/math_parser.py "x squared minus y squared is equal to four"
```

Expected:

```text
x^2 - y^2 = 4
```

## Step 2: Add Your Audio

Record short clips and put them in:

```text
audio/
```

Good first examples:

```text
x squared minus y squared equals four
two x plus three equals seven
a cubed minus b cubed equals zero
x divided by y equals four
square root of x equals five
```

Keep clips short, ideally 2-8 seconds.

Common file types should work, such as:

```text
.wav
.mp3
.m4a
```

If one format fails, use 16 kHz mono WAV.

## Step 3: Run Audio To LaTeX

```bash
python src/run_audio.py audio/your_file.wav
```

Example output:

```text
Transcript: x squared minus y squared equals four
LaTeX: x^2 - y^2 = 4
```

To save a result:

```bash
python src/run_audio.py audio/your_file.wav --output outputs/your_file.json
```

## Step 4: Try A Bigger Model

If `tiny.en` hears math badly:

```bash
python src/run_audio.py audio/your_file.wav --model base.en
```

Then:

```bash
python src/run_audio.py audio/your_file.wav --model small.en
```

Compare the transcript and LaTeX output.

## What We Are Testing

For each recording, check:

```text
Original speech: x squared minus y squared equals four
Whisper transcript: x squared minus y squared equals four
Parser output: x^2 - y^2 = 4
```

If the transcript is wrong, Whisper/model size is the issue.

If the transcript is right but LaTeX is wrong, the parser is the issue.

That separation is the whole point of Phase 1.

## Current Parser Scope

Supported now:

- `plus` -> `+`
- `minus` -> `-`
- `times` -> `\times`
- `divided by` -> `/`
- `over` -> `/`
- `equals`, `equal to`, `is equal to` -> `=`
- `squared` -> `^2`
- `cubed` -> `^3`
- `to the power of four` -> `^4`
- `square root of x` -> `\sqrt{x}`
- number words from zero to twenty
- common Whisper mistakes like `for` -> `4`, `to` -> `2`, `why` -> `y`

Not solved yet:

- Full grouped fractions like `\frac{x + 1}{y - 2}`
- Parentheses and phrase-level grouping
- Long expressions
- Ambiguous speech like `negative x squared`
- Complex calculus notation

## Next Decision After Testing

After you add recordings, run the script and collect failures.

Then decide:

1. Improve the rule parser if Whisper transcripts are good.
2. Try `base.en` or `small.en` if transcripts are weak.
3. Start collecting paired audio/LaTeX data if we need fine-tuning.

