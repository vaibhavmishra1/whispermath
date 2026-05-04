# Phase 1 Experiment Log

Use this file after adding recordings to `audio/`.

## Device And Model

```text
Device: Apple M2 Pro
Model: tiny.en
Runtime: faster-whisper
Device mode: cpu
Compute type: int8
Status: model loads successfully
```

## Recording Checklist

Record short clips first:

```text
x squared minus y squared equals four
two x plus three equals seven
a cubed minus b cubed equals zero
x divided by y equals four
square root of x equals five
```

Run:

```bash
source .venv/bin/activate
python src/run_audio.py audio/your_file.wav --output outputs/your_file.json
```

## Results

| Audio file | Intended phrase | Whisper transcript | Parser output | Issue type | Notes |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

Issue type:

```text
whisper_error
parser_error
both
good
```

## Decision Rules

If Whisper transcript is mostly correct but LaTeX is wrong:

```text
Improve parser.
```

If Whisper transcript misses math words:

```text
Try base.en, then small.en.
```

If bigger Whisper models still fail on common phrases:

```text
Start collecting audio -> LaTeX training pairs for fine-tuning.
```

