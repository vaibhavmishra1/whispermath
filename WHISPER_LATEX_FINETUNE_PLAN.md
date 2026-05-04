# WhisperMath: Speech To LaTeX Fine-Tuning Plan

## Goal

Convert spoken mathematical expressions into compact LaTeX-like text.

Example:

```text
Audio: "x squared minus y squared is equal to four"
Output: x^2 - y^2 = 4
```

The goal is not just speech recognition. Normal Whisper transcription would usually produce:

```text
x squared minus y squared is equal to four
```

WhisperMath should instead learn the mathematical written form:

```text
x^2 - y^2 = 4
```

## Simplest Recommended Path

The simplest way to solve this problem is not to fine-tune Whisper first.

Start with a two-stage pipeline:

```text
voice -> normal Whisper transcription -> math text parser -> LaTeX output
```

For example:

```text
"x squared minus y squared is equal to four"
  -> parser
"x^2 - y^2 = 4"
```

This is easier because:

- You can build and test without collecting a large audio dataset.
- You can debug mistakes in the math conversion layer separately.
- You can improve quickly using text-only examples.
- You avoid expensive GPU fine-tuning until the product direction is clear.

Fine-tuning should become the second phase, once there is a clean dataset of audio paired directly with LaTeX targets.

## Phase 1: MVP Without Fine-Tuning

Build this first:

```text
microphone/audio file
  -> Whisper base transcription
  -> math normalization/parser
  -> LaTeX-style output
```

Example parser input/output pairs:

```text
x squared minus y squared is equal to four -> x^2 - y^2 = 4
square root of x plus one -> \sqrt{x + 1}
integral from zero to one of x squared dx -> \int_0^1 x^2 dx
limit as x approaches zero of sine x over x -> \lim_{x \to 0} \frac{\sin x}{x}
```

The parser can begin as a small rule-based system:

- `squared` -> `^2`
- `cubed` -> `^3`
- `plus` -> `+`
- `minus` -> `-`
- `times` -> `\times`
- `divided by` / `over` -> fraction handling
- `equals` / `is equal to` -> `=`
- number words -> digits
- `square root of ...` -> `\sqrt{...}`
- `open parenthesis` / `close parenthesis` -> `(` / `)`

As expressions get more complex, replace or supplement rules with a text-to-LaTeX model or an LLM-based converter.

## Phase 2: Dataset Collection

To fine-tune Whisper, collect pairs like this:

```json
{
  "audio": "audio/000001.wav",
  "spoken_text": "x squared minus y squared is equal to four",
  "latex": "x^2 - y^2 = 4",
  "domain": "algebra",
  "speaker_id": "speaker_001"
}
```

For Whisper fine-tuning, the most important pair is:

```text
audio -> latex
```

The `spoken_text` field is still useful for debugging, quality control, and building the Phase 1 parser.

### Dataset Sources

Use a mix of these:

1. Human-recorded expressions
   - Ask speakers to read generated prompts.
   - Record multiple accents, speaking speeds, microphones, and noise levels.
   - Keep audio short: ideally 2-12 seconds per sample.

2. Scripted math prompt generation
   - Generate many expression templates programmatically.
   - Store both the spoken form and target LaTeX.
   - Example template:

```text
spoken: "{var} squared minus {var} squared is equal to {number}"
latex: "{var}^2 - {var}^2 = {number}"
```

3. Synthetic text-to-speech data
   - Use TTS to bootstrap many examples.
   - Useful for early experiments, but do not rely on it alone.
   - Real human recordings are still needed because TTS audio is too clean and consistent.

4. Existing lecture or tutorial audio
   - More realistic, but harder to label.
   - Better for later validation than initial supervised training.

### Dataset Size Targets

Start small and increase only when needed:

- Prototype parser: 200-500 text-only expression pairs.
- First Whisper fine-tune smoke test: 500-2,000 audio/LaTeX pairs.
- Useful domain model: 10,000-50,000 pairs.
- Stronger production model: 100,000+ diverse pairs.

Quality matters more than raw size. A small clean dataset beats a large inconsistent one.

### Expression Categories

Collect data by category so progress can be measured clearly:

- Arithmetic: `2 + 3 = 5`
- Algebra: `x^2 - y^2 = 4`
- Fractions: `\frac{x + 1}{y - 2}`
- Exponents and roots: `x^3`, `\sqrt{x}`
- Trigonometry: `\sin x`, `\cos^2 x`
- Calculus: derivatives, integrals, limits
- Linear algebra: matrices, vectors, dot products
- Greek symbols: `\alpha`, `\beta`, `\theta`
- Parentheses and grouping
- Spoken ambiguity cases, such as "negative x squared" vs "negative x, squared"

## Phase 3: Fine-Tuning Whisper

### What To Edit In Whisper

For the first fine-tuning version, do not change Whisper's architecture.

Use the existing encoder-decoder model and train it with:

```text
input audio -> target LaTeX string
```

This means:

- Keep the audio encoder unchanged structurally.
- Keep the decoder unchanged structurally.
- Keep the tokenizer unchanged at first.
- Fine-tune model weights on math audio paired with LaTeX targets.

Whisper can already emit normal text characters like:

```text
x ^ 2 - y ^ 2 = 4 \ frac { x } { y }
```

So the first version does not need custom vocabulary.

### Tokenizer Decision

Start with the default Whisper tokenizer.

Only consider adding custom tokens later if outputs are slow, inconsistent, or tokenization is clearly hurting quality for common LaTeX commands.

Possible future custom tokens:

```text
\frac
\sqrt
\int
\sum
\lim
\theta
\alpha
\beta
```

Adding tokens requires resizing the model's token embeddings and carefully continuing training. This adds complexity, so it should not be part of the first implementation.

### Fine-Tuning Options

Use one of these approaches:

1. Full fine-tuning
   - Best quality.
   - More GPU memory.
   - Higher risk of overfitting on small datasets.

2. LoRA / PEFT fine-tuning
   - Recommended first training approach.
   - Cheaper and faster.
   - Easier to iterate.
   - Keeps the base Whisper model mostly intact.

3. Freeze encoder, train decoder-side adapters
   - Useful when audio recognition is already good.
   - The main task is learning how to write math.

Recommended first experiment:

```text
openai/whisper-small + LoRA + 2,000 clean audio/LaTeX pairs
```

If GPU is limited, start with:

```text
openai/whisper-tiny or openai/whisper-base
```

## Training Data Format

Use a manifest file like:

```json
{"audio": "audio/000001.wav", "text": "x^2 - y^2 = 4"}
{"audio": "audio/000002.wav", "text": "\\sqrt{x + 1}"}
{"audio": "audio/000003.wav", "text": "\\frac{x + 1}{y - 2}"}
```

Recommended folder structure:

```text
whispermath/
  data/
    raw_audio/
    processed_audio/
    manifests/
      train.jsonl
      validation.jsonl
      test.jsonl
  scripts/
    generate_prompts.py
    record_dataset.py
    prepare_dataset.py
    train_whisper_lora.py
    evaluate.py
  models/
  outputs/
  docs/
```

Audio should be normalized to Whisper's expected format:

```text
16 kHz mono WAV
```

## Evaluation

Do not rely only on normal word error rate.

For math output, evaluate:

- Exact string match after normalization.
- Character error rate.
- LaTeX command accuracy.
- Symbol accuracy.
- Structure accuracy for fractions, roots, powers, and grouping.
- Renderability: can the LaTeX be parsed/rendered?

Example normalization before scoring:

```text
x^2-y^2=4
x ^ 2 - y ^ 2 = 4
```

These should be treated as equivalent if spacing is not important.

## Major Risks

### Ambiguous Speech

Spoken math is often ambiguous.

Example:

```text
negative x squared
```

Could mean:

```text
-x^2
```

or:

```text
(-x)^2
```

The dataset needs consistent speaking conventions, or the UI needs a correction flow.

### Grouping Is Hard

Fractions, roots, integrals, and parentheses require knowing where a structure starts and ends.

Example:

```text
one over x plus one
```

Could mean:

```text
\frac{1}{x} + 1
```

or:

```text
\frac{1}{x + 1}
```

This is why early data should include explicit phrases like:

```text
one over the quantity x plus one
```

### Tiny Dataset Overfitting

If the dataset is too small, the model may memorize templates instead of learning robust speech-to-LaTeX conversion.

Use a held-out test set with speakers and expression templates that were not seen during training.

## Recommended Roadmap

### Step 1: Build Text Parser

Create text-only pairs:

```text
spoken math text -> LaTeX
```

Build a simple parser and evaluate it on 200-500 examples.

### Step 2: Add Whisper Transcription

Run normal Whisper first:

```text
audio -> spoken math text -> parser -> LaTeX
```

This proves the end-to-end product without training.

### Step 3: Create Dataset Tools

Add scripts for:

- Generating spoken/LaTeX prompt pairs.
- Recording audio for each prompt.
- Validating audio length and sample rate.
- Splitting train/validation/test.
- Exporting JSONL manifests.

### Step 4: Collect First Dataset

Collect:

- 500-2,000 short recordings.
- At least 5-10 speakers.
- Algebra, fractions, powers, roots, and basic calculus.

### Step 5: Fine-Tune Whisper With LoRA

Train:

```text
audio -> LaTeX
```

Start with `whisper-small` if hardware allows. Otherwise use `whisper-base`.

### Step 6: Compare Against MVP

Compare:

```text
Whisper + parser
```

against:

```text
Fine-tuned Whisper -> LaTeX
```

Keep the simpler pipeline if it performs well enough.

## Final Recommendation

Build the rule/parser MVP first, then collect data, then fine-tune.

The first successful version should probably be:

```text
Whisper transcription + math-aware text-to-LaTeX converter
```

The fine-tuned model should come later, after the dataset is clean and the target LaTeX style is stable.

This avoids spending time training a model before the project has answered the harder product questions:

- What exact LaTeX style should be produced?
- How should ambiguous spoken math be handled?
- Which math domains matter first?
- How much correction/editing will the user need?

