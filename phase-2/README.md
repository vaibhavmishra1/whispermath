# WhisperMath Phase 2: Raw Output Dataset Builder

Phase 2 starts by building a raw output dataset from three public Hugging Face sources.

```text
latex formula text
mixed math question/answer text
normal text
```

The generated JSONL rows use:

```json
{
  "id": "mixed-00000042",
  "output_text": "Why is the volume of a sphere $\\frac{4}{3}\\pi r^3$?",
  "type": "mixed",
  "source_dataset": "math-ai/StackMathQA",
  "source_config": "stackmathqa100k",
  "source_split": "train",
  "source_row_index": 42,
  "source_field": "Q",
  "metadata_json": "{}"
}
```

This step does not call OpenAI and does not generate spoken text. It only prepares raw output-side data and uploads it to Hugging Face.

## Dataset Choices

Pure LaTeX:

```text
OleehyO/latex-formulas
config: cleaned_formulas
split: train
column: latex_formula
type: latex
```

Mixed math text:

```text
math-ai/StackMathQA
config: stackmathqa100k
split: train
fields: Q, A
type: mixed
```

Normal text:

```text
Salesforce/wikitext
config: wikitext-103-raw-v1
split: train
field: text
type: normal
```

## Setup

```bash
cd /Users/vaibhav/Desktop/beyond/whispermath/phase-2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Build A Small Smoke Dataset

```bash
python src/build_raw_output_dataset.py \
  --latex-limit 20 \
  --mixed-limit 20 \
  --normal-limit 20 \
  --output data/compiled/raw_output_smoke.jsonl
```

## Build The Recommended Balanced Dataset

```bash
python src/build_raw_output_dataset.py \
  --latex-limit 50000 \
  --mixed-limit 50000 \
  --normal-limit 50000 \
  --output data/compiled/raw_output_dataset_balanced_50k.jsonl
```

Outputs:

```text
data/compiled/raw_output_dataset_balanced_50k.jsonl
data/compiled/raw_output_dataset_balanced_50k.summary.json
```

Do not use `--latex-limit all` for the first training dataset. `OleehyO/latex-formulas` has more than 500k usable rows, which would make the dataset heavily skewed toward pure LaTeX.

If you already built the full unbalanced file, create a balanced 50k/50k/50k file without redownloading:

```bash
python src/balance_raw_dataset.py \
  --input data/compiled/raw_output_dataset.jsonl \
  --output data/compiled/raw_output_dataset_balanced_50k.jsonl \
  --per-type 50000 \
  --shuffle
```

## Upload To Hugging Face

```bash
export HF_TOKEN="your_huggingface_token"
python src/upload_raw_dataset_to_hf.py \
  --input data/compiled/raw_output_dataset_balanced_50k.jsonl \
  --repo-id vibhuiitj/whispermath-raw-output
```

Or let the upload script ask for the token without putting it in shell history:

```bash
python src/upload_raw_dataset_to_hf.py \
  --input data/compiled/raw_output_dataset_balanced_50k.jsonl \
  --repo-id vibhuiitj/whispermath-raw-output
```

To create a private dataset:

```bash
python src/upload_raw_dataset_to_hf.py --private
```

## Test OpenAI Spoken Outputs

Generate one spoken-text sample for 10 rows from each category:

```bash
export OPENAI_API_KEY="your_openai_key"
python src/test_openai_spoken_outputs.py
```

Outputs:

```text
data/generated/openai_spoken_sample_30.jsonl
data/generated/openai_spoken_sample_30.md
```

The script samples from the local balanced dataset by default. To sample directly from Hugging Face instead:

```bash
python src/test_openai_spoken_outputs.py --use-hf
```

## Generate `input_text` With OpenAI Batch API

Final training dataset target:

```text
latex:  50k rows, OpenAI-generated input_text with one random spoken style per row
mixed:  50k rows, OpenAI-generated input_text
normal: 50k rows, input_text copied from output_text
```

Prepare Batch API request files:

```bash
python src/prepare_input_text_batches.py \
  --input data/compiled/raw_output_dataset_balanced_50k.jsonl \
  --output-dir data/batches/input_text_requests \
  --model gpt-5.4-nano \
  --requests-per-file 10000
```

Submit the OpenAI batches:

```bash
export OPENAI_API_KEY="your_openai_key"
python src/submit_openai_batches.py \
  --manifest data/batches/input_text_requests/manifest.json
```

Check status:

```bash
python src/check_openai_batches.py
```

Download completed outputs:

```bash
python src/download_openai_batch_outputs.py
```

Merge OpenAI outputs with the original balanced dataset:

```bash
python src/merge_input_text_dataset.py --validate-hf
```

Upload final training dataset:

```bash
export HF_TOKEN="your_huggingface_token"
python src/upload_input_output_dataset_to_hf.py \
  --input data/compiled/input_output_dataset_150k.jsonl \
  --repo-id vibhuiitj/whispermath-input-output
```

## Legacy Experiment Scripts

This folder still contains earlier small-scale inspection scripts for LaTeX sampling and review. The raw dataset builder above is the current path for this step.

## Notes

Compiled data files are ignored by git because they can be large.

Run the smoke dataset command first before building the full dataset.
