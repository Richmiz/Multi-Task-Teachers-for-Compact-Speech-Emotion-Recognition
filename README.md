# Does Multi-Task Teacher Training Improve Compact Speech Emotion Models?

This repository contains the code for a controlled study of knowledge distillation in speech emotion recognition (SER).

**Research question.** Does the supervision used to train a wav2vec 2.0 teacher change the accuracy of a compact student of fixed size? Every teacher starts from the same pretrained checkpoint and uses the same data partitions and update budget. Only the training objective changes.

**Status.** The study is under development. No results are available yet.

## Experimental conditions

| Teacher objective | Training tasks |
| --- | --- |
| `ser_only` | Emotion recognition |
| `ser_asr_sid` | Emotion recognition, speech recognition, and speaker identification |
| `ser_asr` | Emotion recognition and speech recognition |
| `ser_sid` | Emotion recognition and speaker identification |
| `asr_then_ser` | Speech recognition, followed by emotion recognition |

Each teacher produces emotion-only students with six or four Transformer layers. The primary comparison is between six-layer students distilled from the `ser_asr_sid` and `ser_only` teachers, evaluated with five leave-one-session-out IEMOCAP folds. Inference is measured on a CPU in FP32 and dynamic INT8 formats.

## Repository structure

```
configs/          Fixed study settings
data/             Dataset instructions (datasets are not distributed)
docs/             Runtime documentation
scripts/          Command-line tools, including the Colab notebook builder
src/compact_ser/  The experiment package
tests/            Unit tests
```

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e . --no-deps
```

On Linux or macOS, use `.venv/bin/python`. `requirements.txt` installs the CPU build of PyTorch used for local development and tests. Training runs on Google Colab; see [the Colab guide](docs/colab-guide.md).

## Development

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Data

The study uses IEMOCAP, RAVDESS, and Berlin Emo-DB. Datasets are not included in this repository; see [`data/README.md`](data/README.md).
