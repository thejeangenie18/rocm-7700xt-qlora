# Datasets

This folder holds JSONL training datasets. Each file contains one JSON object per line.

## Expected Schema

```json
{"instruction": "...", "input": "...", "output": "..."}
```

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | yes | The task or question posed to the model |
| `input` | yes | Additional context (use `""` if none) |
| `output` | yes | The expected model response |

The schema is validated by `schema/schema.json`. For validation tooling, see `scripts/validate_adapter.py`.

## Included Files

| File | Contents |
|------|----------|
| `ada.jsonl` | Accessibility-focused instruction examples derived from ADA regulatory text |
| `codeexercise.jsonl` | Python coding exercise completions |

## Adding Your Own Data

Create a JSONL file following the schema above and pass its path to the training script:

```bash
DATA_PATH=./datasets/my-data.jsonl python scripts/train_rdna3_fix.py
```
