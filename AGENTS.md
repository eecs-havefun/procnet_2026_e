# Agent Guidelines for ProcNet × W2NER

This document provides guidelines for AI agents working on the ProcNet × W2NER coupling project. It covers build commands, testing procedures, code style conventions, and project structure.

## Project Overview

ProcNet is a document-level multi-event extraction system using event proxy nodes and Hausdorff distance minimization (ACL 2023). This repository couples ProcNet with W2NER via a sidecar entity-node mechanism, enabling an end-to-end cascade from named entity recognition to document-level multi-event extraction.

**Data pipeline:** RASA NLU (data_v1b) → ProcNet format → W2NER training/prediction → sidecar JSONL → ProcNet training

**Current status:** W2NER → ProcNet cascade is fully functional. 10-epoch full training achieves Event F1 ≈ 97.3% on test set with W2NER-predicted sidecar entities.

**Future direction:** EPAL integration — role-indexed slot filling to solve cross-event entity reuse and same-event multi-role conflicts.

**Reference repos** (read-only, do not modify):
- `../official_procnet/` — original ProcNet source
- `../official_W2NER/` — original W2NER source

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Full training | `bash run.sh` |
| Custom training | `CUDA_VISIBLE_DEVICES=0 python run.py --run_save_name=<name> --batch_size=32 --epoch=100` |
| Smoke test | `python run_1epoch_test.py` |
| Single-sample verify | `python verify_procnet_trainer_one_sample.py` |
| Generate predictions | `python generate_predictions.py --checkpoint Checkpoint/<name>/<name>_XXX.pth --run_save_name=<name>` |
| Data: RASA → ProcNet | `python scripts/convert_data_v1b_to_procnet.py --input_dir ... --output_dir ...` |
| Data: ProcNet → W2NER | `python scripts/convert_procnet_to_w2ner.py --input_dir ... --output_dir ...` |
| Data: W2NER pred → sidecar | `python scripts/export_doc_typed_entities.py --source_json ... --pred_json ...` |
| Pipeline check | `python scripts/check_full_pipeline_alignment.py` |
| Format code | `black procnet/ scripts/` |

## Environment Setup

```bash
pip install -r requirements.txt
```

**Prerequisites:**
- Pre-trained model `chinese-roberta-wwm-ext` at `../models/chinese-roberta-wwm-ext` (configurable via `--model_name`)
- Dataset: `procnet_format/mixed_data_with_queries/{train,dev,test}.json` (3,360 / 720 / 720 docs)
- Sidecar: `sidecar_entities/{train,dev,test}_doc_typed_entities.jsonl` (W2NER predicted) or `sidecar_entities_gold/` (gold)

## Build and Run Commands

### Training

**Full training:**
```bash
bash run.sh
# Equivalent to: CUDA_VISIBLE_DEVICES=0 python run.py --run_save_name=exp0 --batch_size=32 --epoch=100
```

**Custom training:**
```bash
CUDA_VISIBLE_DEVICES=<gpu> python run.py \
  --run_save_name=<name> \
  --batch_size=<grad_accum_steps> \
  --epoch=<num> \
  --model_name=<path_or_name> \
  --save_top_k=<k>
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--run_save_name` | (required) | Experiment name for output directories |
| `--batch_size` | 32 | Gradient accumulation steps |
| `--epoch` | 50 | Training epochs |
| `--use_procnet_entity_nodes` | true | Model uses sidecar entity nodes |
| `--use_procnet_pred_entities` | true | Processor loads sidecar entities |
| `--return_procnet_entity_nodes` | same as above | Processor/Preparer returns entity nodes in batch |
| `--typed_entities_dir` | `./tmp_sidecar` | Path to sidecar directory |
| `--dataset_dir` | `./Data` | Path to dataset directory |
| `--proxy_slot_num` | 16 | Number of event proxy slots |
| `--node_size` | 512 | Node hidden dimension |
| `--max_len` | 510 | Max token length per fragment |
| `--save_top_k` | 1 | Keep top-K checkpoints by dev F1 (-1 = save all) |
| `--device` | cuda | Training device (cuda/cpu) |
| `--data_loader_shuffle` | true | Shuffle training data |

### Experiment Modes

**Gold sidecar (upper bound):**
```bash
python run.py --run_save_name=gold_exp \
  --dataset_dir=./procnet_format/mixed_data_with_queries \
  --typed_entities_dir=./sidecar_entities_gold \
  --batch_size=32 --epoch=100
```

**W2NER sidecar (cascade):**
```bash
python run.py --run_save_name=w2ner_exp \
  --dataset_dir=./procnet_format/mixed_data_with_queries \
  --typed_entities_dir=./sidecar_entities \
  --batch_size=32 --epoch=100
```

**No sidecar (baseline):**
```bash
python run.py --run_save_name=baseline_exp \
  --dataset_dir=./procnet_format/mixed_data_with_queries \
  --use_procnet_pred_entities=false \
  --use_procnet_entity_nodes=false \
  --batch_size=32 --epoch=100
```

### Output

Training outputs are organized under `Result/{run_save_name}/` and `Checkpoint/{run_save_name}/`:

```
Result/{run_save_name}/
├── {run_save_name}_001.json          # Per-epoch aggregated metrics
├── {run_save_name}_002.json
├── ...
└── {run_save_name}_predictions.json  # Best-epoch per-document predictions

Checkpoint/{run_save_name}/
├── {run_save_name}_001.pth           # Model weights (top-K by dev F1)
├── ...
```

`{run_save_name}_predictions.json` contains:
- `best_epoch` — epoch number with highest dev Event F1
- `best_dev_f1` — the best dev F1 value
- `dev_predictions` — per-document predictions for dev set (doc_id, BIO_ans, event_ans, etc.)
- `test_predictions` — per-document predictions for test set

### Data Conversion

**1. RASA NLU → ProcNet format:**
```bash
python scripts/convert_data_v1b_to_procnet.py \
  --input_dir ./data_v1b/mixed_data_with_queries \
  --output_dir ./procnet_format/mixed_data_with_queries \
  --split all
```

**2. ProcNet → W2NER sentence-level format:**
```bash
python scripts/convert_procnet_to_w2ner.py \
  --input_dir ./procnet_format/mixed_data_with_queries \
  --output_dir ../W2NER/data/mixed_data_with_queries
```

**3. W2NER predictions → ProcNet sidecar JSONL:**
```bash
python scripts/export_doc_typed_entities.py \
  --source_json ../W2NER/data/mixed_data_with_queries/test.json \
  --pred_json ../W2NER/predictions/test.json \
  --output_jsonl ./sidecar_entities/test_doc_typed_entities.jsonl \
  --report_json ./sidecar_entities/test_export_report.json
```

### Pipeline Verification

```bash
# Full pipeline alignment check (RASA vs ProcNet vs W2NER)
python scripts/check_full_pipeline_alignment.py

# Data consistency via MD5
python scripts/check_data_pipeline_alignment.py

# Complete pipeline check (samples, sentences, entities, types)
python scripts/full_data_pipeline_check.py

# Data loss check across pipeline stages
python scripts/check_data_loss.py
```

## Testing and Validation

### Smoke Test

```bash
python run_1epoch_test.py
```

Validates the full pipeline: sidecar loading → processor → preparer → model → trainer. Uses `sidecar_entities/` directory and `procnet_format/mixed_data_with_queries/` dataset. Runs 10 epochs with `proxy_slot_num=8`.

### Single-Sample Verification

```bash
python verify_procnet_trainer_one_sample.py \
  --split train \
  --pick non_empty
```

Picks a single sample from the dataset and runs `trainer.model_fn()` to verify the forward pass. Useful for debugging. Options:
- `--split`: train/dev/test
- `--pick`: non_empty/empty
- `--target_doc_id`: pick a specific document
- `--run_eval`: run in eval mode

### Linting and Formatting

No pre-configured linter exists. Recommended:
- **Black:** `black procnet/ scripts/`
- **isort:** `isort procnet/ scripts/`
- **Type checking:** `mypy procnet/ --ignore-missing-imports`

## Code Style Guidelines

### Imports

Order: (1) standard library, (2) third-party (torch, transformers, numpy), (3) internal (procnet modules).

```python
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from transformers import PreTrainedModel

from procnet.conf.DocEE_conf import DocEEConfig
from procnet.data_processor.DocEE_processor import DocEEProcessor
```

### Naming Conventions

- **Classes:** `CamelCase` (e.g., `DocEEProcessor`, `DocEEConfig`)
- **Functions/methods:** `snake_case` (e.g., `_resolve_sidecar_paths`)
- **Variables:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private members:** prefix with `_`
- **Config flags:** descriptive boolean names (e.g., `use_procnet_entity_nodes`)

### Type Annotations

Use Python type hints. The codebase uses `typing` extensively.

```python
def process_document(
    doc: DocEEDocumentExample,
    config: DocEEConfig,
) -> List[DocEETypedEntity]:
    ...
```

### Error Handling

- Use explicit `try-except` blocks for predictable failures (file I/O, network calls)
- Log errors with `logging.error()` or `logging.warning()`
- Never silently catch exceptions; at minimum log them
- Use `raise FileNotFoundError` / `raise ValueError` for invalid paths/configs

### Logging

Follow the pattern used in `run.py`:
```python
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S'
)
```

### Documentation

New code should include docstrings in Google-style format:
```python
def build_procnet_entity_nodes(fragment, config):
    """Build typed entity nodes for a document fragment.

    Args:
        fragment: A single document fragment with tokenized text.
        config: DocEEConfig instance with entity node settings.

    Returns:
        List[Dict]: Entity nodes with span, type, and token_id information.
    """
```

### Formatting

- Indent with 4 spaces (no tabs)
- Line length: under 100 characters
- Use double quotes for strings
- Class structure: `__init__` first, then public methods, then private methods
- Use `@staticmethod` for utility functions without instance state

## Project Structure

```
procnet/
├── run.py                              # Main entry: arg parsing + pipeline orchestration
├── run.sh                              # Quick training wrapper (GPU 0, batch=32, epoch=100)
├── run_1epoch_test.py                  # Smoke test: validates full sidecar→training pipeline
├── verify_procnet_trainer_one_sample.py # Single-sample forward pass verification
├── generate_predictions.py              # Generate predictions from checkpoint
│
├── procnet/                            # Core library
│   ├── conf/
│   │   ├── basic_conf.py               # BasicConfig: lr, epochs, device, grad_accum
│   │   ├── DocEE_conf.py               # DocEEConfig: proxy_slot_num, node_size, save_top_k
│   │   └── global_config_manager.py    # Global path configuration
│   ├── data_example/
│   │   ├── DocEEexample.py             # DocEEDocumentExample, DocEEEntity, DocEETypedEntity
│   │   └── DuEEfin_example.py
│   ├── data_processor/
│   │   ├── DocEE_processor.py          # Parse JSON + load sidecar + composite key resolution
│   │   └── DuEE_fin_processor.py
│   ├── data_preparer/
│   │   ├── DocEE_preparer.py           # Tokenize + BIO tags + fragment splitting + DataLoader
│   │   └── DuEE_fin_preparer.py
│   ├── model/
│   │   ├── DocEE_proxy_node_model.py   # BERT + BIO + GCN (FiLMConv) + Hausdorff loss
│   │   └── basic_model.py
│   ├── trainer/
│   │   ├── DocEE_proxy_node_trainer.py # Training loop + Top-K checkpoint + predictions output
│   │   └── basic_trainer.py
│   ├── metric/
│   │   ├── DocEE_metric.py             # BIO scoring + event table-filling metrics
│   │   └── basic_metric.py
│   ├── optimizer/
│   │   └── basic_optimizer.py          # Optimizer wrapper (grad accum + model saving)
│   ├── dee/                            # Doc2EDAG metric code
│   ├── utils/                          # UtilData, UtilString, UtilStructure, UtilMath
│   └── data_example/                   # Data example classes
│
├── scripts/                            # Data conversion and verification scripts
│   ├── data_paths.py                   # Centralized path configuration
│   ├── convert_data_v1b_to_procnet.py  # RASA NLU → ProcNet format
│   ├── convert_procnet_to_w2ner.py     # ProcNet → W2NER sentence-level format
│   ├── export_doc_typed_entities.py    # W2NER predictions → document-level sidecar JSONL
│   ├── check_data_loss.py              # Entity/text loss check across pipeline
│   ├── check_data_pipeline_alignment.py # MD5-based data consistency check
│   ├── check_full_pipeline_alignment.py # Full RASA→ProcNet→W2NER alignment
│   ├── full_data_pipeline_check.py     # Comprehensive pipeline validation
│   ├── check_data_v1b_procnet.py       # Source data verification
│   └── compare_with_original_v1b.py    # Original vs converted data comparison
│
├── data_v1b/                           # Source data (RASA NLU format)
├── procnet_format/                     # Converted ProcNet format (train/dev/test per dataset)
├── sidecar_entities_gold/              # Gold sidecar entities (upper-bound experiments)
├── sidecar_entities/                   # W2NER-predicted sidecar entities (cascade experiments)
├── conversation_summaries/             # Session conversation summaries
├── figures/                            # Architecture diagrams
├── Checkpoint/                         # Model checkpoints (organized by run_save_name)
├── Result/                             # Training results (per-epoch JSON + predictions)
├── requirements.txt
├── AGENTS.md                           # This file
└── README.md                           # Project documentation
```

## Git Practices

- **Ignored files:** Check `.gitignore` for logs, results, cache, checkpoints, and temporary files
- **Commits:** Write clear messages describing the "why" not the "what"
- **Branches:** Use feature branches for new work
- **Never modify** `../official_procnet/` or `../official_W2NER/` — these are read-only references
- **Before modifying any code**, always ask the user for confirmation first
- **Before pushing**, review what will be pushed and ask the user for confirmation

## Agent-Specific Notes

- **Sidecar entity nodes** are the key coupling mechanism — understand `procnet_entity_nodes`, `typed_entities`, and the 6-tuple batch format before modifying
- **When adding dependencies**, update `requirements.txt` and verify compatibility
- **When modifying data processing**, ensure backward compatibility with existing JSON schemas
- **When adding config options**, extend `DocEEConfig` in `procnet/conf/DocEE_conf.py` and add CLI arg in `run.py`
- **Save conversation summaries** to `conversation_summaries/` after each session. Use date-based filenames (e.g., `2026-04-02-w2ner-procnet-cascade.md`). Include: goals, analysis process, bugs found/fixed, key findings, next steps
- **Do not repeat the same analysis** — if a pattern has been verified once, don't re-verify it unnecessarily
- **EPAL integration** is planned but not yet implemented. See `epal_procnet_report_and_dialogue_summary.md` for the design document. Do not implement EPAL without explicit user confirmation

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `--batch_size` (gradient accumulation steps) |
| Missing pre-trained model | Download `chinese-roberta-wwm-ext` and place in `../models/` or use `--model_name` |
| Import errors | Scripts use `sys.path.insert` — ensure `procnet` is importable |
| Sidecar not loading | Check `--typed_entities_dir` points to a directory with `{split}_doc_typed_entities.jsonl` files |
| No checkpoint saved | Verify `save_top_k >= 1` (default: 1). Check `Checkpoint/` directory permissions |
| Verification fails | Run `run_1epoch_test.py` first for a quick sanity check |
| Sidecar doc_id mismatch | Processor supports relaxed matching (normalizes doc_id by stripping `.json` suffix and lowercasing) |

---

*This document is intended for AI agents working on the ProcNet × W2NER coupling project. Update it when conventions or tooling change.*
