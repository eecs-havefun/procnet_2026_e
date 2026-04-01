# Agent Guidelines for ProcNet × W2NER

This document provides guidelines for AI agents working on the ProcNet × W2NER coupling project. It covers build commands, testing procedures, code style conventions, and project structure.

## Project Overview

ProcNet is a document-level multi-event extraction system using event proxy nodes and Hausdorff distance minimization. This repository couples ProcNet with W2NER via a sidecar entity-node mechanism. The `procnet/` directory contains the modified ProcNet library; root-level scripts handle training, inference, and verification.

**Reference repos** (read-only, do not modify):
- `../official_procnet/` — original ProcNet source
- `../official_W2NER/` — original W2NER source

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Full training | `bash run.sh` |
| Custom training | `CUDA_VISIBLE_DEVICES=0 python run.py --run_save_name=<name> --batch_size=32 --epoch=100` |
| 1-epoch smoke test | `python run_1epoch_test.py` |
| Sidecar inference | `python run_w2ner_sidecar_inference.py` |
| Full chain verification | `python verify_procnet_w2ner_chain.py` |
| Minimal verification | `python verify_procnet_w2ner_minimal.py` |
| Pipeline alignment check | `python scripts/check_full_pipeline_alignment.py` |
| Format code | `black procnet/ scripts/` |

## Environment Setup

```bash
pip install -r requirements.txt
```

**Prerequisites:**
- Pre-trained model `chinese-roberta-wwm-ext` must exist at `../models/chinese-roberta-wwm-ext`
- Dataset files (`train.json`, `dev.json`, `test.json`) should be in `procnet_format/mixed_data_with_queries/`
- Sidecar entity files should be in `sidecar_entities_gold/` or `sidecar_entities/`

## Build and Run Commands

### Training
- **Full training:** `bash run.sh` (GPU 0, batch_size=32, 100 epochs)
- **Custom training:**
  ```bash
  CUDA_VISIBLE_DEVICES=<gpu> python run.py \
    --run_save_name=<name> \
    --batch_size=<size> \
    --epoch=<num> \
    --model_name=<path_or_name>
  ```
- **Key flags:**
  - `--use_procnet_entity_nodes` — whether model uses sidecar entity nodes (default: true)
  - `--use_procnet_pred_entities` — whether processor loads procnet typed-entity sidecar (default: true)
  - `--typed_entities_dir` — path to sidecar directory
  - `--dataset_dir` — path to dataset directory

### Inference
- Sidecar inference: `python run_w2ner_sidecar_inference.py`

### Data Conversion
Scripts in `scripts/` convert between data formats:
- `convert_data_v1b_to_procnet.py` — v1b → ProcNet format
- `convert_procnet_to_w2ner.py` — ProcNet → W2NER format
- `export_doc_typed_entities.py` — export typed entities
- `regenerate_full_pipeline.py` — full pipeline regeneration

Run with `python scripts/<script_name>.py`.

## Testing and Validation

### Verification Scripts
No formal unit-test suite exists. Use these verification scripts:

**Core verification (run any directly):**
- `verify_procnet_w2ner_chain.py` — full chain integration test (static + runtime)
- `verify_procnet_w2ner_chain_local_model.py` — chain test with local model
- `verify_procnet_w2ner_minimal.py` — minimal W2NER verification
- `verify_procnet_w2ner_minimal_fixed.py` — fixed minimal verification
- `verify_procnet_w2ner_minimal_scan.py` — scanning minimal verification
- `verify_procnet_trainer_one_sample.py` — trainer with single sample

**Pipeline checks:**
- `scripts/check_full_pipeline_alignment.py` — full pipeline alignment
- `scripts/check_data_pipeline_alignment.py` — data pipeline alignment
- `scripts/full_data_pipeline_check.py` — complete data pipeline check
- `check_procnet_processor.py` — processor verification

### Running a Single Test
```bash
python verify_procnet_w2ner_chain.py
python verify_procnet_w2ner_minimal.py
python scripts/check_data_pipeline_alignment.py
```

### Linting and Formatting
No pre-configured linter exists. Recommended:
- **Black:** `black procnet/ scripts/`
- **isort:** `isort procnet/ scripts/`
- **Type checking:** `mypy procnet/ --ignore-missing-imports`

If you introduce a linter, add configuration to root and update this guide.

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
- **Functions/methods:** `snake_case` (e.g., `build_procnet_entity_nodes_for_fragment`)
- **Variables:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private members:** prefix with `_` (e.g., `_resolve_sidecar_paths`)
- **Config flags:** descriptive boolean names (e.g., `use_procnet_entity_nodes`, `return_procnet_entity_nodes`)

### Type Annotations
Use Python type hints for function arguments and return values. The codebase uses `typing` extensively.

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
├── conf/           Configuration (DocEEConfig, GlobalConfigManager)
├── data_example/   Data classes (DocEEDocumentExample, DocEETypedEntity)
├── data_preparer/  Data preparation (DocEEPreparer)
├── data_processor/ Data loading (DocEEProcessor with sidecar support)
├── dee/            Metric code from Doc2EDAG
├── metric/         Evaluation metrics (DocEEMetric)
├── model/          Neural network models (DocEEProxyNodeModel)
├── optimizer/      Optimizer wrappers (BasicOptimizer)
├── trainer/        Training loops (DocEETrainer)
├── utils/          Utilities (UtilString, UtilStructure, UtilData)
└── __init__.py
```

**Root-level scripts:** `run.py`, `run_1epoch_test.py`, `run_w2ner_sidecar_inference.py`, `verify_*.py`
**Data scripts:** `scripts/` directory for format conversion and pipeline checks

## Git Practices

- **Ignored files:** Check `.gitignore` for logs, results, cache, and temporary files
- **Commits:** Write clear messages describing the "why" not the "what"
- **Branches:** Use feature branches for new work
- **Never modify** `../official_procnet/` or `../official_W2NER/` — these are read-only references

## Agent-Specific Notes

- **No Cursor/Copilot rules** found in this repository
- **When adding dependencies**, update `requirements.txt` and verify compatibility
- **When modifying data processing**, ensure backward compatibility with existing JSON schemas
- **When adding config options**, extend `DocEEConfig` in `procnet/conf/DocEE_conf.py`
- **Sidecar entity nodes** are the key coupling mechanism — understand `procnet_entity_nodes`, `typed_entities`, and the 6-tuple batch format before modifying
- **Save conversation summaries** to `conversation_summaries/` after each session. Use date-based filenames (e.g., `2026-04-01.md`). Include: goals, analysis process, bugs found/fixed, key findings, next steps.
- **Before modifying any code**, always ask the user for confirmation first.
- **Do not repeat the same analysis** — if a pattern has been verified once, don't re-verify it unnecessarily.

## Troubleshooting

- **CUDA OOM:** Reduce `--batch_size` in `run.py` or `run.sh`
- **Missing pre-trained model:** Download `chinese-roberta-wwm-ext` and place in `../models/`
- **Import errors:** Ensure `procnet` is in `PYTHONPATH` (scripts use `sys.path.insert`)
- **Sidecar not loading:** Check `--typed_entities_dir` points to valid directory with JSONL files
- **Verification fails:** Run `verify_procnet_w2ner_minimal.py` first for a quick sanity check

---

*This document is intended for AI agents working on the ProcNet × W2NER coupling project. Update it when conventions or tooling change.*
