#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load an existing checkpoint and generate predictions for dev/test sets.
Saves {name}_predictions.json to Result/{name}/ directory.

Usage:
    python generate_predictions.py \
        --checkpoint Checkpoint/w2ner_sidecar_exp1/w2ner_sidecar_exp1_010.pth \
        --run_save_name w2ner_sidecar_exp1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_preparer.DocEE_preparer import DocEEPreparer
from procnet.model.DocEE_proxy_node_model import DocEEProxyNodeModel
from procnet.optimizer.basic_optimizer import BasicOptimizer
from procnet.trainer.DocEE_proxy_node_trainer import DocEETrainer
from procnet.metric.DocEE_metric import DocEEMetric
from procnet.conf.DocEE_conf import DocEEConfig

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
MODEL_PATH = PROJECT_ROOT / "models" / "chinese-roberta-wwm-ext"
DATASET_DIR = REPO_ROOT / "procnet_format" / "mixed_data_with_queries"
SIDECAR_DIR = REPO_ROOT / "sidecar_entities"


def get_config(run_save_name: str):
    config = DocEEConfig()
    config.model_name = str(MODEL_PATH)
    config.proxy_slot_num = 8
    config.node_size = 512
    config.max_len = 510
    config.max_epochs = 1
    config.gradient_accumulation_steps = 32
    config.data_loader_shuffle = False
    config.return_procnet_entity_nodes = True
    config.use_procnet_entity_nodes = True
    config.use_procnet_pred_entities = True
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.model_save_name = run_save_name
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--run_save_name", type=str, required=True, help="Experiment name for output path")
    parser.add_argument("--dataset_dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--sidecar_dir", type=str, default=str(SIDECAR_DIR))
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH))
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    config = get_config(args.run_save_name)
    
    # Setup logging to file
    log_path = REPO_ROOT / f"predict_{args.run_save_name}.log"
    logger = logging.getLogger("predict")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%I:%M:%S'))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%I:%M:%S'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"=== Starting prediction for {args.run_save_name} ===")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Run name:   {args.run_save_name}")
    logger.info(f"Device:     {config.device}")
    logger.info(f"Log file:   {log_path}")

    # 1. Processor
    logger.info("[1/5] Loading data...")
    processor = DocEEProcessor(
        read_pseudo_dataset=False,
        use_procnet_pred_entities=True,
        dataset_dir=args.dataset_dir,
        typed_entities_dir=args.sidecar_dir,
    )
    logger.info(f"  train: {len(processor.train_docs)}, dev: {len(processor.dev_docs)}, test: {len(processor.test_docs)}")

    # 2. Preparer
    logger.info("[2/5] Preparing data...")
    preparer = DocEEPreparer(config=config, processor=processor)
    pre_data = preparer.get_loader_for_flattened_fragment_before_event()
    train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader = pre_data
    logger.info(f"  dev batches: {len(dev_loader)}, test batches: {len(test_loader)}")

    # 3. Model
    logger.info("[3/5] Creating model...")
    model = DocEEProxyNodeModel(config=config, preparer=preparer)
    model = model.to(config.device)

    # 4. Load checkpoint
    logger.info("[4/5] Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    # Checkpoint may be a raw state_dict (OrderedDict) or a dict with model_state_dict key
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Raw state_dict (OrderedDict)
        model.load_state_dict(checkpoint)
    model.eval()
    logger.info(f"  Loaded from {checkpoint_path.name}")

    # 5. Trainer (for model_fn wrapper)
    logger.info("[5/5] Running inference...")
    optimizer = BasicOptimizer(config=config, model=model)
    metric = DocEEMetric(preparer=preparer)
    trainer = DocEETrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        preparer=preparer,
        metric=metric,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
    )

    # Run eval on dev and test
    logger.info("  Evaluating dev set...")
    dev_score, dev_raw = trainer.eval_batch_template(
        model_run_fn=trainer.model_fn,
        score_fn=trainer.score_fn,
        dataloader=dev_loader,
        epoch=999,
    )
    logger.info(f"  Dev Event F1: {dev_score['event']['all_event']['micro_f1']:.4f}")

    logger.info("  Evaluating test set...")
    test_score, test_raw = trainer.eval_batch_template(
        model_run_fn=trainer.model_fn,
        score_fn=trainer.score_fn,
        dataloader=test_loader,
        epoch=999,
    )
    logger.info(f"  Test Event F1: {test_score['event']['all_event']['micro_f1']:.4f}")

    # Post-process: convert tuple keys to strings for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, tuple):
            return str(obj)
        elif isinstance(obj, list):
            return [clean_for_json(x) for x in obj]
        elif isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: clean_for_json(v) for k, v in obj.items()}
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj

    dev_raw_clean = clean_for_json(dev_raw)
    test_raw_clean = clean_for_json(test_raw)

    # Save predictions
    result_dir = REPO_ROOT / "Result" / args.run_save_name
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_file = result_dir / f"{args.run_save_name}_predictions.json"
    pred_data = {
        "best_epoch": int(checkpoint_path.name.split("_")[-1].replace(".pth", "")),
        "best_dev_f1": dev_score["event"]["all_event"]["micro_f1"],
        "dev_predictions": dev_raw_clean,
        "test_predictions": test_raw_clean,
    }
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Predictions saved to: {pred_file}")
    print(f"   Dev samples:   {len(dev_raw)}")
    print(f"   Test samples:  {len(test_raw)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run_save_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--sidecar_dir", type=str, default=str(SIDECAR_DIR))
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH))
    args = parser.parse_args()
    main()
