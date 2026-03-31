#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path configuration for data pipeline scripts.

Usage:
    from data_paths import PROJECT_ROOT, DATA_V1B, PROCNET_FORMAT, W2NER_FORMAT
"""

from pathlib import Path

# Get the project root directory (parent of the script's directory)
# This works whether the script is in procnet/scripts/ or W2NER/scripts_maybeuseful/
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Data directories
DATA_V1B = PROJECT_ROOT / "data_v1b"
PROCNET_FORMAT = PROJECT_ROOT / "procnet" / "procnet_format"
W2NER_FORMAT = PROJECT_ROOT / "W2NER" / "data" / "data_w2ner_folded_with_dev"
W2NER_FOLDED = PROJECT_ROOT / "data_w2ner_folded"
W2NER_ALT = PROJECT_ROOT / "data_w2ner"

# ProcNet directories
PROCNET_ROOT = PROJECT_ROOT / "procnet"
PROCNET_DATA_V1B = PROCNET_ROOT / "Data_v1b"
PROCNET_SIDEAR_ENTITIES = PROCNET_ROOT / "sidecar_entities"
PROCNET_SIDEAR_ENTITIES_GOLD = PROCNET_ROOT / "sidecar_entities_gold"

# W2NER directories
W2NER_ROOT = PROJECT_ROOT / "W2NER"
W2NER_DATA = W2NER_ROOT / "data"

# Dataset names
DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def get_data_v1b() -> Path:
    """Get the data_v1b directory."""
    return DATA_V1B


def get_procnet_format() -> Path:
    """Get the procnet_format directory."""
    return PROCNET_FORMAT


def get_w2ner_format() -> Path:
    """Get the W2NER format data directory."""
    return W2NER_FORMAT
