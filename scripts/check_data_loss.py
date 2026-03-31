#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据链路是否有信息丢失：
原始数据 (data_v1b) → ProcNet → W2NER
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_V1B = Path("/home/mengfanrong/finaldesign/W2NERproject/data_v1b")
PROCNET_FORMAT = Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/procnet_format")
W2NER_FORMAT = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]


def load_rasa_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("rasa_nlu_data", {}).get("common_examples", [])


def load_procnet_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_w2ner_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_rasa_entities(examples):
    """从 Rasa 数据提取所有实体 (text, type) 对"""
    entities = set()
    for ex in examples:
        text = ex.get("text", "")
        for ent in ex.get("entities", []):
            ent_text = ent.get("value", "")
            ent_type = ent.get("entity", "")
            entities.add((ent_text, ent_type))
    return entities


def extract_procnet_entities(docs):
    """从 ProcNet 数据提取所有实体 (text, type) 对"""
    entities = set()
    for doc in docs:
        if isinstance(doc, list) and len(doc) >= 2:
            field = doc[1].get("ann_mspan2guess_field", {})
            for key, ent_type in field.items():
                # 从 unique_key 恢复实体文本
                ent_text = key.split("#")[0] if "#" in key else key
                entities.add((ent_text, ent_type))
    return entities


def extract_w2ner_entities(samples):
    """从 W2NER 数据提取所有实体 (text, type) 对"""
    entities = set()
    for s in samples:
        for ent in s.get("ner", []):
            indices = ent.get("index", [])
            ent_type = ent.get("type", "")
            # 从 sentence 恢复实体文本
            sentence_chars = s.get("sentence", [])
            if indices and sentence_chars:
                ent_text = "".join(sentence_chars[i] for i in indices)
                entities.add((ent_text, ent_type))
    return entities


def main():
    print("=" * 100)
    print(" " * 25 + "数据链路信息丢失检查")
    print("=" * 100)
    
    for dataset in DATASETS:
        print(f"\n{'='*100}")
        print(f"【{dataset}】")
        print("=" * 100)
        
        # ========== 1. 加载数据 ==========
        v1b_train = load_rasa_data(DATA_V1B / dataset / "train.json")
        v1b_test = load_rasa_data(DATA_V1B / dataset / "test.json")
        
        procnet_train = load_procnet_data(PROCNET_FORMAT / dataset / "train.json")
        procnet_dev = load_procnet_data(PROCNET_FORMAT / dataset / "dev.json")
        procnet_test = load_procnet_data(PROCNET_FORMAT / dataset / "test.json")
        
        w2ner_train = load_w2ner_data(W2NER_FORMAT / dataset / "train.json")
        w2ner_dev = load_w2ner_data(W2NER_FORMAT / dataset / "dev.json")
        w2ner_test = load_w2ner_data(W2NER_FORMAT / dataset / "test.json")
        
        # ========== 2. 提取实体 ==========
        v1b_train_ents = extract_rasa_entities(v1b_train)
        v1b_test_ents = extract_rasa_entities(v1b_test)
        
        # 合并 ProcNet train+dev+test
        procnet_all_ents = extract_procnet_entities(procnet_train) | \
                          extract_procnet_entities(procnet_dev) | \
                          extract_procnet_entities(procnet_test)
        
        # 合并 W2NER train+dev+test
        w2ner_all_ents = extract_w2ner_entities(w2ner_train) | \
                        extract_w2ner_entities(w2ner_dev) | \
                        extract_w2ner_entities(w2ner_test)
        
        # ========== 3. 对比 ==========
        print(f"\n原始数据 (v1b) 实体：{len(v1b_train_ents | v1b_test_ents)} 种")
        print(f"ProcNet 全量实体：{len(procnet_all_ents)} 种")
        print(f"W2NER 全量实体：{len(w2ner_all_ents)} 种")
        
        # 检查实体类型
        v1b_types = set(t for _, t in v1b_train_ents | v1b_test_ents)
        procnet_types = set(t for _, t in procnet_all_ents)
        w2ner_types = set(t for _, t in w2ner_all_ents)
        
        print(f"\n实体类型对比:")
        print(f"  原始数据 (v1b): {len(v1b_types)} 种 - {sorted(v1b_types)}")
        print(f"  ProcNet:        {len(procnet_types)} 种 - {sorted(procnet_types)}")
        print(f"  W2NER:          {len(w2ner_types)} 种 - {sorted(w2ner_types)}")
        
        # 检查是否有类型丢失
        lost_types = v1b_types - procnet_types
        if lost_types:
            print(f"\n  ⚠️  丢失的实体类型：{sorted(lost_types)}")
        
        # 检查实体提及丢失
        v1b_entity_texts = set(t for t, _ in v1b_train_ents | v1b_test_ents)
        procnet_entity_texts = set(t for t, _ in procnet_all_ents)
        w2ner_entity_texts = set(t for t, _ in w2ner_all_ents)
        
        lost_texts = v1b_entity_texts - procnet_entity_texts
        if lost_texts:
            print(f"\n  ⚠️  丢失的实体文本提及：{len(lost_texts)} 个")
            if len(lost_texts) <= 10:
                for t in sorted(lost_texts):
                    print(f"      - {t}")
            else:
                print(f"      (前 10 个): {sorted(lost_texts)[:10]}")
        
        # 检查是否有新增（可能是数据增强或分割导致）
        new_texts = procnet_entity_texts - v1b_entity_texts
        if new_texts:
            print(f"\n  ℹ️  新增的实体文本提及：{len(new_texts)} 个")
            if len(new_texts) <= 10:
                for t in sorted(new_texts):
                    print(f"      - {t}")
        
        # ========== 4. 句子分割检查 ==========
        print(f"\n句子分割检查:")
        
        # 统计原始数据的总文本长度
        v1b_train_text_len = sum(len(ex.get("text", "")) for ex in v1b_train)
        v1b_test_text_len = sum(len(ex.get("text", "")) for ex in v1b_test)
        
        # 统计 ProcNet 的总文本长度
        procnet_text_len = 0
        for doc in procnet_train + procnet_dev + procnet_test:
            if isinstance(doc, list) and len(doc) >= 2:
                for sent in doc[1].get("sentences", []):
                    procnet_text_len += len(sent)
        
        # 统计 W2NER 的总文本长度
        w2ner_text_len = 0
        for s in w2ner_train + w2ner_dev + w2ner_test:
            w2ner_text_len += len(s.get("text", ""))
        
        print(f"  原始数据总文本长度：{v1b_train_text_len + v1b_test_text_len:,}")
        print(f"  ProcNet 总文本长度：{procnet_text_len:,}")
        print(f"  W2NER 总文本长度：{w2ner_text_len:,}")
        
        # 检查文本是否完整保留
        if abs(procnet_text_len - w2ner_text_len) > 100:
            print(f"  ⚠️  ProcNet 和 W2NER 文本长度差异较大")
        
        print(f"\n{'='*100}")
    
    print("\n" + "=" * 100)
    print("检查完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
