#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整检查数据链路：
1. 原始数据 (data_v1b) → 有什么
2. 数据链路各阶段是否对齐
3. 是否有信息丢失（实体、句子等）
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

# 数据目录
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
    """加载 Rasa NLU 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("rasa_nlu_data", {}).get("common_examples", [])


def load_procnet_data(path):
    """加载 ProcNet 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_w2ner_data(path):
    """加载 W2NER 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_rasa_entities(examples):
    """统计 Rasa 格式的实体数"""
    total_entities = 0
    for ex in examples:
        total_entities += len(ex.get("entities", []))
    return total_entities


def count_procnet_sentences(docs):
    """统计 ProcNet 格式的句子数"""
    count = 0
    for doc in docs:
        if isinstance(doc, list) and len(doc) >= 2:
            count += len(doc[1].get("sentences", []))
    return count


def count_procnet_entities(docs):
    """统计 ProcNet 格式的实体数"""
    count = 0
    for doc in docs:
        if isinstance(doc, list) and len(doc) >= 2:
            count += len(doc[1].get("ann_valid_dranges", []))
    return count


def count_w2ner_sentences(samples):
    """统计 W2NER 格式的句子数"""
    return len(samples)


def count_w2ner_entities(samples):
    """统计 W2NER 格式的实体数"""
    total = 0
    for s in samples:
        total += len(s.get("ner", []))
    return total


def get_entity_types_rasa(examples):
    """获取 Rasa 格式的实体类型"""
    types = set()
    for ex in examples:
        for ent in ex.get("entities", []):
            types.add(ent.get("entity", "unknown"))
    return types


def get_entity_types_procnet(docs):
    """获取 ProcNet 格式的实体类型"""
    types = set()
    for doc in docs:
        if isinstance(doc, list) and len(doc) >= 2:
            field = doc[1].get("ann_mspan2guess_field", {})
            for t in field.values():
                types.add(t)
    return types


def get_entity_types_w2ner(samples):
    """获取 W2NER 格式的实体类型"""
    types = set()
    for s in samples:
        for ent in s.get("ner", []):
            types.add(ent.get("type", "unknown"))
    return types


def main():
    print("=" * 100)
    print(" " * 30 + "数据链路完整检查报告")
    print("=" * 100)
    
    for dataset in DATASETS:
        print(f"\n{'='*100}")
        print(f"【{dataset}】")
        print("=" * 100)
        
        stats = {}
        entity_types = {}
        
        # ========== 1. 原始数据 (data_v1b) ==========
        print(f"\n1. 原始数据 (data_v1b - Rasa NLU 格式)")
        print("-" * 50)
        
        for split in ["train", "test"]:
            path = DATA_V1B / dataset / f"{split}.json"
            if path.exists():
                data = load_rasa_data(path)
                stats[f"v1b_{split}_samples"] = len(data)
                stats[f"v1b_{split}_sentences"] = "N/A"  # Rasa 格式没有明确句子分割
                stats[f"v1b_{split}_entities"] = count_rasa_entities(data)
                entity_types[f"v1b_{split}"] = get_entity_types_rasa(data)
                
                print(f"  {split}: {len(data)} 样本，{count_rasa_entities(data)} 实体，"
                      f"{len(entity_types[f'v1b_{split}'])} 种类型")
            else:
                print(f"  {split}: ❌ 文件不存在")
                stats[f"v1b_{split}_samples"] = 0
        
        print(f"  dev: ❌ 无 dev.json (原始数据只有 train/test)")
        
        # ========== 2. ProcNet 格式 ==========
        print(f"\n2. ProcNet 格式 (procnet/procnet_format)")
        print("-" * 50)
        
        for split in ["train", "dev", "test"]:
            path = PROCNET_FORMAT / dataset / f"{split}.json"
            if path.exists():
                data = load_procnet_data(path)
                stats[f"procnet_{split}_sentences"] = count_procnet_sentences(data)
                stats[f"procnet_{split}_entities"] = count_procnet_entities(data)
                entity_types[f"procnet_{split}"] = get_entity_types_procnet(data)
                
                print(f"  {split}: {count_procnet_sentences(data)} 句子，"
                      f"{count_procnet_entities(data)} 实体，"
                      f"{len(entity_types[f'procnet_{split}'])} 种类型")
            else:
                print(f"  {split}: ❌ 文件不存在")
        
        # ========== 3. W2NER 格式 ==========
        print(f"\n3. W2NER 格式 (W2NER/data/data_w2ner_folded_with_dev)")
        print("-" * 50)
        
        for split in ["train", "dev", "test"]:
            path = W2NER_FORMAT / dataset / f"{split}.json"
            if path.exists():
                data = load_w2ner_data(path)
                stats[f"w2ner_{split}_sentences"] = count_w2ner_sentences(data)
                stats[f"w2ner_{split}_entities"] = count_w2ner_entities(data)
                entity_types[f"w2ner_{split}"] = get_entity_types_w2ner(data)
                
                print(f"  {split}: {count_w2ner_sentences(data)} 句子，"
                      f"{count_w2ner_entities(data)} 实体，"
                      f"{len(entity_types[f'w2ner_{split}'])} 种类型")
            else:
                print(f"  {split}: ❌ 文件不存在")
        
        # ========== 4. 对齐检查 ==========
        print(f"\n4. 数据链路对齐检查")
        print("-" * 50)
        
        issues = []
        
        # 检查 ProcNet 和 W2NER 是否对齐
        for split in ["train", "dev", "test"]:
            procnet_sents = stats.get(f"procnet_{split}_sentences", 0)
            w2ner_sents = stats.get(f"w2ner_{split}_sentences", 0)
            procnet_ents = stats.get(f"procnet_{split}_entities", 0)
            w2ner_ents = stats.get(f"w2ner_{split}_entities", 0)
            
            if procnet_sents == w2ner_sents and procnet_ents == w2ner_ents:
                print(f"  ✅ {split}: ProcNet 和 W2NER 完全对齐 "
                      f"(句子={procnet_sents}, 实体={procnet_ents})")
            else:
                print(f"  ❌ {split}: ProcNet 和 W2NER 不对齐!")
                print(f"      ProcNet: {procnet_sents} 句子，{procnet_ents} 实体")
                print(f"      W2NER:   {w2ner_sents} 句子，{w2ner_ents} 实体")
                issues.append(f"{split} 不对齐")
        
        # 检查实体类型是否一致
        print(f"\n5. 实体类型对比")
        print("-" * 50)
        
        procnet_train_types = entity_types.get("procnet_train", set())
        w2ner_train_types = entity_types.get("w2ner_train", set())
        
        if procnet_train_types == w2ner_train_types:
            print(f"  ✅ ProcNet 和 W2NER 实体类型一致 ({len(procnet_train_types)} 种)")
        else:
            added = w2ner_train_types - procnet_train_types
            missing = procnet_train_types - w2ner_train_types
            print(f"  ⚠️  实体类型不一致:")
            if added:
                print(f"      W2NER 新增：{sorted(added)}")
            if missing:
                print(f"      W2NER 缺失：{sorted(missing)}")
        
        # 总结
        print(f"\n{'='*100}")
        if not issues:
            print(f"✅ {dataset}: 数据链路完全对齐!")
        else:
            print(f"❌ {dataset}: 发现以下问题:")
            for issue in issues:
                print(f"   - {issue}")
    
    print("\n" + "=" * 100)
    print("检查完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
