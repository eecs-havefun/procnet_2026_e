#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成完整数据链路，确保严格对齐：
1. 从 data_v1b 生成 ProcNet 格式（包含 train/dev/test 划分）
2. 从 ProcNet 生成 W2NER 格式（保持相同划分）

所有数据原地覆盖。
"""

import json
import re
import os
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# ========== 配置 ==========
DATA_V1B_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/data_v1b")
PROCNET_OUTPUT_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/procnet_format")
W2NER_OUTPUT_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")
W2NER_FOLDED_OUTPUT = Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner_folded")
W2NER_ALT_OUTPUT = Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner")

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]

# 划分比例
DEV_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# 角色折叠映射
ROLE_FOLD_MAP = {
    "startDate": "date",
    "endDate": "date",
    "startTime": "time",
    "endTime": "time",
}


# ========== 工具函数 ==========
def split_into_sentences(text: str) -> List[str]:
    """将文本分割成句子"""
    pattern = r'([。！？!?；;]+)'
    parts = re.split(pattern, text)
    
    sentences = []
    current_sentence = ""
    
    for part in parts:
        current_sentence += part
        if re.match(pattern, part):
            cleaned = current_sentence.strip()
            if cleaned:
                sentences.append(cleaned)
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences


def compute_sentence_offsets(sentences: List[str], original_text: str) -> List[int]:
    """计算每个句子在原文本中的起始位置"""
    offsets = []
    current_pos = 0
    
    for sent in sentences:
        pos = original_text.find(sent, current_pos)
        if pos != -1:
            offsets.append(pos)
            current_pos = pos + len(sent)
        else:
            offsets.append(current_pos)
            current_pos += len(sent)
    
    return offsets


def map_entity_to_sentences(
    entity_text: str,
    entity_start: int,
    entity_end: int,
    sentences: List[str],
    sentence_offsets: List[int]
) -> List[Tuple[int, int, int]]:
    """将实体位置映射到句子中"""
    positions = []
    
    for sent_idx, (sentence, sent_offset) in enumerate(zip(sentences, sentence_offsets)):
        rel_start = entity_start - sent_offset
        rel_end = entity_end - sent_offset
        
        if 0 <= rel_start < len(sentence) and 0 <= rel_end <= len(sentence):
            if sentence[rel_start:rel_end] == entity_text:
                positions.append((sent_idx, rel_start, rel_end))
    
    return positions


def load_rasa_data(path: Path) -> List[Dict]:
    """加载 Rasa NLU 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("rasa_nlu_data", {}).get("common_examples", [])


def save_json(data: Any, path: Path):
    """保存 JSON 数据"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ========== 转换函数 ==========
def convert_rasa_to_procnet_doc(
    example: Dict[str, Any],
    doc_id: str
) -> Dict[str, Any]:
    """将单个 Rasa example 转换为 ProcNet 文档"""
    text = example.get('text', '')
    intent = example.get('intent', 'unknown')
    entities = example.get('entities', [])
    
    if not text.strip():
        return None
    
    # 1. 分割句子
    sentences = split_into_sentences(text)
    if not sentences:
        return None
    
    # 2. 计算句子偏移量
    sentence_offsets = compute_sentence_offsets(sentences, text)
    
    # 3. 处理实体
    ann_valid_mspans = []
    ann_valid_dranges = []
    ann_mspan2dranges = defaultdict(list)
    ann_mspan2guess_field = {}
    
    processed_spans = set()
    
    for entity in entities:
        entity_text = entity.get('value', '')
        entity_type = entity.get('entity', 'unknown')
        entity_start = entity.get('start', -1)
        entity_end = entity.get('end', -1)
        
        if not entity_text or entity_start < 0 or entity_end < 0:
            continue
        
        positions = map_entity_to_sentences(
            entity_text, entity_start, entity_end,
            sentences, sentence_offsets
        )
        
        for sent_idx, start, end in positions:
            unique_key = f"{entity_text}#{sent_idx}_{start}_{end}#{entity_type}"
            
            span_key = (sent_idx, start, end, entity_type)
            if span_key in processed_spans:
                continue
            processed_spans.add(span_key)
            
            ann_valid_mspans.append(entity_text)
            
            drange = [sent_idx, start, end]
            ann_mspan2dranges[unique_key].append(drange)
            ann_valid_dranges.append(drange)
            
            ann_mspan2guess_field[unique_key] = entity_type
    
    # 4. 构建事件
    event_type = intent
    event_dict = {}
    for entity in entities:
        entity_text = entity.get('value', '')
        entity_type = entity.get('entity', 'unknown')
        if entity_text:
            event_dict[entity_type] = entity_text
    
    recguid_eventname_eventdict_list = []
    if event_dict:
        recguid_eventname_eventdict_list.append([0, event_type, event_dict])
    
    return {
        'sentences': sentences,
        'ann_valid_mspans': ann_valid_mspans,
        'ann_valid_dranges': ann_valid_dranges,
        'ann_mspan2dranges': dict(ann_mspan2dranges),
        'ann_mspan2guess_field': ann_mspan2guess_field,
        'recguid_eventname_eventdict_list': recguid_eventname_eventdict_list
    }


def procnet_doc_to_w2ner_samples(
    doc: Dict[str, Any],
    doc_id: str,
    fold_role_types: bool = False
) -> List[Dict[str, Any]]:
    """将 ProcNet 文档转换为 W2NER 样本列表"""
    samples = []
    
    sentences = doc.get("sentences", [])
    ann_mspan2dranges = doc.get("ann_mspan2dranges", {})
    ann_mspan2guess_field = doc.get("ann_mspan2guess_field", {})
    
    for sent_idx, sentence in enumerate(sentences):
        if isinstance(sentence, list):
            sent_text = "".join(sentence)
            sent_chars = sentence
        else:
            sent_text = sentence
            sent_chars = list(sentence)
        
        ner = []
        entities = []
        
        for mspan_key, dranges in ann_mspan2dranges.items():
            orig_type = ann_mspan2guess_field.get(mspan_key)
            if orig_type is None:
                continue
            
            if fold_role_types:
                entity_type = ROLE_FOLD_MAP.get(orig_type, orig_type)
            else:
                entity_type = orig_type
            
            entity_text = mspan_key.split("#", 1)[0] if "#" in mspan_key else mspan_key
            
            for dr in dranges:
                if not (isinstance(dr, list) and len(dr) == 3):
                    continue
                
                dr_sent_idx, start, end = dr
                if dr_sent_idx != sent_idx:
                    continue
                
                if not (0 <= start < end <= len(sent_text)):
                    continue
                
                if sent_text[start:end] != entity_text:
                    continue
                
                indices = list(range(start, end))
                ner.append({
                    "index": indices,
                    "type": entity_type,
                })
                
                entities.append({
                    "start": start,
                    "end": end,
                    "text": entity_text,
                    "type_name": entity_type,
                    "orig_type_name": orig_type,
                    "mspan_key": mspan_key,
                    "doc_id": doc_id,
                    "sent_idx": sent_idx,
                })
        
        samples.append({
            "sample_id": f"{doc_id}__sent_{sent_idx}",
            "doc_id": doc_id,
            "sent_id": sent_idx,
            "text": sent_text,
            "sentence": sent_chars,
            "ner": ner,
            "entities": entities,
        })
    
    return samples


def split_docs_into_sets(
    all_docs: List[Tuple[str, Dict]],
    dev_ratio: float = DEV_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED
) -> tuple:
    """将文档划分为 train/dev/test"""
    random.seed(seed)
    
    doc_ids = [doc[0] for doc in all_docs]
    random.shuffle(doc_ids)
    
    n_docs = len(doc_ids)
    dev_n = max(1, int(n_docs * dev_ratio))
    test_n = max(1, int(n_docs * test_ratio))
    
    test_ids = set(doc_ids[:test_n])
    dev_ids = set(doc_ids[test_n:test_n + dev_n])
    train_ids = set(doc_ids[test_n + dev_n:])
    
    train_docs = [(d, data) for d, data in all_docs if d in train_ids]
    dev_docs = [(d, data) for d, data in all_docs if d in dev_ids]
    test_docs = [(d, data) for d, data in all_docs if d in test_ids]
    
    return train_docs, dev_docs, test_docs


def process_dataset(dataset: str):
    """处理单个数据集"""
    print(f"\n{'='*70}")
    print(f"处理：{dataset}")
    print("=" * 70)
    
    # 1. 加载原始数据
    train_path = DATA_V1B_ROOT / dataset / "train.json"
    test_path = DATA_V1B_ROOT / dataset / "test.json"
    
    train_examples = load_rasa_data(train_path) if train_path.exists() else []
    test_examples = load_rasa_data(test_path) if test_path.exists() else []
    
    print(f"原始数据：train={len(train_examples)}, test={len(test_examples)}")
    
    # 2. 转换为 ProcNet 文档
    all_procnet_docs = []
    doc_id_counter = 0
    
    for example in train_examples + test_examples:
        doc_id = f"doc_{doc_id_counter:06d}"
        doc = convert_rasa_to_procnet_doc(example, doc_id)
        if doc:
            all_procnet_docs.append((doc_id, doc))
            doc_id_counter += 1
    
    print(f"ProcNet 文档：{len(all_procnet_docs)}")
    
    # 3. 划分为 train/dev/test
    train_docs, dev_docs, test_docs = split_docs_into_sets(all_procnet_docs)
    
    print(f"划分后：train={len(train_docs)}, dev={len(dev_docs)}, test={len(test_docs)}")
    
    # 4. 保存 ProcNet 格式
    for split_name, docs in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
        procnet_list = [[doc_id, doc] for doc_id, doc in docs]
        procnet_list.sort(key=lambda x: x[0])
        save_json(procnet_list, PROCNET_OUTPUT_ROOT / dataset / f"{split_name}.json")
    
    # 5. 生成 W2NER 格式（未折叠）
    for split_name, docs in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
        w2ner_samples = []
        for doc_id, doc in docs:
            samples = procnet_doc_to_w2ner_samples(doc, doc_id, fold_role_types=False)
            w2ner_samples.extend(samples)
        
        w2ner_samples.sort(key=lambda x: x["sample_id"])
        save_json(w2ner_samples, W2NER_OUTPUT_ROOT / dataset / f"{split_name}.json")
    
    # 6. 生成 W2NER 格式（折叠）- data_w2ner_folded
    for split_name, docs in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
        w2ner_samples = []
        for doc_id, doc in docs:
            samples = procnet_doc_to_w2ner_samples(doc, doc_id, fold_role_types=True)
            w2ner_samples.extend(samples)
        
        w2ner_samples.sort(key=lambda x: x["sample_id"])
        save_json(w2ner_samples, W2NER_FOLDED_OUTPUT / dataset / f"{split_name}.json")
    
    # 7. 生成 W2NER 格式（折叠）- data_w2ner
    for split_name, docs in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
        w2ner_samples = []
        for doc_id, doc in docs:
            samples = procnet_doc_to_w2ner_samples(doc, doc_id, fold_role_types=True)
            w2ner_samples.extend(samples)
        
        w2ner_samples.sort(key=lambda x: x["sample_id"])
        save_json(w2ner_samples, W2NER_ALT_OUTPUT / dataset / f"{split_name}.json")
    
    # 8. 统计
    print(f"\n统计:")
    for split_name, docs in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
        total_sentences = sum(len(doc[1].get('sentences', [])) for doc in docs)
        
        # 统计实体（未折叠）
        total_entities_unfolded = 0
        for _, doc in docs:
            total_entities_unfolded += len(doc.get('ann_valid_dranges', []))
        
        # 统计实体（折叠后）
        total_entities_folded = 0
        for _, doc in docs:
            for key, etype in doc.get('ann_mspan2guess_field', {}).items():
                folded_type = ROLE_FOLD_MAP.get(etype, etype)
                total_entities_folded += 1
        
        print(f"  {split_name}: {len(docs)} 文档，{total_sentences} 句子，{total_entities_unfolded} 实体")
    
    return {
        "train": len(train_docs),
        "dev": len(dev_docs),
        "test": len(test_docs),
    }


def main():
    print("=" * 80)
    print("重新生成完整数据链路 (从 data_v1b)")
    print("=" * 80)
    
    all_stats = {}
    
    for dataset in DATASETS:
        all_stats[dataset] = process_dataset(dataset)
    
    print("\n" + "=" * 80)
    print("生成完成!")
    print("=" * 80)
    
    print("\n汇总:")
    print(f"{'数据集':<35} {'train':>10} {'dev':>10} {'test':>10}")
    print("-" * 80)
    for dataset, stats in all_stats.items():
        print(f"{dataset:<35} {stats['train']:>10} {stats['dev']:>10} {stats['test']:>10}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
