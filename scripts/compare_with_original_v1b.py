#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与最原始数据 data_v1b 对比，检查数据链路是否正确：
1. 文本内容是否完整保留
2. 实体标注是否正确保留
3. 是否有信息丢失
"""

import json
import re
from pathlib import Path
from collections import defaultdict

DATA_V1B = Path("/home/mengfanrong/finaldesign/W2NERproject/data_v1b")
W2NER_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")

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


def load_w2ner_data(path):
    """加载 W2NER 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_into_sentences(text):
    """与转换脚本相同的句子分割逻辑"""
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


def extract_rasa_entities(example):
    """从 Rasa 示例提取实体"""
    entities = []
    text = example.get("text", "")
    
    for ent in example.get("entities", []):
        ent_text = ent.get("value", "")
        ent_type = ent.get("entity", "")
        start = ent.get("start", -1)
        end = ent.get("end", -1)
        
        # 验证实体文本
        if start >= 0 and end <= len(text):
            actual_text = text[start:end]
            if actual_text != ent_text:
                print(f"    ⚠️  实体文本不匹配：标注='{ent_text}', 实际='{actual_text}'")
        
        entities.append({
            "text": ent_text,
            "type": ent_type,
            "start": start,
            "end": end,
        })
    
    return entities


def find_entity_in_w2ner(ent_text, ent_type, w2ner_samples):
    """在 W2NER 数据中查找实体"""
    found = []
    for sample in w2ner_samples:
        sentence_chars = sample.get("sentence", [])
        sent_text = "".join(sentence_chars)
        
        for ner in sample.get("ner", []):
            indices = ner.get("index", [])
            ner_type = ner.get("type", "")
            
            if indices and sentence_chars:
                ner_text = "".join(sentence_chars[i] for i in indices)
                if ner_text == ent_text and ner_type == ent_type:
                    found.append({
                        "sample_id": sample.get("sample_id"),
                        "text": ner_text,
                        "type": ner_type,
                    })
    
    return found


def main():
    print("=" * 100)
    print(" " * 25 + "与原始数据 data_v1b 对比检查")
    print("=" * 100)
    
    for dataset in DATASETS:
        print(f"\n{'='*100}")
        print(f"【{dataset}】")
        print("=" * 100)
        
        # 加载原始数据
        v1b_train_path = DATA_V1B / dataset / "train.json"
        v1b_test_path = DATA_V1B / dataset / "test.json"
        
        if not v1b_train_path.exists():
            print(f"  ❌ 原始数据不存在：{v1b_train_path}")
            continue
        
        v1b_train = load_rasa_data(v1b_train_path)
        v1b_test = load_rasa_data(v1b_test_path) if v1b_test_path.exists() else []
        
        # 加载 W2NER 数据
        w2ner_train_path = W2NER_PATH / dataset / "train.json"
        w2ner_test_path = W2NER_PATH / dataset / "test.json"
        
        w2ner_train = load_w2ner_data(w2ner_train_path) if w2ner_train_path.exists() else []
        w2ner_test = load_w2ner_data(w2ner_test_path) if w2ner_test_path.exists() else []
        
        print(f"\n1. 样本数对比")
        print("-" * 60)
        print(f"  原始数据 train: {len(v1b_train)} 样本")
        print(f"  原始数据 test:  {len(v1b_test)} 样本")
        print(f"  W2NER train:    {len(w2ner_train)} 样本 (句子)")
        print(f"  W2NER test:     {len(w2ner_test)} 样本 (句子)")
        
        # 计算原始数据分割后的句子数
        v1b_train_sentences = sum(len(split_into_sentences(ex.get("text", ""))) for ex in v1b_train)
        v1b_test_sentences = sum(len(split_into_sentences(ex.get("text", ""))) for ex in v1b_test)
        
        print(f"\n  原始数据 train 分割后约: {v1b_train_sentences} 句子")
        print(f"  原始数据 test 分割后约:  {v1b_test_sentences} 句子")
        
        # 检查文本内容是否保留
        print(f"\n2. 文本内容检查")
        print("-" * 60)
        
        # 提取原始数据的所有文本
        v1b_texts = set()
        for ex in v1b_train + v1b_test:
            text = ex.get("text", "")
            # 分割成句子
            sents = split_into_sentences(text)
            for sent in sents:
                v1b_texts.add(sent)
        
        # 提取 W2NER 的所有文本
        w2ner_texts = set()
        for sample in w2ner_train + w2ner_test:
            text = sample.get("text", "")
            w2ner_texts.add(text)
        
        # 对比
        missing_texts = v1b_texts - w2ner_texts
        extra_texts = w2ner_texts - v1b_texts
        
        print(f"  原始数据句子数：{len(v1b_texts)}")
        print(f"  W2NER 句子数：{len(w2ner_texts)}")
        print(f"  缺失的句子：{len(missing_texts)}")
        print(f"  额外的句子：{len(extra_texts)}")
        
        if missing_texts:
            print(f"\n  缺失的句子示例 (前 5 个):")
            for text in list(missing_texts)[:5]:
                print(f"    - {text[:80]}...")
        
        # 检查实体标注
        print(f"\n3. 实体标注检查")
        print("-" * 60)
        
        # 统计原始数据的实体
        v1b_entities = defaultdict(int)
        v1b_entity_set = set()
        for ex in v1b_train + v1b_test:
            for ent in ex.get("entities", []):
                ent_text = ent.get("value", "")
                ent_type = ent.get("entity", "")
                v1b_entities[ent_type] += 1
                v1b_entity_set.add((ent_text, ent_type))
        
        # 统计 W2NER 的实体
        w2ner_entities = defaultdict(int)
        w2ner_entity_set = set()
        for sample in w2ner_train + w2ner_test:
            for ner in sample.get("ner", []):
                indices = ner.get("index", [])
                ner_type = ner.get("type", "")
                sentence_chars = sample.get("sentence", [])
                if indices and sentence_chars:
                    ner_text = "".join(sentence_chars[i] for i in indices)
                    w2ner_entities[ner_type] += 1
                    w2ner_entity_set.add((ner_text, ner_type))
        
        print(f"  原始数据实体类型分布:")
        for etype, count in sorted(v1b_entities.items(), key=lambda x: -x[1])[:10]:
            print(f"    {etype}: {count}")
        
        print(f"\n  W2NER 实体类型分布:")
        for etype, count in sorted(w2ner_entities.items(), key=lambda x: -x[1])[:10]:
            print(f"    {etype}: {count}")
        
        # 检查实体类型映射
        print(f"\n  实体类型对比:")
        v1b_types = set(v1b_entities.keys())
        w2ner_types = set(w2ner_entities.keys())
        
        print(f"    原始数据类型：{len(v1b_types)} 种 - {sorted(v1b_types)}")
        print(f"    W2NER 数据类型：{len(w2ner_types)} 种 - {sorted(w2ner_types)}")
        
        merged_types = v1b_types - w2ner_types
        new_types = w2ner_types - v1b_types
        
        if merged_types:
            print(f"\n    ⚠️  原始数据有但 W2NER 没有的类型：{sorted(merged_types)}")
            print(f"        (可能被合并了，如 startDate/endDate -> date)")
        
        if new_types:
            print(f"\n    ⚠️  W2NER 新增的类型：{sorted(new_types)}")
        
        # 检查具体实体是否保留
        print(f"\n4. 实体提及保留检查")
        print("-" * 60)
        
        missing_entities = v1b_entity_set - w2ner_entity_set
        extra_entities = w2ner_entity_set - v1b_entity_set
        
        print(f"  原始数据唯一实体提及：{len(v1b_entity_set)}")
        print(f"  W2NER 唯一实体提及：{len(w2ner_entity_set)}")
        print(f"  丢失的实体提及：{len(missing_entities)}")
        print(f"  新增的实体提及：{len(extra_entities)}")
        
        if missing_entities:
            print(f"\n  丢失的实体示例 (前 10 个):")
            for ent_text, ent_type in list(missing_entities)[:10]:
                print(f"    - {ent_text} ({ent_type})")
        
        # 总结
        print(f"\n{'='*100}")
        print(f"【{dataset}】总结:")
        
        issues = []
        
        if len(missing_texts) > len(v1b_texts) * 0.1:  # 超过 10% 的句子丢失
            issues.append(f"大量句子丢失 ({len(missing_texts)}/{len(v1b_texts)})")
        
        if len(missing_entities) > len(v1b_entity_set) * 0.1:  # 超过 10% 的实体丢失
            issues.append(f"大量实体丢失 ({len(missing_entities)}/{len(v1b_entity_set)})")
        
        if issues:
            print(f"  ❌ 发现问题:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"  ✅ 数据基本正确，无明显信息丢失")
    
    print("\n" + "=" * 100)
    print("检查完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
