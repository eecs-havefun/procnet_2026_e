#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 procnet/Data_v1b 是否是正确的数据源
"""

import json
from pathlib import Path

DATA_V1B_PROCNET = Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/Data_v1b")
W2NER_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")

DATASET = "flight_orders_with_queries"


def load_procnet_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_w2ner_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_procnet_sentences(docs):
    count = 0
    for doc in docs:
        if isinstance(doc, list) and len(doc) >= 2:
            count += len(doc[1].get("sentences", []))
    return count


def count_procnet_entities(docs):
    count = 0
    for doc in docs:
        if isinstance(doc, list) and len(doc) >= 2:
            count += len(doc[1].get("ann_valid_dranges", []))
    return count


def main():
    print("=" * 80)
    print("检查 procnet/Data_v1b 数据源")
    print("=" * 80)
    
    for split in ["train", "dev", "test"]:
        print(f"\n--- {split}.json ---")
        
        # 加载 Data_v1b ProcNet 格式
        v1b_path = DATA_V1B_PROCNET / DATASET / f"{split}.json"
        if v1b_path.exists():
            v1b_data = load_procnet_data(v1b_path)
            v1b_sents = count_procnet_sentences(v1b_data)
            v1b_ents = count_procnet_entities(v1b_data)
            print(f"procnet/Data_v1b:")
            print(f"  文档数：{len(v1b_data)}")
            print(f"  句子数：{v1b_sents}")
            print(f"  实体数：{v1b_ents}")
        else:
            print(f"procnet/Data_v1b: 不存在")
        
        # 加载 W2NER 格式
        w2ner_path = W2NER_PATH / DATASET / f"{split}.json"
        if w2ner_path.exists():
            w2ner_data = load_w2ner_data(w2ner_path)
            print(f"W2NER/data_w2ner_folded_with_dev:")
            print(f"  样本数：{len(w2ner_data)}")
            print(f"  实体数：{sum(len(s.get('ner', [])) for s in w2ner_data)}")
        else:
            print(f"W2NER/data_w2ner_folded_with_dev: 不存在")
    
    # 检查第一个文档的内容
    print(f"\n{'='*60}")
    print("检查第一个文档内容")
    
    v1b_path = DATA_V1B_PROCNET / DATASET / "train.json"
    if v1b_path.exists():
        v1b_data = load_procnet_data(v1b_path)
        print(f"\nprocnet/Data_v1b 第一个文档:")
        doc = v1b_data[0]
        print(f"  doc_id: {doc[0]}")
        print(f"  句子数：{len(doc[1].get('sentences', []))}")
        for i, sent in enumerate(doc[1].get('sentences', [])[:3]):
            print(f"    [{i}] {sent[:60]}...")
    
    w2ner_path = W2NER_PATH / DATASET / "train.json"
    if w2ner_path.exists():
        w2ner_data = load_w2ner_data(w2ner_path)
        print(f"\nW2NER 第一个样本:")
        sample = w2ner_data[0]
        print(f"  sample_id: {sample.get('sample_id')}")
        print(f"  doc_id: {sample.get('doc_id')}")
        print(f"  text: {sample.get('text', '')[:60]}...")
    
    print(f"\n{'='*60}")
    print("结论:")
    print("如果 procnet/Data_v1b 是正确的数据源，应该从它重新生成 W2NER 格式。")
    print("=" * 80)


if __name__ == "__main__":
    main()
