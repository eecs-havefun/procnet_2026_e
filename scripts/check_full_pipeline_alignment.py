#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据链路对齐（修正版）：
- data_v1b: Rasa 格式，只有 train/test
- procnet/procnet_format: ProcNet 文档格式 [[doc_id, data], ...]
- W2NER 格式：句子级别列表
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

# 数据目录
DATA_LOCATIONS = {
    "原始数据 (data_v1b)": Path("/home/mengfanrong/finaldesign/W2NERproject/data_v1b"),
    "ProcNet 格式 (procnet/procnet_format)": Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/procnet_format"),
    "W2NER 格式 (W2NER/data/data_w2ner_folded_with_dev)": Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev"),
    "W2NER 格式 (data_w2ner)": Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner"),
    "W2NER 格式 (data_w2ner_folded)": Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner_folded"),
}

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]

SPLITS = ["train", "dev", "test"]


def count_procnet_sentences(data):
    """统计 ProcNet 格式的句子数"""
    count = 0
    for item in data:
        if isinstance(item, list) and len(item) >= 2:
            doc_data = item[1]
            sentences = doc_data.get("sentences", [])
            count += len(sentences)
    return count


def count_w2ner_sentences(data):
    """统计 W2NER 格式的句子数"""
    return len(data)


def get_procnet_texts(data):
    """从 ProcNet 格式提取所有文本"""
    texts = []
    for item in data:
        if isinstance(item, list) and len(item) >= 2:
            doc_data = item[1]
            sentences = doc_data.get("sentences", [])
            for sent in sentences:
                if isinstance(sent, str):
                    texts.append(sent)
                elif isinstance(sent, list):
                    texts.append("".join(sent))
    return set(texts)


def get_w2ner_texts(data):
    """从 W2NER 格式提取所有文本"""
    return set(s.get("text", "") for s in data)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 120)
    print(" " * 40 + "数据链路对齐检查报告")
    print("=" * 120)
    
    for dataset in DATASETS:
        print(f"\n{'='*120}")
        print(f"【{dataset}】")
        print("=" * 120)
        
        for split in SPLITS:
            print(f"\n  --- {split}.json ---")
            
            # 收集各位置的数据
            sentence_counts = {}
            text_sets = {}
            
            for loc_name, loc_path in DATA_LOCATIONS.items():
                file_path = loc_path / dataset / f"{split}.json"
                
                if file_path.exists():
                    try:
                        data = load_json(file_path)
                        
                        if "ProcNet" in loc_name:
                            sentence_counts[loc_name] = count_procnet_sentences(data)
                            text_sets[loc_name] = get_procnet_texts(data)
                        else:
                            sentence_counts[loc_name] = count_w2ner_sentences(data)
                            text_sets[loc_name] = get_w2ner_texts(data)
                    except Exception as e:
                        sentence_counts[loc_name] = f"Error: {e}"
                else:
                    sentence_counts[loc_name] = "N/A"
            
            # 打印句子数
            print(f"\n  句子数对比:")
            print(f"  {'位置':<60} {'句子数':>12}")
            print(f"  {'-'*60} {'-'*12}")
            for loc_name, count in sentence_counts.items():
                print(f"  {loc_name:<60} {count:>12}")
            
            # 检查句子数是否一致（排除原始数据，因为它只有文档没有句子拆分）
            w2ner_counts = [c for name, c in sentence_counts.items() 
                          if isinstance(c, int) and "W2NER" in name]
            procnet_counts = [c for name, c in sentence_counts.items() 
                            if isinstance(c, int) and "ProcNet" in name]
            
            if len(w2ner_counts) > 1:
                if max(w2ner_counts) != min(w2ner_counts):
                    print(f"  ⚠️  W2NER 格式之间句子数不一致！max={max(w2ner_counts)}, min={min(w2ner_counts)}")
                else:
                    print(f"  ✅ W2NER 格式之间句子数一致：{w2ner_counts[0]}")
            
            if procnet_counts and w2ner_counts:
                if procnet_counts[0] != w2ner_counts[0]:
                    print(f"  ⚠️  ProcNet 与 W2NER 句子数不一致！ProcNet={procnet_counts[0]}, W2NER={w2ner_counts[0]}")
                else:
                    print(f"  ✅ ProcNet 与 W2NER 句子数一致：{procnet_counts[0]}")
            
            # 文本内容对比（W2NER 格式之间）
            print(f"\n  W2NER 格式之间文本对比:")
            w2ner_texts = {}
            for loc_name, texts in text_sets.items():
                if "W2NER" in loc_name:
                    w2ner_texts[loc_name] = texts
            
            if len(w2ner_texts) >= 2:
                ref_texts = list(w2ner_texts.values())[0]
                ref_name = list(w2ner_texts.keys())[0]
                
                for loc_name, texts in w2ner_texts.items():
                    if loc_name == ref_name:
                        continue
                    
                    overlap = len(ref_texts & texts)
                    only_ref = len(ref_texts - texts)
                    only_loc = len(texts - ref_texts)
                    
                    if len(ref_texts) > 0:
                        match_pct = overlap / len(ref_texts) * 100
                    else:
                        match_pct = 0
                    
                    if match_pct == 100 and only_loc == 0:
                        print(f"  ✅ {loc_name[:50]:<50} 100% 匹配")
                    else:
                        print(f"  ⚠️  {loc_name[:50]:<50} {match_pct:.1f}% (差异:{only_loc})")
    
    print("\n" + "=" * 120)
    print("检查完成")
    print("=" * 120)


if __name__ == "__main__":
    main()
