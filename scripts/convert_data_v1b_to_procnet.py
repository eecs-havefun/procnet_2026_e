#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 data_v1b (RASA NLU 格式) 转换为 ProcNet 格式

输入格式 (RASA NLU):
{
  "rasa_nlu_data": {
    "common_examples": [
      {
        "text": "订单文本内容...",
        "intent": "order_type",
        "entities": [
          {"start": 0, "end": 5, "value": "实体值", "entity": "实体类型"}
        ]
      }
    ]
  }
}

输出格式 (ProcNet):
[
  [
    "doc_id",
    {
      "sentences": ["句子 1", "句子 2", ...],
      "ann_valid_mspans": ["实体提及 1", "实体提及 2", ...],
      "ann_valid_dranges": [[sent_idx, start, end], ...],
      "ann_mspan2dranges": {"实体提及": [[sent_idx, start, end], ...]},
      "ann_mspan2guess_field": {"实体提及": "实体类型"},
      "recguid_eventname_eventdict_list": [[0, "事件类型", {"角色": "实体提及", ...}]]
    }
  ]
]

使用方法:
  python convert_data_v1b_to_procnet.py \
    --input_dir data_v1b \
    --output_dir procnet/Data_v1b \
    --split train
"""

import json
import os
import re
import argparse
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def split_into_sentences(text: str) -> List[str]:
    """
    将文本分割成句子
    
    规则：
    - 按句号、问号、感叹号、分号分割
    - 保留标点符号在句子末尾
    - 过滤空句子
    """
    # 匹配中文和英文的句子结束标点
    pattern = r'([。！？!?；;]+)'
    
    # 分割文本
    parts = re.split(pattern, text)
    
    sentences = []
    current_sentence = ""
    
    for i, part in enumerate(parts):
        current_sentence += part
        # 如果当前部分是标点符号，则一个句子结束
        if re.match(pattern, part):
            # 清理句子（去除首尾空白）
            cleaned = current_sentence.strip()
            if cleaned:
                sentences.append(cleaned)
            current_sentence = ""
    
    # 处理剩余部分（没有标点结尾的）
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences


def find_entity_in_sentences_with_offset(entity_text: str, sentences: List[str], 
                                          sentence_offsets: List[int]) -> List[Tuple[int, int, int]]:
    """
    在句子中找到实体的位置（使用句子偏移量计算绝对位置）
    
    Args:
        entity_text: 实体文本
        sentences: 句子列表
        sentence_offsets: 每个句子在原文本中的起始位置
    
    返回：[(sent_idx, start_in_sentence, end_in_sentence), ...]
    """
    positions = []
    
    for sent_idx, sentence in enumerate(sentences):
        start = sentence.find(entity_text)
        if start != -1:
            end = start + len(entity_text)
            positions.append((sent_idx, start, end))
    
    return positions


def map_entity_to_sentences(entity_start: int, entity_end: int, entity_text: str,
                            sentences: List[str], sentence_offsets: List[int]) -> List[Tuple[int, int, int]]:
    """
    将原始标注的实体位置映射到分割后的句子中
    
    Args:
        entity_start: 实体在原文本中的起始位置
        entity_end: 实体在原文本中的结束位置
        entity_text: 实体文本
        sentences: 句子列表
        sentence_offsets: 每个句子在原文本中的起始位置
    
    Returns:
        [(sent_idx, start_in_sentence, end_in_sentence), ...]
    """
    positions = []
    
    for sent_idx, (sentence, sent_offset) in enumerate(zip(sentences, sentence_offsets)):
        # 计算实体相对于句子的位置
        rel_start = entity_start - sent_offset
        rel_end = entity_end - sent_offset
        
        # 检查实体是否完全在这个句子内
        if 0 <= rel_start < len(sentence) and 0 <= rel_end <= len(sentence):
            # 验证提取的文本是否匹配
            if sentence[rel_start:rel_end] == entity_text:
                positions.append((sent_idx, rel_start, rel_end))
    
    return positions


def compute_sentence_offsets(sentences: List[str], original_text: str) -> List[int]:
    """
    计算每个句子在原文本中的起始位置
    
    返回：[sent0_offset, sent1_offset, ...]
    """
    offsets = []
    current_pos = 0
    
    for sent in sentences:
        # 在原文本中查找句子的位置
        pos = original_text.find(sent, current_pos)
        if pos != -1:
            offsets.append(pos)
            current_pos = pos + len(sent)
        else:
            # 如果找不到，使用当前位置（可能有不匹配）
            offsets.append(current_pos)
            current_pos += len(sent)
    
    return offsets


def convert_rasa_example_to_procnet(
    example: Dict[str, Any],
    doc_id: str,
    event_type_mapping: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    将单个 RASA example 转换为 ProcNet 文档格式

    【重要修复】保留重复实体（同一文本跨度有多个类型）
    
    原始问题：
      - 原始代码使用 entity_text 作为 key，导致同一文本的多个类型被覆盖
      - 例如："11 月 11 日" -> startDate 和 endDate，只保留了最后一个
    
    修复方案：
      - 使用唯一 key：f"{entity_text}#{sent_idx}_{start}_{end}#{entity_type}"
      - 这样每个 (位置，类型) 组合都有独立的 entry
      - 同时保留原始的 entity_text 用于显示
    
    【位置映射修复】
      - 原始标注的实体位置是相对于完整文本的
      - 句子分割后，需要计算实体在每个句子中的相对位置
      - 使用 map_entity_to_sentences 函数进行正确的映射

    Args:
        example: RASA example，包含 text, intent, entities
        doc_id: 文档 ID
        event_type_mapping: intent 到事件类型的映射

    Returns:
        ProcNet 文档内容（不包含 doc_id）
    """
    text = example.get('text', '')
    intent = example.get('intent', 'unknown')
    entities = example.get('entities', [])

    # 1. 分割句子
    sentences = split_into_sentences(text)

    # 如果没有句子，添加一个空句子
    if not sentences:
        sentences = ['']
    
    # 2. 计算每个句子在原文本中的起始位置
    sentence_offsets = []
    current_pos = 0
    for sent in sentences:
        pos = text.find(sent, current_pos)
        if pos != -1:
            sentence_offsets.append(pos)
            current_pos = pos + len(sent)
        else:
            # 如果找不到，使用当前位置
            sentence_offsets.append(current_pos)
            current_pos += len(sent)

    # 3. 处理实体
    ann_valid_mspans = []  # 实体提及列表（显示用）
    ann_valid_dranges = []  # 实体提及的位置 [sent_idx, start, end]
    ann_mspan2dranges = defaultdict(list)  # 唯一 key -> 位置列表
    ann_mspan2guess_field = {}  # 唯一 key -> 实体类型
    
    # 用于跟踪已处理的 (sent_idx, start, end, entity_type) 组合，避免重复
    processed_spans = set()

    for entity in entities:
        entity_text = entity.get('value', '')
        entity_type = entity.get('entity', 'unknown')
        entity_start = entity.get('start', -1)
        entity_end = entity.get('end', -1)

        # 跳过空实体
        if not entity_text or entity_start < 0 or entity_end < 0:
            continue

        # 找到实体在哪个句子中（使用正确的相对位置计算）
        positions = []
        for sent_idx, (sentence, sent_offset) in enumerate(zip(sentences, sentence_offsets)):
            # 计算实体相对于句子的位置
            rel_start = entity_start - sent_offset
            rel_end = entity_end - sent_offset
            
            # 检查实体是否完全在这个句子内
            if 0 <= rel_start < len(sentence) and 0 <= rel_end <= len(sentence):
                # 验证提取的文本是否匹配
                if sentence[rel_start:rel_end] == entity_text:
                    positions.append((sent_idx, rel_start, rel_end))

        if positions:
            # 为每个位置创建唯一 key
            for sent_idx, start, end in positions:
                # 创建唯一 key：包含文本、位置和类型
                unique_key = f"{entity_text}#{sent_idx}_{start}_{end}#{entity_type}"
                
                # 检查是否已处理过这个组合
                span_key = (sent_idx, start, end, entity_type)
                if span_key in processed_spans:
                    continue
                processed_spans.add(span_key)
                
                # 添加到提及列表（用于显示，允许重复文本）
                ann_valid_mspans.append(entity_text)

                # 添加位置
                drange = [sent_idx, start, end]
                ann_mspan2dranges[unique_key].append(drange)
                ann_valid_dranges.append(drange)

                # 添加实体类型映射
                ann_mspan2guess_field[unique_key] = entity_type

    # 4. 构建事件
    # 将 intent 映射到事件类型
    event_type = event_type_mapping.get(intent, intent) if event_type_mapping else intent

    # 构建事件参数字典（保留所有类型）
    event_dict = {}
    for entity in entities:
        entity_text = entity.get('value', '')
        entity_type = entity.get('entity', 'unknown')

        if entity_text:
            # 使用实体类型作为角色名
            role_name = entity_type
            event_dict[role_name] = entity_text

    # 构建事件列表
    # 格式：[[0, "事件类型", {"角色": "实体提及", ...}]]
    recguid_eventname_eventdict_list = []
    if event_dict:
        recguid_eventname_eventdict_list.append([0, event_type, event_dict])

    # 5. 构建 ProcNet 文档
    procnet_doc = {
        'sentences': sentences,
        'ann_valid_mspans': ann_valid_mspans,
        'ann_valid_dranges': ann_valid_dranges,
        'ann_mspan2dranges': dict(ann_mspan2dranges),
        'ann_mspan2guess_field': ann_mspan2guess_field,
        'recguid_eventname_eventdict_list': recguid_eventname_eventdict_list
    }

    return procnet_doc


def convert_rasa_file_to_procnet(
    input_file: str,
    output_file: str,
    event_type_mapping: Dict[str, str] = None,
    max_docs: int = None
):
    """
    转换整个 RASA 文件到 ProcNet 格式
    
    Args:
        input_file: 输入文件路径 (RASA JSON 格式)
        output_file: 输出文件路径 (ProcNet JSON 格式)
        event_type_mapping: intent 到事件类型的映射
        max_docs: 最大转换文档数（用于测试）
    """
    # 读取 RASA 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        rasa_data = json.load(f)
    
    # 获取 common_examples
    rasa_nlu_data = rasa_data.get('rasa_nlu_data', {})
    examples = rasa_nlu_data.get('common_examples', [])
    
    if max_docs:
        examples = examples[:max_docs]
    
    # 转换每个 example
    procnet_docs = []
    
    for idx, example in enumerate(examples):
        # 生成文档 ID
        doc_id = f"doc_{idx:06d}"
        
        # 转换
        procnet_content = convert_rasa_example_to_procnet(
            example, doc_id, event_type_mapping
        )
        
        # 添加到结果
        procnet_docs.append([doc_id, procnet_content])
        
        if (idx + 1) % 100 == 0:
            print(f"  已转换 {idx + 1}/{len(examples)} 篇文档...")
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(procnet_docs, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成：{len(procnet_docs)} 篇文档 -> {output_file}")


def process_all_splits(
    input_dir: str,
    output_dir: str,
    event_type_mapping: Dict[str, str] = None,
    max_docs: int = None
):
    """
    处理所有数据分割（train, dev, test）
    
    Args:
        input_dir: 输入目录 (包含 train.json, test.json 等)
        output_dir: 输出目录
        event_type_mapping: intent 到事件类型的映射
        max_docs: 最大转换文档数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个子目录（flight_orders, hotel_orders, 等）
    subdirs = [d for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"发现 {len(subdirs)} 个子目录：{subdirs}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        
        print(f"\n处理子目录：{subdir}")
        
        # 处理每个分割文件
        for split_name in ['train', 'test']:
            input_file = os.path.join(subdir_path, f'{split_name}.json')
            output_file = os.path.join(output_subdir, f'{split_name}.json')
            
            if os.path.exists(input_file):
                print(f"  转换 {split_name}.json ...")
                convert_rasa_file_to_procnet(
                    input_file, output_file, 
                    event_type_mapping, max_docs
                )
            else:
                print(f"  跳过 {split_name}.json (文件不存在)")


def main():
    parser = argparse.ArgumentParser(
        description='将 data_v1b (RASA NLU 格式) 转换为 ProcNet 格式'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data_v1b',
        help='输入目录路径 (包含多个子目录，每个子目录有 train.json, test.json)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='procnet/Data_v1b',
        help='输出目录路径 (ProcNet 格式)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['all', 'train', 'test'],
        default='all',
        help='处理哪个数据分割'
    )
    
    parser.add_argument(
        '--max_docs',
        type=int,
        default=None,
        help='最大转换文档数（用于测试，默认全部转换）'
    )
    
    parser.add_argument(
        '--event_mapping',
        type=str,
        default=None,
        help='JSON 文件路径，包含 intent 到事件类型的映射'
    )
    
    args = parser.parse_args()
    
    # 加载事件类型映射（如果有）
    event_type_mapping = None
    if args.event_mapping:
        with open(args.event_mapping, 'r', encoding='utf-8') as f:
            event_type_mapping = json.load(f)
        print(f"加载事件类型映射：{len(event_type_mapping)} 个映射")
    
    # 处理数据
    if args.split == 'all':
        process_all_splits(
            args.input_dir, args.output_dir,
            event_type_mapping, args.max_docs
        )
    else:
        # 处理特定分割
        print(f"警告：--split={args.split} 尚未实现，使用 --split=all")
        process_all_splits(
            args.input_dir, args.output_dir,
            event_type_mapping, args.max_docs
        )


if __name__ == '__main__':
    main()
