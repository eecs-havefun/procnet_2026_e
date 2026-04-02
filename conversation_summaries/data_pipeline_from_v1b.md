# W2NER × ProcNet 数据链路 — 从 data_v1b 开始

> 整理日期：2026-04-02
> 数据来源：`conversation_summaries/` 下 9 份对话摘要

---

## 一、总览图

```
data_v1b/ (RASA NLU 原始格式)
  │
  ▼ convert_data_v1b_to_procnet.py  [一次性离线转换]
  │   • 按 [。！？!?；;]+ 分割句子
  │   • 字符偏移 → (sent_idx, b, e) 映射
  │   • 构建复合 key: "文本#sent_b_e#类型"
  │   • 构建事件字典: intent → event_type, entities → role→value
  │   • 随机划分 train/dev/test (seed=42, 15%/15%)
  │
  ▼ procnet_format/mixed_data_with_queries/{train,dev,test}.json
  │   格式: [[doc_id, {sentences, ann_mspan2dranges, ann_mspan2guess_field,
  │          recguid_eventname_eventdict_list}]]
  │
  ├──────────────────────────────────────────────────────┐
  │                                                      │
  ▼ (导出 JSONL)                                         ▼ (W2NER 训练/预测)
sidecar_entities_gold/                            W2NER 数据转换
(JSONL, source="procnet_gold")                         │
32,727 实体, 3,360 文档                                 ▼
用途: ProcNet gold label 上限实验           W2NER/data/{dataset}/{split}.json
                                              │
                                              ▼ data_loader.load_data_bert()
                                              │  BERT tokenizer, piece_map,
                                              │  dist_inputs, grid_labels
                                              ▼
                                        Model.forward()
                                        BERT → BiLSTM → 2D conv → Biaffine
                                              │
                                              ▼ Trainer.predict() + decode_for_procnet()
                                              │
                                              ▼ output.json (句子级预测)
                                              │  mixed_data_full_output.json: 2165条, 720篇
                                              │
                                              ▼ export_doc_typed_entities.py
                                              │  按 (doc_id, sent_id) 聚合
                                              ▼
                                        sidecar_entities/*.jsonl
                                        (JSONL, source="w2ner")
                                        用途: W2NER→ProcNet 级联实验
  │                                                      │
  ▼                                                      ▼
┌────────────────────────────────────────────────────────────────┐
│                    DocEEProcessor.parse_json_one()              │
│                                                                 │
│  a) _unwrap_json_item() → doc_id + data dict                   │
│  b) _normalize_gold_entity_from_annotation():                   │
│     - 解析复合 key "文本#sent_b_e#类型" (正则)                   │
│     - 提取 span_text, sent_idx, b, e, field                     │
│     - 验证 span 与 sentences[sent_idx][b:e] 一致                 │
│  c) 从 recguid_eventname_eventdict_list 构建 events              │
│  d) _attach_typed_entities_from_sidecar():                      │
│     - 加载 sidecar JSONL (gold 或 W2NER 预测)                   │
│     - 按 doc_id 匹配 → 填充 doc.typed_entities                  │
│                                                                 │
│  输出: DocEEDocumentExample                                     │
└───────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                        DocEEPreparer                            │
│                                                                 │
│  a) tokenize_sentences() — 逐字切分                              │
│  b) longer_sentence_process_simple_cut() — 截断 >max_len         │
│  c) seq_label_BIO_tags_generate() — 从 doc.entities 生成 BIO     │
│  d) 构建词表: BIO tags, event types, event roles                 │
│  e) _collect_procnet_type_ids() — 从 sidecar 收集类型            │
│  f) get_loader_for_flattened_fragment_before_event():            │
│     - 按 max_len 切分为 fragment                                 │
│     - 计算 flat_b, flat_e (在 [CLS]+tokens 序列中的偏移)         │
│     - 构建 entity node dict (30+ 字段，模型只用 5 个)            │
│                                                                 │
│  输出: (train/dev/test datasets + DataLoaders)                   │
│  Batch: (doc_id, input_ids[], att_masks[], BIO_ids[],            │
│          events_labels[], procnet_entity_nodes[])                │
└───────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                  DocEEProxyNodeModel.forward()                  │
│                                                                 │
│  a) BERT → lm_hidden_states                                    │
│  b) lm_bio_linear → BIO 预测 + loss                             │
│  c) 从 BIO 预测 → pred_position (span 边界)                      │
│  d) 如果使用 procnet_entity_nodes:                              │
│     - 对每个 node: mean(lm_hidden[flat_b:flat_e]) + type_emb    │
│     - 添加 span 节点到 GCN 图 (S-M, S-C, S-S 边)                 │
│  e) GCN 消息传递 (proxy + CLS + span 节点)                       │
│  f) Proxy → event_type, proxy → span → relation 预测             │
│  g) Loss: BIO + event_type + event_relation + event_num          │
│       + span-span-relation                                       │
│                                                                 │
│  模型实际使用的 node 字段: flat_b, flat_e, type_id,              │
│    type_index, token_ids                                        │
│  模型不使用的字段 (25+): text, span_tokens, w2ner_key,           │
│    cluster_key, head, score...                                  │
└───────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                        DocEETrainer                             │
│                                                                 │
│  a) _unpack_batch() — 解包 6-tuple batch                         │
│  b) model_fn() — 调用 model.forward()                            │
│  c) loss.backward() + optimizer.step()                           │
│  d) eval_batch_template() — 评估 + DocEEMetric                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 二、关键转换步骤详解

### Step 1: `data_v1b` → `procnet_format`

| 项目 | 详情 |
|------|------|
| **脚本** | `scripts/convert_data_v1b_to_procnet.py` |
| **输入** | RASA NLU JSON: `{rasa_nlu_data: {common_examples: [{text, intent, entities: [{start, end, value, entity}]}]}}` |
| **核心操作** | 1. 句子分割（正则 `[。！？!?；;]+`）<br>2. 字符偏移→句子内偏移映射<br>3. 复合 key 构建<br>4. 事件字典构建（`role_name = entity_type` 直接透传）<br>5. train/dev/test 随机划分 |
| **输出** | `procnet_format/mixed_data_with_queries/{train,dev,test}.json` |

### Step 2: `procnet_format` → Sidecar（两条路径）

| 路径 | 说明 |
|------|------|
| **Gold 路径** | `procnet_format` → 导出为 JSONL → `sidecar_entities_gold/`（32,727 实体，3,360 文档） |
| **W2NER 路径** | `procnet_format` → 转换为 W2NER 格式 → W2NER 训练 → W2NER 预测 → `export_doc_typed_entities.py` → `sidecar_entities/` |

### Step 3: W2NER 内部数据链路

```
W2NER/data/{dataset}/{split}.json
  ↓ data_loader.load_data_bert()
  │  • BERT tokenizer → subword pieces
  │  • 构建 bert_token_ids = [CLS] + pieces + [SEP]
  │  • 构建 piece_map, dist_inputs, grid_labels
  ↓ Model.forward()
  │  • BERT → BiLSTM → 2D conv → Biaffine → N×N×num_classes
  ↓ Trainer.predict() + decode_for_procnet()
  │  • 从 N×N 网格预测构建关系图
  │  • decode_for_procnet_from_graph() → 提取实体
  │  • build_prediction_record() → {doc_id, sent_id, sentence, entity[], procnet_entities[]}
  ↓ output.json (句子级预测)
  ↓ export_doc_typed_entities.py (按 doc_id 聚合)
  ↓ sidecar_entities/*.jsonl
```

---

## 三、已发现并修复的问题

| # | 问题 | 状态 | 影响 |
|---|------|------|------|
| 1 | `date`/`time` 类型折叠（`startDate`/`endDate`→`date`，`startTime`/`endTime`→`time`） | ✅ 已修复 | 重新转换，30 个类型全部恢复 |
| 2 | `DocEEPreparer.__init__` 初始化顺序（`_collect_procnet_type_ids()` 在 `train_docs` 赋值前调用） | ✅ 已修复 | 调整赋值顺序 |
| 3 | `run_w2ner_sidecar_inference.py` 无法运行 | ✅ 已删除 | 删除无效脚本 |
| 4 | W2NER 预测仅覆盖 test 集（720 篇），ProcNet 需要全部 3,360 篇 | ⚠️ 待解决 | 需修改预测逻辑 |

---

## 四、核心结论

1. **位置索引耦合已成立**：W2NER 的 `b/e` 与 ProcNet 的 `[sent_idx, start, end]` 口径一致（字级、左闭右开）
2. **复合 key 设计有效**：`"文本#sentIdx_start_end#类型"` 天然区分同文本多类型
3. **Event relation learning 有条件成立**：在 character-level tokenize + 同一 tokenizer 前提下，token_ids 匹配成立
4. **主要剩余风险**：
   - W2NER 对 `person`、`arrivalCity` 等关键类型预测覆盖不足
   - W2NER 全量预测导出尚未完成（仅 720/3360 篇）
