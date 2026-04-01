# W2NER × ProcNet 数据链路分析

> 分析日期：2026-04-01
> 审查范围：两个仓库的完整数据流，从 data_v1b 到模型输入

---

## 一、完整数据链路

```
data_v1b/mixed_data_with_queries/train.json (RASA NLU 格式)
  │
  │ 格式: { "rasa_nlu_data": { "common_examples": [
  │   { "text": "...", "intent": "flight",
  │     "entities": [{"start":9,"end":12,"value":"陶端正","entity":"person"}]
  │ } ] } }
  │
  ▼ convert_data_v1b_to_procnet.py  [一次性离线转换]
  │
  │ 操作:
  │   1. split_into_sentences(text) — 按 [。！？!?；;]+ 分割
  │   2. map_entity_to_sentences() — 字符偏移 → (sent_idx, b, e)
  │   3. 构建复合 key: "文本#sent_b_e#类型"
  │   4. 构建事件字典: intent → event_type, entities → role→value
  │   5. 随机划分 train/dev/test (seed=42, 15%/15%)
  │
  ▼ procnet_format/mixed_data_with_queries/{train,dev,test}.json
  │
  │ 格式: [[doc_id, {
  │   "sentences": [...],
  │   "ann_valid_mspans": [...],          ← 冗余，Processor 不读取
  │   "ann_valid_dranges": [...],         ← 冗余，Processor 不读取
  │   "ann_mspan2dranges": {key: [[s,b,e]]},
  │   "ann_mspan2guess_field": {key: type},
  │   "recguid_eventname_eventdict_list": [[0, event_type, {role: value}]]
  │ }]]
  │
  ├─────────────────────────────────────────────────────┐
  │                                                     │
  ▼ (导出)                                              ▼
sidecar_entities_gold/                           W2NER 训练/预测
(JSONL, source="procnet_gold")                      │
  │                                                  │
  │ 32,727 实体 = procnet_format 的 JSONL 展平        │
  │ 用途: ProcNet gold label 上限实验                 │  export_doc_typed_entities.py
  │                                                  │
  │                                                  ▼
  │                                     sidecar_entities/
  │                                     (JSONL, source="w2ner")
  │                                     用途: W2NER→ProcNet 级联实验
  │
  ▼                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│              DocEEProcessor.parse_json_one()                 │
│                                                              │
│  输入: procnet_format/{split}.json                           │
│  Sidecar: sidecar_entities_gold/ 或 sidecar_entities/        │
│                                                              │
│  a) _unwrap_json_item() → doc_id + data dict                 │
│  b) _normalize_gold_entity_from_annotation():                │
│     - 解析复合 key "文本#sent_b_e#类型" (正则)                │
│     - 提取 span_text, sent_idx, b, e, field                  │
│     - 验证 span 与 sentences[sent_idx][b:e] 一致              │
│     - 返回 DocEEEntity(span, positions, field)               │
│  c) 从 recguid_eventname_eventdict_list 构建 events           │
│  d) _attach_typed_entities_from_sidecar():                   │
│     - 加载 sidecar JSONL (use_procnet_pred_entities=True)    │
│     - 按 doc_id 匹配 → 填充 doc.typed_entities               │
│                                                              │
│  输出: DocEEDocumentExample                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    DocEEPreparer                             │
│                                                              │
│  a) tokenize_sentences() — 逐字切分                          │
│  b) longer_sentence_process_simple_cut() — 截断 >max_len     │
│  c) seq_label_BIO_tags_generate() — 从 doc.entities 生成 BIO │
│  d) 构建词表: BIO tags, event types, event roles             │
│  e) _collect_procnet_type_ids() — 从 sidecar 收集类型        │
│  f) get_loader_for_flattened_fragment_before_event():        │
│     - 按 max_len 切分为 fragment                             │
│     - 计算 flat_b, flat_e (在 [CLS]+tokens 序列中的偏移)     │
│     - 构建 entity node dict (30+ 字段，模型只用 5 个)        │
│                                                              │
│  输出: (train/dev/test datasets + DataLoaders)               │
│  Batch: (doc_id, input_ids[], att_masks[], BIO_ids[],        │
│          events_labels[], procnet_entity_nodes[])            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              DocEEProxyNodeModel.forward()                   │
│                                                              │
│  a) BERT → lm_hidden_states                                 │
│  b) lm_bio_linear → BIO 预测 + loss                         │
│  c) 从 BIO 预测 → pred_position (span 边界)                  │
│  d) 如果使用 procnet_entity_nodes:                           │
│     - 对每个 node: mean(lm_hidden[flat_b:flat_e]) + type_emb │
│     - 添加 span 节点到 GCN 图 (S-M, S-C, S-S 边)             │
│  e) GCN 消息传递 (proxy + CLS + span 节点)                   │
│  f) Proxy → event_type, proxy → span → relation 预测         │
│  g) Loss: BIO + event_type + event_relation + event_num      │
│       + span-span-relation                                   │
│                                                              │
│  模型实际使用的 node 字段:                                    │
│    flat_b, flat_e, type_id, type_index, token_ids            │
│  模型不使用的字段 (25+):                                      │
│    text, span_tokens, w2ner_key, cluster_key, head, score... │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    DocEETrainer                              │
│                                                              │
│  a) _unpack_batch() — 解包 6-tuple batch                     │
│  b) model_fn() — 调用 model.forward()                        │
│  c) loss.backward() + optimizer.step()                       │
│  d) eval_batch_template() — 评估 + DocEEMetric               │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、W2NER 仓库数据链路

```
W2NER/data/{dataset}/{split}.json (W2NER 句子级格式)
  │
  │ 格式: [{sample_id, doc_id, sent_id, text, sentence:[chars],
  │         ner:[{index:[token_ids], type}], entities:[...]}]
  │
  ▼ data_loader.load_data_bert()
  │
  │ 操作:
  │   1. fill_vocab() — 收集实体类型
  │   2. process_bert():
  │      - BERT tokenizer → subword pieces
  │      - 构建 bert_token_ids = [CLS] + pieces + [SEP]
  │      - 构建 piece_map (word_idx → piece positions)
  │      - 构建 dist_inputs (相对距离矩阵 → 20 bins)
  │      - 构建 grid_labels (N×N 网格，chain links + type)
  │
  ▼ Model.forward()
  │
  │ 操作:
  │   1. BERT → bert_embs → max-pool → word_reps
  │   2. BiLSTM → contextualized word reps
  │   3. CLN (Conditional LayerNorm)
  │   4. 距离嵌入 + 区域嵌入
  │   5. 膨胀卷积
  │   6. Biaffine + FFNN → N×N×num_classes 网格输出
  │
  ▼ Trainer.predict() + decode_for_procnet()
  │
  │ 操作:
  │   1. 从 N×N 网格预测构建关系图
  │   2. decode_for_procnet_from_graph():
  │      - 遍历 THW + NNW 边
  │      - 提取实体 (token_indices, type_id, score, head)
  │   3. build_prediction_record():
  │      - 每句: {doc_id, sent_id, sentence, entity[], procnet_entities[]}
  │
  ▼ output.json (句子级预测)
  │
  │ 已找到的输出文件:
  │   - output.json: 180 条记录，180 篇文档，仅 id_card 领域 (8 种类型)
  │   - mixed_data_full_output.json: 2165 条记录，720 篇文档，28 种类型
  │     (5.2MB，包含多句文档，最多 7 句/篇)
  │
  ▼ export_doc_typed_entities.py
  │
  │ 操作:
  │   1. 按 (doc_id, sent_id) 对齐预测
  │   2. 句子级实体聚合为文档级 typed_entities
  │   3. 验证 span，去重
  │
  ▼ sidecar_entities/*.jsonl (W2NER 预测)
```

---

## 三、必要的组件

| 组件 | 原因 |
|------|------|
| `data_v1b/` | 唯一数据源 |
| `convert_data_v1b_to_procnet.py` | 句子分割 + 实体位置映射 + 事件构建，不可跳过 |
| `procnet_format/` | Processor 的输入格式，当前代码强依赖 |
| `sidecar_entities_gold/` | gold label 训练的上限实验 |
| `sidecar_entities/` | W2NER 预测 → ProcNet 级联 |
| `DocEEProcessor` | 解析 procnet_format + 加载 sidecar |
| `DocEEPreparer` | tokenize + BIO + fragment + entity nodes |
| `DocEEProxyNodeModel` | 模型 forward |
| `DocEETrainer` | 训练循环 |

---

## 四、不必要的组件（冗余）

| 冗余项 | 位置 | 说明 | 严重性 |
|--------|------|------|--------|
| `regenerate_full_pipeline.py` | scripts/ | **完全重复** `convert_data_v1b_to_procnet.py` 的转换逻辑，inline 复制粘贴 | 高 |
| `ann_valid_mspans` + `ann_valid_dranges` | procnet_format JSON | Processor 只读 `ann_mspan2dranges` 和 `ann_mspan2guess_field`，这两个平行数组**从未被读取** | 中 |
| 两个 `MyDataSet` 类 | `DocEE_preparer.py:366` 和 `:653` | **80% 代码重复**，应合并为一个类 | 高 |
| 3 个 W2NER 输出目录 | `regenerate_full_pipeline.py` | 生成 `data_w2ner_folded_with_dev`、`data_w2ner_folded`、`data_w2ner`，其中两个是折叠副本 | 中 |
| Entity node dict 中 25+ 个字段 | `build_procnet_entity_nodes_for_fragment` | 模型 forward 只用了 `flat_b`、`flat_e`、`type_id`、`type_index`、`token_ids`，其余从未被消费 | 中 |
| `DocEETypedEntity` 的 4 种 key | `key`、`cluster_key`、`w2ner_key`、`procnet_span_key` | 实际只需 2 个：去重用 + 溯源用 | 低 |
| `run_w2ner_sidecar_inference.py` + `.sh` | 已删除 | 参数错误，无法运行 | 已处理 |

---

## 五、Gold sidecar 来源

**gold sidecar = procnet_format 的另一种序列化格式。**

| 指标 | Gold sidecar | ProcNet format |
|------|-------------|----------------|
| 实体总数 | 32,727 | 32,727 |
| 文档数 | 3,360 | 3,360 |
| 来源 | `data_v1b` → `convert` → `procnet_format` → 导出为 JSONL | 同一来源 |
| `source` 字段 | `"procnet_gold"` | N/A |

**生成路径**：
```
data_v1b → convert_data_v1b_to_procnet.py → procnet_format → (导出) → sidecar_entities_gold/
```

gold sidecar 本质上就是 procnet_format 中的 `ann_mspan2dranges` + `ann_mspan2guess_field` 展平为 JSONL 格式，每个实体一行。

**用途**：
1. ProcNet 上限实验：用 gold 实体训练 ProcNet，测事件抽取的理论上限
2. 对比基线：与 W2NER 预测 sidecar 对比，量化 W2NER 预测误差对 ProcNet 的影响

---

## 六、W2NER 预测输出文件

| 文件 | 大小 | 记录数 | 文档数 | 实体类型 | 多句文档 |
|------|------|--------|--------|---------|---------|
| `output.json` | 1.2MB | 180 | 180 | 8 (仅 id_card) | 0 |
| `mixed_data_full_output.json` | 5.2MB | 2165 | 720 | 28 (全领域) | 526 (最多 7 句/篇) |

**`mixed_data_full_output.json` 是完整的 W2NER 预测输出**，覆盖了 mixed_data 的所有领域。

---

## 七、W2NER vs Gold 实体质量对比

| 指标 | W2NER 预测 | Gold |
|------|-----------|------|
| 平均每篇实体数 | 5.4 | 9.7 |
| 缺失类型 | `person`(0), `startDate`(0), `endDate`(0), `departureCity`(0), `arrivalCity`(7), `ticketGate`(0) 等 12 个类型 | 全覆盖 |
| 独有类型 | `date`(3128), `time`(520) | 无 |
| 类型名不一致 | `date` vs `startDate`/`endDate`，`time` vs `startTime`/`endTime` | 精确 |

**W2NER 把 `startDate`/`endDate` 合并成了泛化的 `date`，把 `startTime`/`endTime` 合并成了 `time`，完全丢失了 `person`、`departureCity`、`arrivalCity`、`ticketGate` 等关键类型。**

---

## 八、实验架构建议

```
实验 A（上限）: data_v1b → gold sidecar → ProcNet
实验 B（级联）: data_v1b → W2NER 训练 → W2NER sidecar → ProcNet
```

两者对比才能说明 W2NER→ProcNet 级联的有效性。

---

## 九、核心结论

1. **数据链路本身是通的**，但中间层过度工程化
2. **真正必要的数据流**只有：`data_v1b → procnet_format → Processor → Preparer → Model` + sidecar
3. **位置索引耦合已成立**：W2NER 的 `b/e` 与 ProcNet 的 `[sent_idx, start, end]` 口径一致
4. **event relation learning 有条件成立**：在 character-level tokenize + 同一 tokenizer 的前提下，token_ids 匹配成立
5. **主要风险**：W2NER 预测质量不足（recall ~55%，类型映射混乱），而非数据格式问题
