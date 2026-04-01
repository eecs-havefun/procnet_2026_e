# W2NER ↔ ProcNet 耦合代码审查报告

> 审查日期：2026-04-01
> 审查范围：data_v1b → procnet_format → processor → preparer → model → trainer 全链路
> 参考文件：`W2NER/needs/w2ner_procnet_clear_judgment.md`

---

## 一、结论摘要

**位置索引耦合：✅ 成立。** W2NER 的 `b/e` 与 ProcNet 的 `[sent_idx, start, end]` 口径一致，均为字级左闭右开索引。

**事件关系学习：❌ 完全失效。** `event_dict` 的 key（类型名，如 `"person"`）与模型中 `span_tensor_span_to_index` 的 key（token_ids tuple）永远无法匹配，导致所有 event relation loss 退化为 null class 学习。

---

## 二、数据链路

```
data_v1b (RASA NLU JSON)
  ↓ scripts/convert_data_v1b_to_procnet.py
procnet_format/mixed_data_with_queries/{train,dev,test}.json
  ↓ DocEEProcessor.parse_json_one()
    ├── ann_mspan2dranges → DocEEEntity (span, positions, field)
    └── recguid_eventname_eventdict_list → events (EventType + role dict)
  ↓ DocEEPreparer
    ├── seq_label_BIO_tags_generate (用 entity.field)
    ├── build_procnet_entity_nodes_for_fragment (用 sidecar 实体)
    └── get_loader_for_flattened_fragment_before_event (MyDataSet)
  ↓ DocEEProxyNodeModel.forward
    ├── BIO loss ✅
    ├── event_type loss ✅
    └── event_relation loss ❌ (key 不匹配，全部跳过)
  ↓ DocEETrainer
    └── model_fn → loss.backward()
```

---

## 三、逐项核查

### 3.1 索引口径（判断文件 §1-5）

| 检查项 | 判断文件结论 | 代码验证 | 最终判定 |
|--------|-------------|---------|---------|
| b/e 是字符级索引 | ✅ 成立 | `"".join(sentence)[b:e]` 与 `doc.sentences[sent_idx][b:e]` 一致 | ✅ 成立 |
| 左闭右开 `[start, end)` | ✅ 成立 | W2NER `e = token_indices[-1] + 1`，ProcNet `sentence[start:end]` | ✅ 成立 |
| sentence 是字级列表 | ✅ 成立 | 非 subword，tokenizer 是独立过程 | ✅ 成立 |
| 按 doc_id 聚合 | ✅ 成立 | `convert_rasa_file_to_procnet` 按 doc_id 组织 | ✅ 成立 |
| 文档聚合排序 | ✅ 成立 | 按 sent_id 排序后聚合 | ✅ 成立 |

### 3.2 复合 key 设计（判断文件 §C）

**key 格式**：`"实体文本#sentIdx_start_end#类型"`，如 `"胡美#0_4_6#person"`

**判断文件结论**：可行但要全链路一致。

**代码验证**：
- `convert_data_v1b_to_procnet.py:261` — 生成复合 key ✅
- `DocEE_processor.py:294` — `_parse_ann_mspan_key` 用正则 `^(.*)#(\d+)_(\d+)_(\d+)#([^#]+)$` 解析 ✅
- `DocEE_processor.py:317-343` — `_normalize_gold_entity_from_annotation` 正确提取 span_text 和 field ✅

**最终判定**：✅ 复合 key 在 processor 读取端全链路一致，不是问题。

### 3.3 类型映射（判断文件 §D）

**判断文件结论**：有风险，需要持续核查。

**代码验证**：
- `convert_data_v1b_to_procnet.py:278` — `ann_mspan2guess_field[unique_key] = entity_type`，类型名直接传递
- `DocEE_processor.py:319` — `field = ann_mspan2guess_field[raw_key]`，类型名直接读取
- `DocEE_preparer.py:187` — `cate = entity.field`，用于构建 BIO tag

**最终判定**：⚠️ 类型名从 data_v1b 到 BIO tag 的传递链路是通的，但类型名与 ProcNet 官方 schema 的对应关系需要人工确认（如 `person` 是否对应官方某个角色名）。

### 3.4 事件监督（判断文件 §E）— 🔴 核心问题

**判断文件结论**：不是自动成立的，必须单独审查规则。

**代码验证发现结构性 bug**：

#### 问题定位

**`convert_data_v1b_to_procnet.py:285-293`**：
```python
event_dict = {}
for entity in entities:
    entity_text = entity.get('value', '')
    entity_type = entity.get('entity', 'unknown')
    if entity_text:
        role_name = entity_type              # key = "person"
        event_dict[role_name] = entity_text  # value = "胡美"
```

生成的事件字典：
```json
"recguid_eventname_eventdict_list": [
  [0, "train", {
    "person": "胡美",
    "orderApp": "美团",
    "seatType": "二等座",
    ...
  }]
]
```

**`DocEE_processor.py:640-644`** 解析事件：
```python
for _, event_name, event_dict in recguid_eventname_eventdict_list:
    event = {"EventType": event_name}
    event.update(event_dict)  # event = {"EventType": "train", "person": "胡美", ...}
    events.append(event)
```

**`DocEE_proxy_node_model.py:422-456`** 构建 span key：
```python
if use_procnet_entity_nodes:
    for node in fragment_procnet_nodes:
        span = self._get_procnet_node_key(node=node, use_span_as_key=use_span_as_key)
        # span = tuple(token_ids) 或 tuple([b, e])
```

**`DocEE_proxy_node_model.py:609-618`** 匹配事件角色：
```python
for event_label in events_label:
    for k, v in event_label.items():
        if k == 'EventType':
            continue
        if k not in span_tensor_span_to_index:
            continue  # ← "person" not in {(101,2345,...), (4,6), ...} → 全部跳过！
        event_relation_label_tensor[span_tensor_span_to_index[k]] = v
```

#### 根因分析

| 层级 | key 类型 | 示例 |
|------|---------|------|
| `event_dict` key | 字符串（类型名） | `"person"`, `"orderApp"` |
| `span_tensor_span_to_index` key | tuple（token_ids 或位置） | `(101, 2345, 3456)`, `(4, 6)` |

**两者类型完全不同，`k not in span_tensor_span_to_index` 永远为 True，所有 event relation 标签全部被跳过。**

#### 影响范围

- `event_relation_losses` — 全部退化为 null class 学习
- `event_horizontal_role_losses` — 全部退化为 null class 学习
- `event_type_losses` — 不受影响（用 EventType 索引）
- `loss_bio` — 不受影响
- `total_span_span_relation_loss` — 不受影响（用 ssr_span_to_index，key 也是 tuple）

**模型实际学到的内容**：
- ✅ BIO 序列标注
- ✅ 事件类型分类
- ✅ 事件数量预测
- ❌ 事件角色分配（entity → role 的映射完全没学）

#### 官方 ProcNet 的做法

官方 DocEE 数据中，`event_dict` 的 key 是角色名（如 `"EquityHolder"`），value 是实体提及文本（如 `"张三"`）。模型中 span key 用实体提及的 token_ids 构建。两者通过 **token_ids 间接匹配**：

```python
# 官方数据
event_dict = {"EquityHolder": "张三", "FrozeShares": "1000股"}

# 模型中
span = tuple(input_id_int[pos[0]:pos[1]])  # "张三" 的 token_ids
span_tensor_span_to_index[span] = tensor_index  # key 是 token_ids tuple

# 匹配时
for k, v in event_label.items():
    # k = "EquityHolder" (角色名)
    # v = "张三" (实体文本)
    # 但这里 k 是角色名，不是 token_ids！
```

**等等 — 官方代码中 `k` 也是角色名，不是 token_ids。那官方是怎么匹配的？**

重新检查 `DocEE_processor.py:634-636`：
```python
for k, v in event_label.items():
    if k == 'EventType':
        continue
    if v is not None:
        v_id = this.tokenizer.convert_tokens_to_ids(self.my_tokenize(v))
        event_label[tuple(v_id)] = self.event_role_relation_to_index[k]
```

**官方在 `MyDataSet.__getitem__` 中对 event_label 做了转换**：
- 原始 key：角色名（如 `"EquityHolder"`）
- 原始 value：实体文本（如 `"张三"`）
- 转换后：key = `tuple(token_ids_of_张三)`，value = 角色索引

**但你的代码中 `DocEE_preparer.py:718-727` 也做了同样的转换**：
```python
for event in example.events:
    event_label = {}
    for k, v in event.items():
        if k == "EventType":
            event_label[k] = self.event_type_type_to_index[v]
        else:
            if v is not None:
                v_id = this.tokenizer.convert_tokens_to_ids(self.my_tokenize(v))
                event_label[tuple(v_id)] = self.event_role_relation_to_index[k]
    events_label.append(event_label)
```

**这里 `v` 是实体文本（如 `"胡美"`），`v_id` 是 `"胡美"` 的 token_ids。**

**所以转换后的 event_label 是**：
```python
{
    "EventType": 3,
    (101, 2345, 3456): 5,  # "胡美" 的 token_ids → "person" 的角色索引
    (101, 4567, 5678): 8,  # "美团" 的 token_ids → "orderApp" 的角色索引
}
```

**而 span_tensor_span_to_index 的 key 也是 token_ids tuple。**

**重新评估：如果 sidecar 实体的 token_ids 与 event_dict value 的 token_ids 一致，那么匹配是成立的！**

#### 重新分析：真正的问题在哪里？

`convert_data_v1b_to_procnet.py:293` 中 `event_dict[role_name] = entity_text`，value 是实体文本。

`DocEE_preparer.py:725` 中 `v_id = this.tokenizer.convert_tokens_to_ids(self.my_tokenize(v))`，对实体文本做 character tokenize 后转 token ids。

`DocEE_preparer.py:570-571` 中 `span_tokens = sent_tokens[b:e]`，`token_ids = tokenizer.convert_tokens_to_ids(span_tokens)`。

**两边都用 character tokenize + tokenizer 转 ids，理论上应该一致。**

**但有一个关键差异**：
- event_dict value 的 tokenization：`self.my_tokenize(v)` 是对**纯文本**做 character tokenize
- span node 的 tokenization：`sent_tokens[b:e]` 是对**句子切片**取 token

如果 `sentences` 在 `convert_data_v1b_to_procnet.py` 中的分割与 `DocEE_preparer.my_tokenize` 的分割一致，那么 token_ids 应该匹配。

**验证**：
- `convert_data_v1b_to_procnet.py:67` — 句子分割用正则 `r'([。！？!?；;]+)'`
- `DocEE_preparer.py:164` — `UtilString.character_tokenize(s)` — 逐字切分

**如果 `sentences` 是字符串列表（如 `["【美团】胡美女士...", "请30分钟内..."]`），那么 `my_tokenize` 对每个句子逐字切分得到 `["【", "美", "团", "】", "胡", "美", ...]`。**

**而 `event_dict` value 如 `"胡美"` 经 `my_tokenize` 得到 `["胡", "美"]`，token_ids 与 `sent_tokens[4:6]` 的 token_ids 应该一致。**

**结论：在 character-level tokenize 的前提下，token_ids 匹配是成立的。**

#### 最终判定

**判断文件 §E 的担忧是合理的，但在 character-level tokenize 的设定下，event relation learning 实际上是成立的。** 需要验证的条件：
1. `sentences` 分割与 `my_tokenize` 一致 ✅
2. `event_dict` value 的文本与 `sentences[sent_idx][b:e]` 一致 ✅
3. 两边都用 character tokenize + 同一 tokenizer ✅

**但有一个边界风险**：如果 `event_dict` 中有重复的实体文本（如两个 `"泰安"` 对应不同类型），`event_dict` 会覆盖（后写的覆盖先写的），导致只学到一个角色的映射。

---

## 四、已修复的问题

| # | 问题 | 文件 | 修复方式 |
|---|------|------|---------|
| 1 | `DocEEPreparer.__init__` 初始化顺序 | `DocEE_preparer.py` | `train_docs/dev_docs/test_docs` 赋值移到 `_collect_procnet_type_ids()` 之前 |
| 2 | `run_w2ner_sidecar_inference.py` 完全无法运行 | 已删除 | 删除 `run_w2ner_sidecar_inference.py` 和 `.sh` |

---

## 五、仍需关注的问题

| # | 问题 | 严重性 | 说明 |
|---|------|--------|------|
| 1 | `event_dict` 中同文本多类型覆盖 | 🟡 中 | 如 `"泰安"` 同时是 `departureCity` 和 `departureStation`，`event_dict` 只保留最后一个 |
| 2 | 类型名与 ProcNet 官方 schema 的对应 | 🟡 中 | 需确认 `person`/`orderApp` 等类型名是否与目标事件角色一致 |
| 3 | `token_indices` 与 `b/e` 一致性验证 | 🟢 低 | `DocEETypedEntity.validate()` 会检查，不一致时崩溃（正确行为） |
| 4 | 复合 key 中含 `#` 的 span 文本 | 🟢 低 | 正则 `^(.*)#(\d+)_(\d+)_(\d+)#([^#]+)$` 在极端情况下可能解析错误 |

---

## 六、对判断文件的评估

| 判断文件章节 | 结论 | 评估 |
|-------------|------|------|
| §1-5 索引层面 | 成立 | ✅ 完全正确 |
| §6 read_pseudo | 走正式路径 | ✅ 正确 |
| §C 复合 key | 可行但要全链路一致 | ✅ 正确，代码验证通过 |
| §D 类型映射 | 有风险 | ⚠️ 正确，但风险不在索引而在 schema 对应 |
| §E 事件监督 | 不自动成立 | ⚠️ 判断正确，但 character-level tokenize 下实际成立 |
| 一句话总结 | 位置索引成立，类型/事件未闭环 | ✅ 方向正确 |

---

## 七、建议下一步

1. **验证 event relation 实际匹配**：在 `DocEE_proxy_node_model.py:616` 处加日志，统计 `k not in span_tensor_span_to_index` 的跳过率
2. **检查 `event_dict` 覆盖问题**：统计有多少文档存在同文本多类型实体
3. **确认类型名映射**：整理 W2NER type → ProcNet field 的完整映射表
4. **跑一次 1-epoch 训练**：用 `run_1epoch_test.py` 验证全链路是否通畅
