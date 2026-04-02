# 2026-04-02 W2NER→ProcNet 全量级联实验

## 目标
将 W2NER 预测的 `procnet_entities` 转换为 ProcNet 可消费的 sidecar 格式，完成全量（train/dev/test 共 4800 篇）级联训练。

## 数据链路
```
data_v1b (4800条 RASA NLU)
  → convert_data_v1b_to_procnet.py (70/15/15 随机划分)
  → procnet_format/mixed_data_with_queries/ (train=3360, dev=720, test=720)
  → W2NER 训练 + 预测
  → W2NER/predictions/{train,dev,test}.json
  → convert_w2ner_to_procnet_sidecar.py
  → sidecar_entities/{train,dev,test}_doc_typed_entities.jsonl
  → ProcNet 训练（挂载 sidecar entity nodes）
```

## 完成的工作

### 1. W2NER 全量预测
- **修改 `W2NER/main.py`**：训练结束后依次对 train/dev/test 做预测
- **新建 `W2NER/predict_all_splits.py`**：独立预测脚本，加载已有模型对三个 split 做预测
- **运行结果**（10 epoch 模型，batch_size=8）：

| Split | Sentences | Entity F1 | Precision | Recall | 预测实体数 |
|-------|-----------|-----------|-----------|--------|-----------|
| train | 10,261 | 0.9594 | 0.9626 | 0.9563 | 32,512 |
| dev | 2,290 | 0.9530 | 0.9550 | 0.9511 | 7,083 |
| test | 2,165 | 0.9552 | 0.9588 | 0.9515 | 6,942 |

- **输出文件**：`W2NER/predictions/{train,dev,test}.json`

### 2. W2NER→ProcNet 转换脚本
- **新建 `convert_w2ner_to_procnet_sidecar.py`**
- 类型归一化：30 个 W2NER 小写类型 → ProcNet 驼峰类型（如 `orderapp` → `orderApp`）
- 按 `doc_id` 聚合 + `sent_id` 排序
- 复合 key：`文本#sentIdx_b_e#类型`
- span/text 校验（严格模式）
- **转换结果**：

| Split | 文档数 | 实体数 | 类型数 | 验证错误 |
|-------|--------|--------|--------|----------|
| train | 3,360 | 32,512 | 30 | 0 |
| dev | 720 | 7,083 | 30 | 0 |
| test | 720 | 6,942 | 30 | 0 |

- **输出文件**：`sidecar_entities/{train,dev,test}_doc_typed_entities.jsonl`

### 3. ProcNet 训练脚本修复
- **修复 `run_1epoch_test.py`**：
  - 路径解析：`Path(__file__).resolve().parent` 替代 `Path(__file__).parent`
  - 使用正确的 API：`get_loader_for_flattened_fragment_before_event()`
  - 正确的 Trainer 初始化参数
  - 日志输出到文件 + 控制台

- **修复 `DocEE_proxy_node_trainer.py`**：
  - 启用 checkpoint 保存：`self.model_save_folder_path = self.checkpoint_folder_init(...)`
  - 禁用 tqdm 进度条（`disable=True`），确保日志干净可读

### 4. Smoke Test 验证
- 3360/3360 篇 train 文档成功加载 sidecar
- 首个 sidecar 实体：type=orderApp, text=美团, b=1, e=3
- 模型参数量：109,219,428
- **1 epoch 结果**（proxy_slot_num=16, grad_accum=8）：

| 指标 | Dev | Test |
|------|-----|------|
| Event F1 | 0.4753 | 0.4954 |
| id_card F1 | 0.9881 | 0.9886 |
| train F1 | 0.5649 | 0.5609 |
| flight F1 | 0 | 0 |
| hotel F1 | 0 | 0 |
| BIO valid_span F1 | 0.9040 | 0.9094 |

## 训练配置
| 参数 | 值 | 含义 |
|------|-----|------|
| proxy_slot_num | 8 | 事件代理节点数量 |
| node_size | 512 | 节点向量维度 |
| gradient_accumulation_steps | 32 | 等效 batch_size |
| max_epochs | 10 | 训练轮数 |
| device | cuda:3 (CUDA_VISIBLE_DEVICES=3) | GPU |
| model_save_name | w2ner_sidecar_exp1 | 实验名称 |
| use_procnet_pred_entities | True | 使用 W2NER 预测 sidecar |
| return_procnet_entity_nodes | True | 打包 sidecar 到 batch |
| use_procnet_entity_nodes | True | 模型使用 sidecar 节点 |

## 输出位置
| 类型 | 路径 |
|------|------|
| 训练日志 | `training_w2ner_sidecar_exp1.log` |
| 每 epoch 结果 JSON | `procnet/Result/w2ner_sidecar_exp1/w2ner_sidecar_exp1_XXX.json` |
| 模型 checkpoint | `procnet/Checkpoint/w2ner_sidecar_exp1/w2ner_sidecar_exp1_XXX.pth` |

## 验证 sidecar 是否生效
训练日志中出现以下标记即表示 sidecar 已被模型使用：
- `[PROCNET_DEBUG][trainer_in] ... has_procnet_entity_nodes=True`
- `[PROCNET_DEBUG][trainer_in] ... fragment_node_counts=[...]`
- `[PROCNET_HIT] ... used_procnet_entity_nodes=True`

## 待办
- [ ] 运行 10 epoch 全量训练
- [ ] 分析各 epoch 的 Event F1 变化趋势
- [ ] 对比 gold sidecar vs W2NER predicted sidecar 的性能差异
- [ ] 解决 flight/hotel 事件 F1 为 0 的问题
