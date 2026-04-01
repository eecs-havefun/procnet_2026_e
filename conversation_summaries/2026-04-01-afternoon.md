# W2NER × ProcNet 对话摘要 — 2026-04-01 (下午)

> 会话时间：2026-04-01 下午
> 核心目标：修复 W2NER 训练数据折叠问题，重新训练并导出 ProcNet 可用格式

---

## 一、本次会话完成的工作

### 1. 删除无效脚本
- 删除 `run_w2ner_sidecar_inference.py` 和 `run_w2ner_sidecar_inference.sh`（参数错误，无法运行）

### 2. 数据链路完整分析
- 追踪了 `data_v1b → procnet_format → W2NER → sidecar → ProcNet` 全链路
- 确认位置索引耦合成立（字级、左闭右开、复合 key 区分同文本多类型）
- 确认 event relation learning 在 character-level tokenize 下有条件成立

### 3. 发现并修复 `date`/`time` 折叠问题
- **根因**：`convert_procnet_to_w2ner.py` 中的 `ROLE_FOLD_MAP` 将 `startDate`/`endDate` 合并为 `date`，`startTime`/`endTime` 合并为 `time`
- **影响**：W2NER 训练数据丢失了 4 个细粒度类型，只剩 28 个
- **修复**：重新运行转换脚本，不启用 `--fold_role_types`，原地覆盖 W2NER 训练数据
- **结果**：30 个类型全部恢复（`startDate`: 2141, `endDate`: 1518, `startTime`: 424, `endTime`: 368）

### 4. W2NER 10 epoch 训练
- **配置**：batch_size=12, GPU 1, 10 epochs
- **结果**：
  | 指标 | 数值 |
  |------|------|
  | Entity F1 (Test) | **0.9552** |
  | Label F1 (Test) | **0.9678** |
  | Precision | 0.9588 |
  | Recall | 0.9515 |
  | Best Dev F1 | 0.9530 |
- **结论**：不折叠策略成功，30 个细粒度类型全部被有效学习

### 5. 导出尝试（未完成）
- 尝试用 `export_doc_typed_entities.py` 导出 W2NER 预测为 ProcNet sidecar 格式
- **问题**：W2NER 的 `predict_final` 只跑了 test 集（720 篇），而 ProcNet 需要全部 3360 篇
- **状态**：待解决

---

## 二、关键发现

### 数据层面
| 发现 | 状态 |
|------|------|
| 位置索引耦合成立 | ✅ 确认 |
| 复合 key 设计有效 | ✅ 确认 |
| `date`/`time` 折叠导致类型丢失 | ✅ 已修复 |
| W2NER 预测覆盖不足（仅 test 集） | ⚠️ 待解决 |
| 同文本多类型（912/3360 篇） | ✅ 复合 key 天然区分 |

### 模型层面
| 发现 | 状态 |
|------|------|
| 30 类型 Entity F1 = 0.9552 | ✅ 优秀 |
| Precision/Recall 均衡（差 0.7%） | ✅ 良好 |
| 无明显过拟合（Dev/Test 一致） | ✅ 良好 |

---

## 三、待解决问题

### 1. W2NER 全量预测导出
- **问题**：W2NER 的 `main.py` 中 `predict_final` 只跑 test_loader
- **需要**：修改预测逻辑，对 train/dev/test 全部数据生成预测
- **影响**：ProcNet 训练需要全部 3360 篇的 sidecar 实体

### 2. W2NER 预测质量提升
- `person` 类型 recall 偏低（训练数据中有，但预测时可能漏检）
- `arrivalCity`、`cardAddress`、`idNumber` 等低频类型 F1 约 0.80-0.85

### 3. ProcNet 级联实验
- 待 W2NER 全量预测导出后，用 sidecar 方式连接 W2NER → ProcNet
- 对比 gold sidecar（上限）vs W2NER sidecar（级联）的实验结果

---

## 四、文件变更

| 文件 | 变更 |
|------|------|
| `W2NER/data/mixed_data_with_queries/{train,dev,test}.json` | 重新生成，不折叠 `date`/`time` |
| `W2NER/config/mixed_data_with_queries.json` | epochs: 1→10, dataset 路径修正 |
| `W2NER/mixed_data_with_queries_unfolded_model.pt` | 新训练模型 |
| `W2NER/mixed_data_with_queries_unfolded_output.json` | test 集预测结果 |
| `procnet/run_w2ner_sidecar_inference.py` | 已删除 |
| `procnet/run_w2ner_sidecar_inference.sh` | 已删除 |
| `procnet/conversation_summaries/data_pipeline_analysis_2026-04-01.md` | 新增 |
| `procnet/conversation_summaries/w2ner_procnet_needed_schema_list.md` | 已填充 |
| `procnet/conversation_summaries/w2ner_procnet_current_conclusion.md` | 已查看 |
| `procnet/conversation_summaries/code_review_2026-04-01.md` | 新增 |

---

## 五、一句话总结

**不折叠 `date`/`time` 的 W2NER 训练达到 Entity F1=0.9552，证明 30 个细粒度类型可被有效学习；下一步需解决 W2NER 全量预测导出问题，才能完成 W2NER→ProcNet 级联实验。**
