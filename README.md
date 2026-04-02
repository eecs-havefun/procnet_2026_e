# ProcNet × W2NER — 文档级多事件抽取与实体识别耦合框架

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

本项目是 [ProcNet](https://github.com/xnyuwg/procnet)（ACL 2023）的扩展分支，将其与 [W2NER](https://github.com/ljynlp/W2NER) 耦合，实现从**命名实体识别**到**文档级多事件抽取**的统一流水线。

> ProcNet 使用事件代理节点（Event Proxy Nodes）和 Hausdorff 距离最小化，实现文档级多事件抽取。通过代理节点建立事件间的全局依赖关系，直接最小化预测事件集合与真实事件集合之间的 Hausdorff 距离。
>
> 本扩展引入 W2NER 预测的实体作为 sidecar 输入，替代 gold 实体标注，实现 W2NER → ProcNet 级联事件抽取。
>
> 未来计划引入 EPAL 的角色索引槽位填充机制，解决跨事件实体复用和同事件多角色冲突问题。

---

## 目录

- [架构概览](#架构概览)
- [数据链路](#数据链路)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [训练命令](#训练命令)
- [W2NER 耦合](#w2ner-耦合)
- [EPAL 集成规划](#epal-集成规划)
- [脚本参考](#脚本参考)
- [项目结构](#项目结构)
- [已知问题](#已知问题)
- [引用](#引用)

---

## 架构概览

### 模型架构

<p align="center">
  <img src="./figures/architecture.jpg" width="600"/>
</p>

**ProcNet 架构**：BERT → BIO 序列标注 → 事件代理节点 → GCN 消息传递 → 事件类型/角色预测 → Hausdorff 距离损失

**耦合架构**：W2NER 实体预测 → sidecar JSONL → ProcNet entity nodes → 事件抽取

---

## 数据链路

```
data_v1b/ (RASA NLU 原始格式, 4,800 篇文档)
  │
  ▼ scripts/convert_data_v1b_to_procnet.py
  │   句子分割 → 字符偏移映射 → 复合 key → 事件构建 → 70/15/15 划分
  │
  ▼ procnet_format/ (ProcNet 格式, 4,800 篇)
  │   train: 3,360 | dev: 720 | test: 720
  │
  ├──────────────────────────────────────────────┐
  │                                              │
  ▼ (导出 JSONL)                                 ▼ (W2NER 预测 → 导出)
sidecar_entities_gold/                    sidecar_entities/
(Gold sidecar, 上限实验)                  (W2NER 预测 sidecar)
32,727 实体, 3,360 文档                   用途: 级联实验
  │                                              │
  ▼                                              ▼
┌────────────────────────────────────────────────────────┐
│                  DocEEProcessor                         │
│                                                         │
│  • 解析 procnet_format JSON                             │
│  • 解析复合 key "文本#sent_b_e#类型"                     │
│  • 加载 sidecar JSONL → doc.typed_entities              │
│  • 构建 DocEEDocumentExample                            │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│                   DocEEPreparer                         │
│                                                         │
│  • 逐字 tokenize → BIO tags → 词表构建                  │
│  • 按 max_len 切分为 fragment                           │
│  • 构建 entity node dict (flat_b, flat_e, type_id...)   │
│  • 输出 DataLoader                                      │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│               DocEEProxyNodeModel                       │
│                                                         │
│  • BERT → lm_hidden_states                              │
│  • BIO 预测 + loss                                      │
│  • 从 entity nodes 构建 span 节点                       │
│  • GCN 消息传递 (proxy + CLS + span)                    │
│  • 事件类型/角色/数量预测                                │
│  • Loss: BIO + event_type + event_relation + Hausdorff  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│                   DocEETrainer                          │
│                                                         │
│  • 训练循环 + 梯度累积                                  │
│  • Top-K checkpoint 保存（按 dev F1）                   │
│  • 评估 + DocEEMetric                                   │
│  • 最佳 epoch 预测输出 → {name}_predictions.json        │
└────────────────────────────────────────────────────────┘
```

### 关键设计

| 设计 | 说明 |
|------|------|
| **复合 key** | `文本#sentIdx_start_end#类型`，区分同文本多位置/多类型 |
| **索引口径** | 字级、左闭右开 `[start, end)`，与 W2NER 一致 |
| **Sidecar 机制** | JSONL 格式存储预提取实体，按 `doc_id` 匹配加载 |
| **Checkpoint 策略** | 按 dev F1 保留 Top-K，自动清理低分 checkpoint |
| **预测输出** | 训练结束后自动保存最佳 epoch 的逐文档预测 |
| **两种实验模式** | Gold sidecar（上限）vs W2NER sidecar（级联） |

---

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.4（GPU 训练）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 关键依赖

| 包 | 说明 |
|----|------|
| torch | 深度学习框架 |
| torch-geometric | 图神经网络（GCN） |
| transformers | 预训练语言模型 |
| numpy | 数值计算 |
| tqdm | 进度条 |

### 预训练模型

默认使用 `chinese-roberta-wwm-ext`，路径在 `run.py` 中配置。可在命令行通过 `--model_name` 修改。

---

## 快速开始

### 1. 数据准备

确保以下目录存在且包含正确格式的数据：

```bash
# 源数据（RASA NLU 格式）
data_v1b/mixed_data_with_queries/

# 已转换的 ProcNet 格式
procnet_format/mixed_data_with_queries/

# Sidecar 实体（二选一）
sidecar_entities_gold/     # Gold 标注（上限实验）
sidecar_entities/          # W2NER 预测（级联实验）
```

### 2. 冒烟测试（验证链路）

```bash
python run_1epoch_test.py
```

验证 W2NER sidecar 能否正确加载到 ProcNet 训练流程中。

### 3. 基础训练

```bash
bash run.sh
```

或使用 `run.py` 直接指定参数：

```bash
python run.py \
  --run_save_name=exp0 \
  --batch_size=32 \
  --epoch=100
```

---

## 训练命令

### 完整参数列表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run_save_name` | 本次运行的名称（必填） | — |
| `--batch_size` | 梯度累积步数 | `32` |
| `--epoch` | 训练轮数 | `50` |
| `--read_pseudo` | 是否读取伪数据集 | `false` |
| `--dataset_dir` | 数据集目录路径 | `./Data` |
| `--typed_entities_dir` | Sidecar 实体目录 | `./tmp_sidecar` |
| `--use_procnet_entity_nodes` | 模型是否使用 sidecar 实体节点 | `true` |
| `--return_procnet_entity_nodes` | Processor/Preparer 是否返回实体节点 | 同 `use_procnet_entity_nodes` |
| `--use_procnet_pred_entities` | Processor 是否加载 sidecar | `true` |
| `--proxy_slot_num` | 代理节点槽位数 | `16` |
| `--node_size` | 节点隐藏层大小 | `512` |
| `--max_len` | 最大 token 长度 | `510` |
| `--model_name` | 骨干模型路径 | `../models/chinese-roberta-wwm-ext` |
| `--device` | 训练设备 | `cuda` |
| `--data_loader_shuffle` | 是否打乱训练数据 | `true` |
| `--save_top_k` | 保留 Top-K 个最佳 checkpoint（-1=全部） | `1` |

### 实验模式

#### 实验 A：Gold Sidecar（上限实验）

```bash
python run.py \
  --run_save_name=gold_sidecar_exp \
  --dataset_dir=./procnet_format/mixed_data_with_queries \
  --typed_entities_dir=./sidecar_entities_gold \
  --use_procnet_pred_entities=true \
  --use_procnet_entity_nodes=true \
  --batch_size=32 \
  --epoch=100
```

#### 实验 B：W2NER Sidecar（级联实验）

```bash
python run.py \
  --run_save_name=w2ner_sidecar_exp \
  --dataset_dir=./procnet_format/mixed_data_with_queries \
  --typed_entities_dir=./sidecar_entities \
  --use_procnet_pred_entities=true \
  --use_procnet_entity_nodes=true \
  --batch_size=32 \
  --epoch=100
```

#### 无 Sidecar 基线

```bash
python run.py \
  --run_save_name=baseline_no_sidecar \
  --dataset_dir=./procnet_format/mixed_data_with_queries \
  --use_procnet_pred_entities=false \
  --use_procnet_entity_nodes=false \
  --batch_size=32 \
  --epoch=100
```

### 输出结果

训练结果保存在 `Result/{run_save_name}/` 目录下：

```
Result/
└── {run_save_name}/
    ├── {run_save_name}_001.json          # 每 epoch 聚合指标
    ├── {run_save_name}_002.json
    ...
    ├── {run_save_name}_{epoch}.json
    └── {run_save_name}_predictions.json  # 最佳 epoch 逐文档预测
```

每个 epoch JSON 包含 dev/test 的 BIO 指标和事件抽取指标（Event F1/P/R、各事件类型 F1）。

`{run_save_name}_predictions.json` 包含：
- `best_epoch` — 最佳 epoch 编号
- `best_dev_f1` — 最佳 dev Event F1
- `dev_predictions` — dev 集逐文档预测（doc_id, BIO_ans, event_ans 等）
- `test_predictions` — test 集逐文档预测

---

## W2NER 耦合

### 耦合范式

本项目采用 **Sidecar 范式（范式 B）**：保留 ProcNet 原始事件样本不变，将 W2NER 预测的实体作为 sidecar 输入注入 ProcNet 训练流程。

```
┌─────────────────────────────────────────────────────┐
│  范式 A（不采用）：把 W2NER 输出完全改写为 ProcNet 样本  │
│  范式 B（采用）：保留 ProcNet 原始样本，W2NER 实体作 sidecar │
└─────────────────────────────────────────────────────┘
```

这种设计的优势：
- ProcNet 的事件监督信号保持完整
- W2NER 只负责提供高质量的 mention/entity 候选层
- 事件类型/角色组装由 ProcNet 自行完成

### 耦合原理

ProcNet 原本使用 gold 实体标注进行事件抽取。本扩展引入 W2NER 预测的实体作为 sidecar 输入，实现端到端的级联：

```
W2NER 预测 → sidecar JSONL → ProcNet entity nodes → 事件抽取
```

### 两层耦合策略

W2NER 的预测结果在 ProcNet 中扮演 **mention/entity 候选层**，而非直接替代完整事件层：

```
┌──────────────────────────────────────────────┐
│  Mention 层（W2NER 直接提供）                   │
│  • 实体边界 (b/e)                              │
│  • 实体文本                                    │
│  • 实体类型候选                                 │
│  • 预测分数                                    │
├──────────────────────────────────────────────┤
│  Event 层（ProcNet / 规则组装）                  │
│  • 事件类型判定                                 │
│  • 角色分配                                    │
│  • 事件数量预测                                 │
│  • 多事件关系                                   │
└──────────────────────────────────────────────┘
```

**为什么不让 W2NER 直接替代事件层？**
- W2NER 是实体识别模型，不建模事件间的全局依赖关系
- 同一文本可能对应多个角色（如 "泰安" 同时是 departureCity 和 departureStation）
- ProcNet 的代理节点机制和 Hausdorff 距离最小化专门为此设计

### Sidecar 格式

Sidecar 文件为 JSONL 格式，每行一个文档：

```jsonl
{"doc_id": "doc_000007", "typed_entities": [
  {"text": "春秋航空", "type_name": "orderApp", "b": 1, "e": 5, "sent_idx": 0, "source": "w2ner"}
]}
```

| 字段 | 说明 |
|------|------|
| `doc_id` | 文档标识符 |
| `typed_entities` | 实体列表 |
| `text` | 实体文本 |
| `type_name` | 实体类型（ProcNet field 名） |
| `b`, `e` | 字级左闭右开索引 |
| `sent_idx` | 句子索引 |
| `source` | 数据来源（`procnet_gold` 或 `w2ner`） |

### 类型映射

W2NER 的实体类型通过驼峰归一化映射到 ProcNet 的事件角色字段。当前模型已直接输出细粒度类型（`startdate`/`enddate`/`starttime`/`endtime`），不再需要泛化 `date`/`time`。

#### 稳定映射（直接驼峰归一化）

| W2NER type | ProcNet field | W2NER type | ProcNet field |
|------------|---------------|------------|---------------|
| `orderapp` | `orderApp` | `seatclass` | `seatClass` |
| `seatnumber` | `seatNumber` | `seattype` | `seatType` |
| `departurestation` | `departureStation` | `arrivalstation` | `arrivalStation` |
| `departurecity` | `departureCity` | `arrivalcity` | `arrivalCity` |
| `vehiclenumber` | `vehicleNumber` | `dateofbirth` | `dateOfBirth` |
| `cardnumber` | `cardNumber` | `cardaddress` | `cardAddress` |
| `ordernumber` | `orderNumber` | `ethnicgroup` | `ethnicGroup` |
| `validfrom` | `validFrom` | `validto` | `validTo` |
| `idnumber` | `idNumber` | `roomtype` | `roomType` |
| `ticketgate` | `ticketGate` | `person` | `person` |
| `name` | `name` | `price` | `price` |
| `status` | `status` | `address` | `address` |
| `city` | `city` | `gender` | `gender` |

#### 时间类型（已解决语义分裂）

| W2NER type | ProcNet field | 说明 |
|------------|---------------|------|
| `startdate` | `startDate` | ✅ 直接映射 |
| `enddate` | `endDate` | ✅ 直接映射 |
| `starttime` | `startTime` | ✅ 直接映射 |
| `endtime` | `endTime` | ✅ 直接映射 |

> **历史说明**：早期版本中 W2NER 将所有日期合并为泛化 `date`、时间合并为 `time`，导致无法区分 start/end。已通过不折叠策略修复，当前模型直接输出 4 个细粒度类型。

### 复合 Key 设计

ProcNet 使用复合 key 区分同文本多位置/多类型的实体：

```
文本#sentIdx_start_end#类型
```

例如：`泰安#0_21_23#departureCity` 和 `泰安#0_21_23#departureStation` 是两个独立的 entry。

这个设计由修改版 ProcNet 的 `DocEE_processor.py` 支持，通过正则 `^(.*)#(\d+)_(\d+)_(\d+)#([^#]+)$` 解析。

### 关键配置参数

ProcNet 通过以下参数控制 sidecar 行为：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--use_procnet_pred_entities` | 是否加载 sidecar 实体 | `true` |
| `--use_procnet_entity_nodes` | 模型是否使用 sidecar 节点 | `true` |
| `--typed_entities_dir` | sidecar 目录路径 | `./sidecar_entities` 或 `./sidecar_entities_gold` |

### 数据转换脚本

#### 脚本 1：`convert_data_v1b_to_procnet.py`

将 RASA NLU 格式的源数据转换为 ProcNet 格式。

```bash
python scripts/convert_data_v1b_to_procnet.py \
  --input_dir ./data_v1b/mixed_data_with_queries \
  --output_dir ./procnet_format/mixed_data_with_queries \
  --split all
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | RASA NLU 源数据目录 | — |
| `--output_dir` | ProcNet 格式输出目录 | — |
| `--split` | 处理的分割（`all`/`train`/`test`） | `all` |
| `--event_mapping` | intent → 事件类型的 JSON 映射文件 | `None` |
| `--max_docs` | 最大转换文档数（测试用） | 全部 |

**输出**：`{output_dir}/{train,dev,test}.json`，格式为 `[[doc_id, {sentences, ann_mspan2dranges, ann_mspan2guess_field, recguid_eventname_eventdict_list}]]`。

#### 脚本 2：`convert_procnet_to_w2ner.py`

将 ProcNet 格式转换为 W2NER 训练所需的句子级格式。

```bash
# 目录模式（推荐）
python scripts/convert_procnet_to_w2ner.py \
  --input_dir ./procnet_format/mixed_data_with_queries \
  --output_dir ../W2NER/data/mixed_data_with_queries

# 单文件模式
python scripts/convert_procnet_to_w2ner.py \
  --input ./procnet_format/mixed_data_with_queries/train.json \
  --output ../W2NER/data/mixed_data_with_queries/train.json
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` / `--input` | ProcNet 输入目录或文件 | — |
| `--output_dir` / `--output` | W2NER 输出目录或文件 | — |
| `--fold_role_types` | 折叠 startDate/endDate→date, startTime/endTime→time | `False` |
| `--no_strict_alignment` | 关闭 span 文本严格对齐检查 | `False` |
| `--keep_duplicates` | 保留同 span 多类型实体 | `True` |
| `--slim_entities` | 不保留原始类型信息 | `False` |
| `--write_manifest` | 写入转换清单文件 | `False` |

**输出**：`{output_dir}/{train,dev,test}.json`，格式为 `[{sample_id, doc_id, sent_id, text, sentence, ner, entities}]`。

#### 脚本 3：`export_doc_typed_entities.py`

将 W2NER 句子级预测聚合为文档级 sidecar JSONL，供 ProcNet 使用。

```bash
python scripts/export_doc_typed_entities.py \
  --source_json ../W2NER/data/mixed_data_with_queries/test.json \
  --pred_json ../W2NER/predictions/test.json \
  --output_jsonl ./sidecar_entities/test_doc_typed_entities.jsonl \
  --report_json ./sidecar_entities/test_export_report.json
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source_json` | 源数据 JSON（用于对齐句子） | — |
| `--pred_json` | W2NER 预测输出 JSON | — |
| `--output_jsonl` | 输出的文档级 sidecar JSONL | — |
| `--report_json` | 导出报告（含验证统计） | — |
| `--no_strict_sentence_match` | 关闭句子文本严格匹配 | `False` |
| `--no_strict_range_check` | 关闭 span 范围严格检查 | `False` |

**输出**：每行一个文档的 JSONL 文件，格式为 `{"doc_id": "...", "typed_entities": [...]}`。

---

## EPAL 集成规划

### 背景

[EPAL](https://arxiv.org/abs/2501.00000)（Hu et al., 2025）提出了一种文档级多事件抽取方法，核心贡献包括：

1. **事件特定探针（Event-specific Probe）** — 为每个候选实体初始化探针，作为事件实例检测器
2. **事件特定参数库（Event-specific Argument Library）** — 在每个事件视图下重建候选参数表示
3. **角色索引槽位填充（Role-indexed Slot Filling）** — 对每个角色，从候选参数中选择最优填充

### 当前瓶颈

W2NER → ProcNet 级联已打通，但存在两个结构性问题：

| 问题 | 原因 |
|------|------|
| **跨事件实体复用** | 同一实体出现在多个事件中，ProcNet 可能合并或混淆这些事件 |
| **同事件多角色冲突** | 同一实体在同一事件中应填充多个角色，但 ProcNet 的实体索引分类倾向于将实体推向单一角色 |

### EPAL 的解决方案

EPAL 的**角色索引槽位填充**机制天然解决上述问题：

- **当前 ProcNet**：实体索引 — 对每个实体，分类到一个角色/null
- **EPAL 方式**：角色索引 — 对每个角色，从候选参数中选择一个

```
当前:  实体 → [角色A / 角色B / 角色C / null]  (每个实体只能选一个)
EPAL:  角色A → [实体1 / 实体2 / 实体3 / CLS]  (每个角色独立选择)
       角色B → [实体1 / 实体2 / 实体3 / CLS]  (同一实体可被多个角色选中)
```

### 三阶段集成路线

#### Stage 1: EPAL-lite（最小改动）

- 保留 ProcNet 代理节点作为事件假设
- 为每个代理节点构建事件特定参数库
- 将事件解码替换为角色索引槽位填充
- 每个事件添加虚拟 CLS 候选表示缺失角色

**预期收益**：立即支持同事件多角色复用，更清晰的槽位式事件表

#### Stage 2: 事件条件参数对比

- 计算事件条件下的参数表示
- 添加角色对比损失（role contrastive loss）

**预期收益**：减少同质事件间的混淆，提升多事件召回率

#### Stage 3: 对齐感知的代理优化

- 添加代理到事件的对齐机制
- 从高置信度实体初始化部分代理假设

**预期收益**：减少事件坍缩，多事件文档训练更稳定

### 参考

详细分析见 `epal_procnet_report_and_dialogue_summary.md`。

---

## 项目结构

```
procnet/
├── run.py                              # 主入口，参数解析 + 训练流程编排
├── run.sh                              # 快捷训练脚本（GPU 0, batch=32, epoch=100）
├── run_1epoch_test.py                  # 冒烟测试（验证 sidecar 加载链路）
├── verify_procnet_trainer_one_sample.py # 单样本验证（调试 forward pass）
│
├── procnet/                            # 核心库
│   ├── conf/
│   │   ├── basic_conf.py               # BasicConfig（学习率、epoch、device 等）
│   │   ├── DocEE_conf.py               # DocEEConfig（proxy_slot_num, node_size, save_top_k 等）
│   │   └── global_config_manager.py    # 全局路径配置
│   ├── data_example/
│   │   ├── DocEEexample.py             # DocEEDocumentExample, DocEEEntity, DocEETypedEntity
│   │   └── DuEEfin_example.py
│   ├── data_processor/
│   │   ├── DocEE_processor.py          # 解析 JSON + 加载 sidecar + 复合 key 解析
│   │   └── DuEE_fin_processor.py
│   ├── data_preparer/
│   │   ├── DocEE_preparer.py           # Tokenize + BIO + fragment 切分 + DataLoader
│   │   └── DuEE_fin_preparer.py
│   ├── model/
│   │   ├── DocEE_proxy_node_model.py   # BERT + BIO + GCN + Hausdorff 损失
│   │   └── basic_model.py
│   ├── trainer/
│   │   ├── DocEE_proxy_node_trainer.py # 训练循环 + Top-K checkpoint + 预测输出
│   │   └── basic_trainer.py
│   ├── metric/
│   │   ├── DocEE_metric.py             # BIO 评分 + 事件表格填充指标
│   │   └── basic_metric.py
│   ├── optimizer/
│   │   └── basic_optimizer.py          # 优化器封装（梯度累积 + 模型保存）
│   ├── dee/                            # Doc2EDAG 指标代码
│   ├── utils/                          # 工具函数（UtilData, UtilString, UtilStructure）
│   └── data_example/                   # 数据示例类
│
├── scripts/                            # 数据转换与验证脚本
│   ├── data_paths.py                   # 集中路径配置
│   ├── convert_data_v1b_to_procnet.py  # RASA NLU → ProcNet 格式
│   ├── convert_procnet_to_w2ner.py     # ProcNet → W2NER 句子级格式
│   ├── export_doc_typed_entities.py    # W2NER 预测 → 文档级 sidecar JSONL
│   ├── check_data_loss.py              # 数据流水线损失检查
│   ├── check_data_pipeline_alignment.py # 数据一致性 MD5 校验
│   ├── check_full_pipeline_alignment.py # 全流水线对齐检查
│   ├── full_data_pipeline_check.py     # 完整流水线检查
│   ├── check_data_v1b_procnet.py       # 源数据验证
│   └── compare_with_original_v1b.py    # 原始数据对比
│
├── data_v1b/                           # 源数据（RASA NLU 格式）
├── procnet_format/                     # 转换后的 ProcNet 格式（train/dev/test）
├── sidecar_entities_gold/              # Gold sidecar（上限实验）
├── sidecar_entities/                   # W2NER 预测 sidecar（级联实验）
├── conversation_summaries/             # 对话记录
├── figures/                            # 架构图
├── Checkpoint/                         # 模型 checkpoint（按 run_save_name 分组）
├── Result/                             # 训练结果（每 epoch JSON + 最佳预测）
├── requirements.txt
├── AGENTS.md                           # AI 代理开发指南
└── README.md                           # 本文件
```

---

## 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| `date`/`time` 语义分裂 | ✅ 已修复 | W2NER 已直接输出 startdate/enddate/starttime/endtime |
| W2NER 全量预测覆盖不足 | ✅ 已解决 | 全量 4,800 篇已完成预测（Entity F1 ≈ 95%） |
| 同文本多类型覆盖 | ✅ 已解决 | 复合 key 设计已解决存储层冲突 |
| `event_dict` 初始化顺序 | ✅ 已修复 | `_collect_procnet_type_ids()` 移到文档赋值之后 |
| 跨事件实体复用 | ⚠️ 待解决 | 计划通过 EPAL 角色索引槽位填充解决 |
| 同事件多角色冲突 | ⚠️ 待解决 | 计划通过 EPAL 角色索引槽位填充解决 |
| Event relation 实际匹配率 | 🟡 待验证 | 理论上成立，建议加运行时日志验证 |

---

## 引用

如果使用了本项目的代码或模型，请引用原始论文：

```bibtex
@inproceedings{wang-etal-2023-document,
  title = "Document-Level Multi-Event Extraction with Event Proxy Nodes and Hausdorff Distance Minimization",
  author = "Wang, Xinyu and Gui, Lin and He, Yulan",
  booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2023",
  address = "Toronto, Canada",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.acl-long.563",
  pages = "10118--10133",
}
```

---

## 许可证

本项目采用 MIT 许可证。
