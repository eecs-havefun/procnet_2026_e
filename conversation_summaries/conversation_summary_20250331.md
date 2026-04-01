# 对话总结 - ProcNet 与 W2NER 集成项目

*生成时间：2025年3月31日*  
*项目：ProcNet (文档级多事件抽取系统) 与 W2NER (实体识别) 集成框架*

## 项目目标

用户正在进行 NLP 研究，目标是构建一个结合 W2NER (实体识别) 和 ProcNet (文档级多事件抽取) 的耦合框架。主要目标包括：

1. **分析与修复** ProcNet 代码库中的问题
2. **理解** ProcNet 训练的数据管道
3. **配置** OpenCode (AI 编程助手) 以支持开发工作流

## 已完成的工作

### 1. 创建 AGENTS.md 开发指南
- 创建了详细的 AGENTS.md 文件 (~200 行)，包含：
  - 项目概述与环境设置
  - 构建与运行命令
  - 测试与验证流程
  - 代码风格规范 (导入、命名、类型注解等)
  - 项目结构与 Git 实践
  - 故障排除指南

### 2. ProcNet 代码库问题分析
发现了以下关键问题：

| 问题类型 | 具体问题 | 影响文件 |
|---------|---------|---------|
| 类型注解错误 | `str = None` 应改为 `Optional[str] = None` | `procnet/conf/basic_conf.py` |
| 配置属性缺失 | `DocEEConfig` 缺少 `use_procnet_pred_entities`, `lm_model_name` 等属性 | `procnet/conf/DocEE_conf.py` |
| 运行时风险 | `BasicPreparer` 属性初始化为 `None` 但模型访问时无空值检查 | `procnet/model/DocEE_proxy_node_model.py` |
| 代码质量 | 函数过长、硬编码值、日志记录不一致 | 多个文件 |
| 架构问题 | 初始化顺序不明确、全局状态管理 | - |
| 测试不足 | 无正式单元测试，仅验证脚本 | - |

### 3. 数据管道梳理
明确了数据转换流程：

```
data_v1b/ (RASA NLU 格式)
    ↓ scripts/convert_data_v1b_to_procnet.py
Data_v1b/ (ProcNet 格式 v1)
    ↓ scripts/convert_procnet_to_w2ner.py
procnet_format/ (ProcNet 格式 v2) ← 推荐用于训练
```

**关键数据目录：**
- `data_v1b/` - 原始 RASA NLU 格式数据
- `Data_v1b/` - ProcNet 格式 v1 (已转换)
- `procnet_format/` - ProcNet 格式 v2 (推荐训练数据)
- `sidecar_entities/` - W2NER 预测的实体
- `sidecar_entities_gold/` - 黄金标准实体

**数据拆分：**
- `procnet_format/mixed_data_with_queries/train/`
- `procnet_format/mixed_data_with_queries/dev/`
- `procnet_format/mixed_data_with_queries/test/`

### 4. OpenCode 配置修改
- **修改文件：**
  1. `/home/mengfanrong/finaldesign/W2NERproject/procnet/opencode.json`
  2. `/home/mengfanrong/finaldesign/W2NERproject/W2NER/opencode.json`
- **修改内容：** 添加 `"temperature": 0.2` 到 deepseek-reasoner 模型配置
- **研究状态：** 需要验证 OpenCode 配置格式的正确性
- **待处理：** 确认格式后应用到所有三个配置文件

### 5. 代码库探索
**分析的关键文件：**
- `procnet/conf/basic_conf.py` - 基础配置类
- `procnet/conf/DocEE_conf.py` - DocEE 特定配置
- `procnet/model/DocEE_proxy_node_model.py` - 代理节点模型
- `procnet/data_example/DocEEexample.py` - 数据示例类
- `procnet/data_processor/DocEE_processor.py` - 数据处理器

**关键脚本：**
- `scripts/convert_data_v1b_to_procnet.py` - 数据格式转换
- `scripts/convert_procnet_to_w2ner.py` - 格式转换
- `run_w2ner_sidecar_inference.py` - 集成管道
- `run_1epoch_test.py` - 训练测试脚本

## 当前状态

### ✅ 已完成
1. AGENTS.md 文件创建与完善
2. ProcNet 代码库问题全面分析
3. 数据管道与位置识别
4. 两个 opencode.json 文件的初步修改
5. OpenCode 配置文档研究

### 🔄 进行中
1. 验证 OpenCode 配置格式的正确性
2. 恢复 opencode.json 修改直到确认格式正确

### 📋 待办事项
1. 修复 ProcNet 代码库中的问题
2. 确认并应用正确的 OpenCode 温度配置到所有三个位置
3. 继续 W2NER 和 ProcNet 框架的集成

## 项目结构概览

```
ProcNet 仓库根目录/
├── AGENTS.md                    # AI 代理开发指南
├── opencode.json                # OpenCode 配置文件
├── procnet/                     # 核心代码库
│   ├── conf/                    # 配置类
│   ├── data_example/           # 数据示例类
│   ├── data_preparer/          # 数据准备
│   ├── data_processor/         # 数据处理器
│   ├── model/                  # 神经网络模型
│   └── utils/                  # 工具类
├── data_v1b/                   # 原始数据
├── Data_v1b/                   # 转换数据 v1
├── procnet_format/             # 转换数据 v2 (推荐)
├── sidecar_entities/           # W2NER 预测实体
├── scripts/                    # 转换脚本
└── conversation_summaries/     # 对话总结 (本文件所在)
```

## 相关仓库位置

1. **ProcNet 仓库：** `/home/mengfanrong/finaldesign/W2NERproject/procnet/`
2. **W2NER 仓库：** `/home/mengfanrong/finaldesign/W2NERproject/W2NER/`
3. **根配置文件：** `/home/mengfanrong/finaldesign/opencode.json`

## 下一步建议

1. **优先修复** ProcNet 代码库中的类型注解和配置问题
2. **验证**数据管道是否能够正常运行训练
3. **确认** OpenCode 配置格式，确保 AI 助手工作稳定
4. **逐步实施** W2NER 与 ProcNet 的端到端集成

---
*本总结由 OpenCode 助手生成，基于与用户的对话内容。*