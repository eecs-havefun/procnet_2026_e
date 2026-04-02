# W2NER × ProcNet 对话摘要

生成日期：2026-04-02

---

## 1. 背景与目标

本轮对话围绕 **W2NER 与 ProcNet 的数据耦合** 展开，核心问题是：

- 如何把 W2NER 的预测结果安全地接入 ProcNet
- 现有转换方案有哪些结构性风险
- 修改版仓库与官方仓库之间有哪些关键差异
- 当前 W2NER 输出已经增强后，下一步应该如何落地使用

对话最终逐步收敛到一个更明确的工程结论：

> 当前重点已经不再是 `b/e` 索引本身，而是 **W2NER 输出在 ProcNet 中应该扮演哪一层角色**，以及 **字段 / role 语义是否对齐**。

---

## 2. 最初阶段：对耦合方案的第一轮诊断

最开始的判断是：你的 JSON 外形已经接近 ProcNet 可读格式，但有若干高风险点需要优先核查。

### 第一轮识别出的主要风险

1. **`b/e` 索引口径可能不一致**
   - ProcNet 要求 `entity.span == sentence[start:end]`
   - 一旦 W2NER 的位置不是对最终字符串的字符偏移，Processor 就可能断言失败

2. **W2NER 的不连续 / 嵌套实体能力与 ProcNet 的连续 span 结构不完全一致**
   - 如果输出中存在 discontinuous / nested span，而 ProcNet 只接受连续 span，会造成信息丢失或投影错误

3. **同一文本在同一文档中对应多个类型时，可能出现覆盖**
   - 尤其是把“实体文本”直接作为主键时，这类冲突风险高

4. **`recguid_eventname_eventdict_list = []` 只能说明“能解析”，不能说明“完成事件耦合”**
   - 如果事件结构为空，更像是只喂了实体，没有真正喂事件监督

5. **`ann_mspan2guess_field` 的语义可能不是通用 NER type，而是事件角色槽位**
   - 这意味着简单把 `person/location/...` 填进去，格式上可过，但语义可能偏离 ProcNet 原生设定

---

## 3. 为继续核查而整理的 6 项补充材料

为了把问题从“原则风险”推进到“字段级核查”，后续整理出 6 项必须补充的信息：

1. 一条原始 W2NER 输出  
2. 一条转换后的 ProcNet 样本  
3. `procnet_entities` 中 `b/e` 的索引语义  
4. `end` 是否采用左闭右开  
5. `sentence` 到底是字列表、词列表还是 subword 列表  
6. ProcNet 当前实际走的是哪条读取 / 训练分支

之后这些内容被单独整理成了一个 Markdown 清单，并用于后续逐项核查。

---

## 4. 结合你自己的仓库与官方仓库做逐项核查

在你提供了仓库信息后，核查范围扩展到四个仓库：

### 你的仓库
- `eecs-havefun/procnet_2026_e`
- `eecs-havefun/W2NER_2026_e`

### 官方上游仓库
- `xnyuwg/procnet`
- `ljynlp/W2NER`

核查后发现一个关键事实：

> 你当前的 ProcNet 已经不是“官方原样 ProcNet”，而是加入了 **typed entities sidecar**、增强配置项、复合 key 兼容等机制的修改版实现。

这意味着问题不再是“如何严格适配官方 ProcNet 唯一路径”，而是要先分清楚你当前到底在走哪一种耦合范式。

---

## 5. 对 6 项材料的逐项结论

### 5.1 原始 W2NER 输出
结论：
- **成立，但它不是官方 W2NER 的默认输出**
- 它是你 fork 后专门为 ProcNet 耦合扩展出来的输出结构

也就是说，当前看到的 `procnet_entities`、`b/e/type/text/score` 这套东西，本质上已经是“为 ProcNet 对接准备过的 W2NER 输出”。

### 5.2 ProcNet 格式样本
结论：
- **对你修改版 ProcNet 基本成立**
- **对官方原版 ProcNet 不直接成立**

原因在于：
- 你现在用的是复合 key，例如  
  `文本#sentIdx_start_end#类型`
- 你修改过的 `DocEE_processor.py` 已能解析这种 key
- 官方原版不会自动兼容这套 key

### 5.3 `b/e` 的索引说明
结论：
- 在当前链路里是成立的
- 因为 `sentence` 是字级列表，所以 `b/e` 同时等于字级字符索引和 token 索引

### 5.4 `end` 边界
结论：
- 明确采用 **左闭右开 `[start, end)`**
- 这一点是链路中最稳定的一项

### 5.5 `sentence` 粒度
结论：
- 当前实现下是 **字列表**
- 但这只是当前数据链路的结论，不应泛化为官方 W2NER 的一般事实

### 5.6 ProcNet 读取配置
结论：
- 你的当前入口已经不只是“官方 `run.py + read_pseudo=False`”
- 真正关键的已经变成：
  - `typed_entities_dir`
  - `use_procnet_pred_entities`
  - `use_procnet_entity_nodes`
  - sidecar typed entities 的读取与使用

---

## 6. 第一版明确判断

在完成仓库与样本对照后，明确判断收敛为：

1. **你的方案对修改版 ProcNet 基本兼容**
2. **对官方原版 ProcNet 不直接兼容**
3. **最大的风险已不再是单纯索引，而是 ontology / role mapping**
4. **你当前实际上混用了两种耦合范式**

这两种范式分别是：

### 范式 A
把 W2NER 输出完全改写成 ProcNet 风格样本

### 范式 B
保留 ProcNet 原始事件样本，把 W2NER 实体作为 sidecar 喂进去

而你的当前代码实现，整体更偏向 **范式 B**。

---

## 7. 针对代码审查反馈的复核结论

后续你提交了代码审查反馈，重点讨论了：

- event relation 是否天然失效
- `event_dict` 与 `span_tensor_span_to_index` 的 key space 是否匹配
- 当前最核心的问题到底是索引、键空间，还是语义映射

### 复核后的核心结论

最初曾怀疑：
> event relation learning 完全失效

但在追到 `DocEE_preparer` 之后，结论被修正为：

> 在 **character-level tokenization** 条件下，event relation 是存在成立路径的，不应被笼统地判成“必然失效”。

更稳妥的说法变成：

- 不存在此前假设的“必然型 key-space bug”
- 但是否真的有效学习，仍建议做运行时日志验证
- 真正更大的风险已经转移到：
  - 同文本多类型覆盖
  - entity text 被裁剪或改写
  - W2NER type 与 ProcNet role 的语义映射是否严格一致

---

## 8. schema / 映射材料核查后的新判断

你后来补充了 schema / 映射清单，核查后发现：

### 原先的最大结构性风险
- `date` / `time` 是泛化标签
- ProcNet 需要 `startDate/endDate` 与 `startTime/endTime`

这说明：
- 如果 W2NER 只有泛化 `date/time`
- 那就不可能无损映射到 ProcNet 事件角色

当时的判断是：

> 结构格式已经通了，但真正阻塞项还包括  
> `date/time` 的语义分裂、关键角色的低召回、以及同角色多值覆盖。

---

## 9. 一个关键更新：W2NER 输出已经升级

随后你给出的最新 W2NER 预测输出表明，当前模型已经不再只是输出泛化 `date/time`，而是已经能够直接输出：

- `startdate`
- `enddate`
- `starttime`
- `endtime`

同时还能输出大量更贴近 ProcNet role 的标签，例如：

- `departurecity`
- `departurestation`
- `arrivalcity`
- `arrivalstation`
- `seatclass`
- `seatnumber`
- `orderapp`
- `vehiclenumber`
- `name`
- `person`
- `status`
- `price`

而且你又明确补充了一个非常重要的新前提：

> 你已经修复了之前的问题，并且 **label F1 和 entity F1 都很高**

这一更新会明显改变之前的结论。

---

## 10. 更新后的判断：W2NER 预测结果应该怎么用

在最新输出条件下，结论更新为：

> **W2NER 预测结果最适合直接作为 ProcNet 的 mention / entity 输入层。**

更具体地说：

### 10.1 它不需要再从旧 `entity` 列表重建实体
因为当前输出已经带有：

- `doc_id`
- `sent_id`
- `sentence`
- `procnet_entities`
- 每个实体的 `b`
- `e`
- `type`
- `text`
- `score`

所以它本身就是一份“适合进入 ProcNet mention 层”的结构化实体输出。

### 10.2 它最适合扮演的角色
**不是直接替代完整事件层，而是先充当高质量的实体候选层。**

也就是：

- 先负责给出 mention 边界
- 再负责给出 mention 类型 / field 候选
- 再由 ProcNet 或后续逻辑决定：
  - 哪些 mention 组成事件
  - 哪些 mention 扮演哪个 role
  - 一个文档里有几个事件

### 10.3 现在大多数类型已经可以直接映射
由于最新输出已经显式包含：
- `startdate/enddate/starttime/endtime`

所以之前最棘手的 `date/time` 语义分裂问题，在你的当前版本里已经大幅缓解。

当前更合理的策略是：

#### 稳定字段
直接做同名 / 驼峰归一化映射，例如：
- `orderapp -> orderApp`
- `seatclass -> seatClass`
- `seatnumber -> seatNumber`
- `departurestation -> departureStation`
- `arrivalstation -> arrivalStation`
- `departurecity -> departureCity`
- `arrivalcity -> arrivalCity`
- `vehiclenumber -> vehicleNumber`
- `startdate -> startDate`
- `enddate -> endDate`
- `starttime -> startTime`
- `endtime -> endTime`

#### 事件层暂不直接替代
即使类型已经很好，也仍建议：
- mention 层直接使用 W2NER
- event 层保留 ProcNet 或规则进行组装

---

## 11. 对当前工程状态的最新一句话总结

对话最终已经收敛到下面这句最实用的判断：

> **现在的问题已经不是“W2NER 输出能不能转成 ProcNet 可读格式”，而是“如何把这份高质量 `procnet_entities` 输出稳定接入 ProcNet 的 mention 层，并在 event 层上做最小必要的 role / event 组装”。**

---

## 12. 明确的接入建议

在最新阶段，接入建议变成：

### Step 1：按 `doc_id` 聚合、按 `sent_id` 排序
把句子级输出还原成文档级输入。

### Step 2：直接用 `procnet_entities` 构建 mention 层
生成：
- `ann_mspan2dranges`
- `ann_mspan2guess_field`

### Step 3：保留复合 key
建议继续使用类似：

```text
{span_text}#{sent_idx}_{start}_{end}#{field}
```

因为这样可以稳定区分：
- 同文本不同位置
- 同文本不同类型

### Step 4：event 层单独处理
不要简单把所有预测实体直接塞成最终事件字典；更推荐：
- mention 层直接采用 W2NER
- event type / role 由 ProcNet 或规则进行二次决策

---

## 13. 最终产出的辅助文件

在整段对话中，陆续整理并生成了多份辅助 Markdown，主要包括：

1. 6 项补充信息清单  
2. 明确判断摘要  
3. type → field 核查表  
4. 需要补充的 schema 清单  
5. 当前结论摘要  
6. `procnet_entities` 接入 ProcNet 的摘要与脚本骨架

这些文件的共同作用是：把讨论从“口头判断”逐步收束成“可复查、可执行、可落地的工程文档”。

---

## 14. 当前共识（最终版）

截至这轮对话结束，已经形成的共识可以压缩为以下几点：

1. **你的当前 W2NER 输出已经足够强，不需要再从零设计新输出格式**
2. **当前输出最适合作为 ProcNet 的 mention / entity sidecar 输入**
3. **修改版 ProcNet 已支持复合 key 与 typed entities sidecar**
4. **你现在最应该做的不是继续纠结索引，而是固定接入路径**
5. **事件层不建议让 W2NER 直接全量替代，应由 ProcNet / 规则做后续组装**
6. **由于你现在已显式预测 `startdate/enddate/starttime/endtime` 且 F1 很高，之前关于泛化 `date/time` 的最大担忧已明显缓解**

---

## 15. 建议的下一步

如果继续推进，最值得优先做的事情是：

1. 固化 `TYPE_MAPPING`
2. 统一 `procnet_entities -> mention 层` 的转换脚本
3. 为 event 层保留最小必要的 schema 约束
4. 对少量重叠 / 边界歧义实体做审计
5. 在 ProcNet 训练和推理两条路径中分别验证收益

---

## 16. 一句话版本

> 这轮对话最终把问题从“W2NER 与 ProcNet 能不能耦合”推进到了“如何以最小风险把当前高质量 W2NER 输出直接作为 ProcNet 的 mention 层输入，并把事件层留给 ProcNet / 规则完成”。

