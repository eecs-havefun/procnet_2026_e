# W2NER ↔ ProcNet 当前结论整理

> 依据：当前已核查的数据样本、代码审查反馈、schema 清单与映射材料。  
> 用途：作为当前阶段的结论摘要，便于保存、转发或继续讨论。

---

## 一、现在已经能确定的事

### 1. 位置索引这条链路是通的

你这套 **W2NER → ProcNet** 在位置层面已经基本确认成立：

- `sentence` 是**字级列表**
- `b/e` 是**字级字符索引**
- 两边都用 **左闭右开** `[start, end)`
- 按 `doc_id` 聚合、按 `sent_id` 排序恢复文档，这个方向没问题

也就是说，**你现在的问题已经不是 span 对不齐**。

### 2. 复合 key 方案是成立的

你现在的 key 设计：

```text
文本#sentIdx_start_end#类型
```

这个方案已经验证过，优点很明确：

- 能区分同文本不同位置
- 能区分同文本不同类型
- 能避免 `ann_mspan2dranges` 层面的覆盖冲突

所以这部分可以继续保留。

### 3. event relation 不是“天然失效”

结合后续代码审查，之前最担心的那条已经可以修正：

- `event_dict` 初始是 `role -> entity_text`
- 到 `DocEE_preparer` 时，会把 `entity_text` 转成 `token_ids tuple`
- span node 也是同一 tokenizer 下的 `token_ids tuple`

在**字符级切分 + 同一 tokenizer + 文本一致**的前提下，event relation 是**可以对上的**，不是必然失效。

---

## 二、现在真正的主问题

### 1. `date` / `time` 的 schema 不匹配

这是目前最大的结构性问题。

你现在的 W2NER schema 里：

- `date` 是**泛日期**
- `time` 是**泛时间**

但 ProcNet 里需要的是：

- `startDate` / `endDate`
- `startTime` / `endTime`

也就是：

- **W2NER 把它们合并了**
- **ProcNet 把它们拆开了**

这意味着 `date` 和 `time` 不是直接驼峰化就能解决的，而是**语义分裂问题**。

### 2. W2NER 对关键角色的预测覆盖不够

当前几个点很关键：

- `person` 虽然 schema 里有，但实际预测表现偏弱
- `arrivalCity` 预测极少
- `idNumber`、`cardAddress` 等角色也偏弱

这意味着：

**格式能耦合，不等于信息够用。**

也就是：

- 你能把数据送进 ProcNet
- 但 ProcNet 学到的角色信息可能严重缺失

### 3. `name` 这类字段语义过宽

`name` 在 ProcNet 里是个合法 field，但现在的 W2NER `name` 是泛化名称：

- 可能是航司名
- 可能是酒店名
- 可能是别的机构名

所以 `name -> name` 虽然表面上“同名”，但语义并不天然安全。这个属于**中风险字段**。

---

## 三、映射层面的结论

### 基本稳定的部分

当前 28 个 W2NER type 里，**26 个基本都能通过“直接同名 / 驼峰归一化”进入 ProcNet**。例如：

- `orderapp -> orderApp`
- `seatclass -> seatClass`
- `seatnumber -> seatNumber`
- `departurestation -> departureStation`
- `arrivalstation -> arrivalStation`
- `departurecity -> departureCity`
- `arrivalcity -> arrivalCity`
- `vehiclenumber -> vehicleNumber`

这部分整体是稳定的。

### 真正不稳定的部分

只有两类是硬问题：

- `date -> startDate / endDate`
- `time -> startTime / endTime`

所以你现在可以把映射问题理解成：

- **大多数 type 已经能稳定映射**
- **少数关键时间类 type 还没有闭环**

---

## 四、同文本冲突这件事，怎么判断

当前统计显示：

- 3360 篇文档里，有 912 篇存在“同文本多类型”现象
- 典型例子：
  - `"泰安"` 同时是 `departureCity` 和 `departureStation`
  - `"12月19日"` 同时可能是 `startDate` 和 `endDate`

这里要分成两层看：

### 在实体存储层

问题不大，因为用了复合 key。

### 在事件语义层

还是有风险，因为：

- 同一个文本可以承担多个 role
- 而 W2NER 的泛化标签不一定能恢复这种区分
- 特别是 `date/time`，语义歧义最严重

所以这类冲突**不是 key 冲突**，而是**role 判定冲突**。

---

## 五、建议采用的最终口径

你现在完全可以把项目状态总结成下面这段：

> 当前 W2NER → ProcNet 的位置索引耦合已经验证成立；复合 key 方案可稳定避免同文本不同位置/类型的实体冲突；多数实体类型可通过直接同名或驼峰归一化映射到 ProcNet 字段。当前主要风险不在 span 边界，而在 `date/time` 与 `start/end` 时间角色之间的语义分裂，以及 W2NER 对部分关键角色（如 `person`、`arrivalCity`）的预测覆盖不足。

---

## 六、现在最值得做的 3 件事

### 1. 把映射分成两层

不要再把所有 type 当成同一种映射。

建议拆成：

- **稳定映射层**：26 个可直接映射的字段
- **歧义映射层**：`date` / `time`

### 2. 单独处理 `date` 和 `time`

这两个别继续“直接透传”。

至少要二选一：

- **保守策略**：统一落到 `startDate/startTime`
- **放弃策略**：先不把 `date/time` 喂给 ProcNet 的事件角色

### 3. 优先补 W2NER 的关键角色能力

如果 `person`、`arrivalCity` 这些角色预测很差，那即使 schema 对齐了，ProcNet 也学不到足够信息。

所以从效果角度，**提升这些关键类型的 recall**，优先级会高于继续修 span。

---

## 七、一句话版

**一句话：你的系统现在“数据格式已经通了”，但“事件语义还没完全通”；最大的剩余问题是 `date/time` 的语义分裂，以及关键角色预测不足。**
