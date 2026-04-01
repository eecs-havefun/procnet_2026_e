# W2NER → ProcNet `type -> field` 映射核查表（基于当前样本与代码审查）

> 生成日期：2026-04-01  
> 目的：基于当前你提供的样本、代码审查反馈、你的修改版仓库和上游官方仓库，整理一份可执行的 `W2NER type -> ProcNet field` 核查表。  
> 说明：这份表不是“全量语料统计结果”，而是“当前已确认 / 已暴露风险 / 需要继续验证”的审计版结论。

---

## 一、先给结论

### 已经可以明确的部分

1. **位置索引链路成立**
   - W2NER 的 `b/e` 与 ProcNet 的 `[sent_idx, start, end]` 是同一套字级、左闭右开的口径。
   - 按 `doc_id` 聚合、按 `sent_id` 排序恢复文档，这一部分没有结构性问题。

2. **event relation learning 不是天然失效**
   - 你这次代码审查已经把关键点补齐了：
     - `event_dict` 初始是 `role -> entity_text`
     - `DocEE_preparer` 会把 `entity_text` 再转成 `tuple(token_ids)`
     - span node 也是用同一 tokenizer 得到 `tuple(token_ids)`
   - 在 **character-level tokenize + 同一 tokenizer + 文本完全一致** 的条件下，event relation 的 key 实际是可以对上的。

3. **真正的主风险已经收敛到两类**
   - **同文本多类型覆盖**
   - **W2NER type 与 ProcNet field 的语义映射是否真的合理**

---

## 二、当前最可靠的总判断

### 判断 A：索引耦合已经成立
这一点不需要再怀疑。

### 判断 B：event relation 在你的当前实现里“有条件成立”
成立前提是：
- `event_dict` 里的 value 文本与 span 文本一致
- `sentences` 的切分和 `my_tokenize()` 的切分一致
- 两边都走同一 tokenizer

### 判断 C：当前最该查的不是 `b/e`
而是：
- 一个 type 到底映射到哪个 field
- 这个 field 是不是你任务里的“事件角色”
- 是否有重复文本导致覆盖

---

## 三、`W2NER type -> ProcNet field` 核查表（当前样本级）

| W2NER type | ProcNet field | 映射方式 | 当前判断 | 风险级别 | 说明 |
|---|---|---|---|---|---|
| `person` | `person` | 直接同名 | 可接受 | 低 | 样本中稳定；属于最自然的直接映射 |
| `orderapp` | `orderApp` | 大小写 / 驼峰归一化 | 可接受 | 低 | 只是命名规范转换，语义未变 |
| `seatclass` | `seatClass` | 大小写 / 驼峰归一化 | 可接受 | 低 | 语义基本保持一致 |
| `seatnumber` | `seatNumber` | 大小写 / 驼峰归一化 | 可接受 | 低 | 语义基本保持一致 |
| `price` | `price` | 直接同名 | 可接受 | 低 | 语义一致 |
| `date` | `startDate` | 语义收窄 | 需继续确认 | 中 | `date` 是泛化时间，`startDate` 是更具体角色；并非天然等价 |
| `departurestation` | `name` | 语义改写 | 高风险 | 高 | 这个不是简单重命名，而是角色语义明显变化 |
| `seattype` | `seatType` | 大小写 / 驼峰归一化 | 暂可接受 | 中 | 只在代码审查的 train 例子里出现，需要和正式样本继续对照 |
| `departurecity` | 暂未确认 | 未知 | 待查 | 中 | 代码审查里提到和 `departureStation` 可能同文本冲突 |
| `location` | 暂未确认 | 未知 | 待查 | 中 | 需要看你任务中是否作为通用地点实体还是事件角色 |
| `name` | `name` | 直接同名 | 不建议默认安全 | 中 | `name` 过于泛；要看它在你的事件中到底表示什么对象名 |

---

## 四、逐项解释

### 1. 低风险：只是命名规范变化
这一类通常问题不大，只要你在全链路中统一使用即可：

- `orderapp -> orderApp`
- `seatclass -> seatClass`
- `seatnumber -> seatNumber`
- `seattype -> seatType`

这些更像是：
- 小写 snake/flat 命名
- 到驼峰 field 名的规范化

如果代码和数据都统一，这类不是主要风险。

---

### 2. 中风险：从“通用实体类型”映射成“事件角色”
这一类需要确认是不是你任务本身就这样定义的：

- `date -> startDate`

问题不在能不能转，而在：
- `date` 可能是下单时间、乘车时间、出发时间、到达时间
- `startDate` 已经把它限定成“开始时间/出发时间”

如果数据本身所有 `date` 都真的是“开始时间”，那这个映射成立。  
如果不是，就会把本来更泛的实体错误压到一个具体角色上。

---

### 3. 高风险：语义改写，而不是映射
这一类是当前最值得优先盯的：

- `departurestation -> name`

这个映射的问题在于：
- `departurestation` 本来是“出发站/站点”
- `name` 是极宽泛的字段名
- 你给的 paired sample 里还有一个现象：
  - W2NER 预测文本：`首都航空机场`
  - ProcNet 最终事件值：`首都航空`

这已经不是“字段名改写”，而是：
- **文本裁剪**
- **角色重解释**
- **可能的语义漂移**

这一条必须重点排。

---

## 五、关于 event relation 的更新判断

### 我现在的明确修正

你这次反馈纠正了我之前最强的一个担忧：

> “event relation 会因为 key 类型不同而完全学不到”

这个判断现在要改成：

> **初始事件字典阶段确实是 `role -> text`，但在 `DocEE_preparer` 中，text 会被再次转成 token_ids tuple，因此在字符级切分条件满足时，最终匹配是可以成立的。**

也就是说：

- 不是“天然完全失效”
- 而是“依赖一组前提条件成立”

### 这组前提条件是

1. `event_dict` 里的 value 文本没有被改坏  
2. `sentences` 中的 span 文本与 value 文本一致  
3. `my_tokenize()` 与句子侧 span tokenization 一致  
4. 两边都走同一个 tokenizer

只要这四条成立，event relation 就能对上。

---

## 六、现在真正剩下的两大风险

### 风险 1：同文本多类型覆盖
这是你当前最现实、最容易 silently wrong 的问题。

典型场景：

- `"泰安"` 同时可能是 `departureCity`
- 也可能是 `departureStation`

如果你当前构造事件字典时写的是：

```python
event_dict[role_name] = entity_text
```

那当多个 role 的 value 恰好是同一个文本，或者同一个文本在不同角色中复用时，就会出现：
- 覆盖
- 丢角色
- 训练时只学到最后一个

### 风险 2：type 与 field 不是同一层语义
W2NER 更像是做实体识别；
ProcNet 更像是做事件角色抽取。

所以你要一直警惕：

- `type` 是不是“实体类别”
- `field` 是不是“事件参数角色”

这两个概念不是天然等价的。

---

## 七、建议你接下来优先做的 4 件事

### 1. 产出完整映射表
从你的转换脚本里把所有出现过的：

- W2NER type
- ProcNet field

全部导出来，做成全量表，而不是只靠样本。

建议最终表结构：

| W2NER type | ProcNet field | 映射规则来源 | 是否一一映射 | 是否人工确认 |
|---|---|---|---|---|

---

### 2. 统计“同文本多类型”冲突
至少统计这些指标：

- 同一文档中，是否出现同一文本对应多个 type
- 同一事件中，是否出现同一文本填多个 role
- 被覆盖的 role 数量

---

### 3. 在训练前加一致性断言
建议增加三条硬检查：

```python
assert event_value_text == sentence_text[start:end]
assert tokenize(event_value_text) == span_token_ids
assert field_name in allowed_fields_for_event_type[event_type]
```

---

### 4. 单独抽检高风险映射
优先抽检这几条：

- `date -> startDate`
- `departurestation -> name`
- 任何 `xxxstation / xxxcity / xxxname` 之间会发生重解释的映射

---

## 八、当前最推荐的最终口径

如果你现在要写到论文附录、实验记录或 README 里，我建议你用下面这段表述：

> 当前 W2NER → ProcNet 的位置索引转换已验证成立；  
> event relation 在字符级 tokenization 与统一 tokenizer 的条件下可正常对齐；  
> 目前主要待解决问题不在 span 边界，而在实体类型到事件角色字段的语义映射，以及同文本多角色覆盖风险。

---

## 九、一句话结论

**一句话：你现在的问题已经不是“能不能把实体位置转过去”，而是“转过去以后，这个 type 到底是不是你想让 ProcNet 学的那个 role”。**

---

## 十、附：当前建议优先级

### 最高优先级
- `departurestation -> name`
- 同文本多类型覆盖

### 次高优先级
- `date -> startDate`
- 所有“通用实体类型 -> 具体事件角色”的映射

### 低优先级
- 单纯驼峰化命名：
  - `orderapp -> orderApp`
  - `seatclass -> seatClass`
  - `seatnumber -> seatNumber`

---

## 十一、你下一步最适合继续要我的东西

如果继续往下做，我最建议你让我直接生成这两份之一：

1. **全量映射审计模板**
   - 你把脚本里所有 type/field 自动灌进去，我来帮你逐条判定

2. **冲突检查脚本设计说明**
   - 直接告诉你怎么扫出：
     - 同文本多类型
     - 同文本多角色
     - 文本裁剪不一致
     - token_ids 不一致
