# W2NER `procnet_entities` → ProcNet 接入摘要与脚本骨架

> 用途：说明当前 **W2NER 输出已经包含 `procnet_entities`** 时，应该如何接入 ProcNet；并提供一份可直接改造的转换脚本骨架。  
> 说明：本摘要基于前面已经核对过的样本、schema 清单、代码审查结论整理而成。

---

## 一、当前最明确的判断

### 1. 你现在不用再从旧 `entity` 列表重建实体
如果 W2NER 的输出已经是这种结构：

```json
{
  "doc_id": "doc_000007",
  "sent_id": 0,
  "sentence": ["【","春","秋","航","空","】", "..."],
  "procnet_entities": [
    {
      "b": 1,
      "e": 5,
      "type": "orderApp",
      "text": "春秋航空",
      "score": 0.94
    }
  ]
}
```

那么正确做法不是再从旧的 `entity` 列表推 mention，而是：

- **直接使用 `procnet_entities`**
- 直接构造 ProcNet 的 mention 层
- 重点做：
  1. **按文档聚合**
  2. **字段名归一化**
  3. **事件 role 选择**

---

### 2. `procnet_entities` 最适合用作 ProcNet 的 mention 候选层
也就是说，W2NER 现在最适合负责：

- 实体边界
- 实体文本
- 句内位置
- 实体候选类型

然后把这些结果写入 ProcNet 的：

- `ann_mspan2dranges`
- `ann_mspan2guess_field`

而不是直接把它们无条件当成最终 `event_dict`。

---

### 3. 位置索引这条链路已经成立
前面已经确认过：

- `sentence` 是字级列表
- `b/e` 是字级字符索引
- 切片规则是左闭右开 `[start, end)`
- `b/e` 针对的是 `"".join(sentence)`
- 这和 ProcNet 的 `[sent_idx, start, end]` 口径一致

所以现在的问题已经不是 span 对不齐。

---

### 4. 复合 key 方案建议继续保留
建议继续使用这种 key：

```text
{span_text}#{sent_idx}_{start}_{end}#{field}
```

例如：

```text
春秋航空#0_1_5#orderApp
```

这样做的好处：

- 能区分同文本不同位置
- 能区分同文本不同类型
- 能避免 mention 层面的覆盖

---

## 二、W2NER 预测结果应该怎么用

### 1. mention 层：全部都可以进
对每个 `procnet_entities`：

- `sent_idx = sent_id`
- `start = b`
- `end = e`
- `text = text`（如果没有就用 `sentence_str[b:e]`）
- `field = type 归一化后的 ProcNet field`

写入：

- `ann_mspan2dranges`
- `ann_mspan2guess_field`

---

### 2. event role 层：不要把所有 type 都无脑透传
当前大多数类型可以直接进入 ProcNet 的 field 层，通常只需要：

- 直接同名
- 或驼峰归一化

例如：

- `orderapp -> orderApp`
- `seatclass -> seatClass`
- `seatnumber -> seatNumber`
- `seattype -> seatType`
- `departurestation -> departureStation`
- `arrivalstation -> arrivalStation`
- `departurecity -> departureCity`
- `arrivalcity -> arrivalCity`
- `vehiclenumber -> vehicleNumber`
- `person -> person`
- `price -> price`
- `status -> status`
- `address -> address`
- `city -> city`
- `roomtype -> roomType`
- `ticketgate -> ticketGate`
- `cardnumber -> cardNumber`
- `dateofbirth -> dateOfBirth`
- `validfrom -> validFrom`
- `validto -> validTo`

---

### 3. `date` / `time` 先不要直接当最终事件 role
这是当前最大的语义风险。

原因：

- W2NER 把所有日期合并成 `date`
- W2NER 把所有时间合并成 `time`

但 ProcNet 需要区分：

- `startDate` / `endDate`
- `startTime` / `endTime`

所以：

- `date` / `time` **可以先进入 mention 层**
- 但**不要直接无条件进入最终 event role**

更稳的做法是：

- 先保留为候选 mention
- 再通过规则或后续分类器判定它是 `start` 还是 `end`

---

### 4. 你现在最适合的接法
建议分成两层：

#### 稳定映射层
这部分可直接映射到 ProcNet field：

- `orderapp -> orderApp`
- `seatclass -> seatClass`
- `seatnumber -> seatNumber`
- `seattype -> seatType`
- `departurestation -> departureStation`
- `arrivalstation -> arrivalStation`
- `departurecity -> departureCity`
- `arrivalcity -> arrivalCity`
- `vehiclenumber -> vehicleNumber`
- `person -> person`
- `price -> price`
- `status -> status`
- `address -> address`
- `city -> city`
- `roomtype -> roomType`
- `ticketgate -> ticketGate`
- `cardnumber -> cardNumber`
- `dateofbirth -> dateOfBirth`
- `validfrom -> validFrom`
- `validto -> validTo`

#### 歧义映射层
这部分先只保留 mention，不要直接写最终事件角色：

- `date`
- `time`

---

## 三、推荐的数据流

```text
W2NER sentence-level output
  ↓
按 doc_id 聚合 / 按 sent_id 排序
  ↓
构建 ProcNet mention 层：
  - sentences
  - ann_mspan2dranges
  - ann_mspan2guess_field
  ↓
对稳定字段直接映射到 role 候选
  ↓
对 date/time 做二次判别（或暂不进入 role）
  ↓
生成 recguid_eventname_eventdict_list
```

---

## 四、脚本骨架（可直接改造）

```python
import json
from collections import defaultdict
from typing import Dict, List, Any, Optional

# ============================================================
# 1. 类型归一化映射
# ============================================================

TYPE_MAPPING = {
    # 直接同名
    "address": "address",
    "city": "city",
    "gender": "gender",
    "name": "name",
    "person": "person",
    "price": "price",
    "status": "status",

    # 驼峰归一化
    "arrivalcity": "arrivalCity",
    "arrivalstation": "arrivalStation",
    "cardaddress": "cardAddress",
    "cardnumber": "cardNumber",
    "dateofbirth": "dateOfBirth",
    "departurecity": "departureCity",
    "departurestation": "departureStation",
    "ethnicgroup": "ethnicGroup",
    "idnumber": "idNumber",
    "orderapp": "orderApp",
    "ordernumber": "orderNumber",
    "roomtype": "roomType",
    "seatclass": "seatClass",
    "seatnumber": "seatNumber",
    "seattype": "seatType",
    "ticketgate": "ticketGate",
    "validfrom": "validFrom",
    "validto": "validTo",
    "vehiclenumber": "vehicleNumber",

    # 歧义类型：先只进 mention 层，不直接进入最终事件 role
    "date": None,
    "time": None,
}

# 不同事件允许的字段（可按你的 schema 继续补）
ALLOWED_FIELDS = {
    "flight": {
        "person", "departureStation", "arrivalStation", "seatNumber",
        "vehicleNumber", "startDate", "endDate", "startTime", "endTime",
        "seatClass", "name", "orderApp", "price", "status",
        "departureCity", "arrivalCity", "orderNumber", "idNumber"
    },
    "hotel": {
        "person", "roomType", "startDate", "endDate", "name", "price",
        "orderApp", "city", "address", "status", "orderNumber", "idNumber"
    },
    "id_card": {
        "cardNumber", "person", "validTo", "dateOfBirth",
        "gender", "ethnicGroup", "validFrom", "cardAddress"
    },
    "train": {
        "person", "departureStation", "arrivalStation", "seatNumber",
        "vehicleNumber", "startDate", "endDate", "startTime", "endTime",
        "seatType", "orderApp", "price", "status", "ticketGate",
        "departureCity", "arrivalCity", "orderNumber", "idNumber"
    },
}

# ============================================================
# 2. 基础工具
# ============================================================

def normalize_type(raw_type: str) -> Optional[str]:
    """将 W2NER type 归一化为 ProcNet field。
    返回 None 表示这个类型当前不应直接进入最终事件 role。
    """
    if raw_type is None:
        return None
    return TYPE_MAPPING.get(raw_type.lower(), raw_type)


def safe_get_text(sentence_str: str, ent: Dict[str, Any]) -> str:
    """优先用 ent['text']，没有时用 sentence_str[b:e] 回填。"""
    b = ent["b"]
    e = ent["e"]
    text = ent.get("text")
    if text:
        return text
    return sentence_str[b:e]


def validate_entity(sentence_str: str, ent: Dict[str, Any]) -> None:
    """检查 b/e 与 text 是否一致。建议正式跑批前保留。"""
    b = ent["b"]
    e = ent["e"]
    text = safe_get_text(sentence_str, ent)
    sliced = sentence_str[b:e]
    if text != sliced:
        raise ValueError(
            f"Entity span mismatch: text={text!r}, sentence[{b}:{e}]={sliced!r}"
        )


def make_unique_key(span_text: str, sent_idx: int, start: int, end: int, field: str) -> str:
    """构造复合 key。"""
    return f"{span_text}#{sent_idx}_{start}_{end}#{field}"


# ============================================================
# 3. 文档聚合
# ============================================================

def group_by_doc(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["doc_id"]].append(sample)
    for doc_id in grouped:
        grouped[doc_id].sort(key=lambda x: x["sent_id"])
    return grouped


# ============================================================
# 4. mention 层构建
# ============================================================

def build_mentions_for_doc(doc_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """只构建 ProcNet mention 层，不强行生成最终事件 dict。"""
    sentences: List[str] = []
    ann_mspan2dranges: Dict[str, List[List[int]]] = defaultdict(list)
    ann_mspan2guess_field: Dict[str, str] = {}
    ann_valid_mspans: List[str] = []
    ann_valid_dranges: List[List[int]] = []

    for sent_sample in doc_samples:
        sent_idx = sent_sample["sent_id"]
        sentence_tokens = sent_sample["sentence"]
        sentence_str = "".join(sentence_tokens)
        sentences.append(sentence_str)

        for ent in sent_sample.get("procnet_entities", []):
            validate_entity(sentence_str, ent)

            b = ent["b"]
            e = ent["e"]
            raw_type = ent["type"]
            span_text = safe_get_text(sentence_str, ent)

            field = normalize_type(raw_type)
            # mention 层里仍然保留所有实体；歧义类型先用原始类型占位
            field_for_key = field if field is not None else raw_type

            unique_key = make_unique_key(span_text, sent_idx, b, e, field_for_key)
            drange = [sent_idx, b, e]

            ann_mspan2dranges[unique_key].append(drange)
            ann_mspan2guess_field[unique_key] = field_for_key
            ann_valid_mspans.append(span_text)
            ann_valid_dranges.append(drange)

    return {
        "sentences": sentences,
        "ann_valid_mspans": ann_valid_mspans,
        "ann_valid_dranges": ann_valid_dranges,
        "ann_mspan2dranges": dict(ann_mspan2dranges),
        "ann_mspan2guess_field": ann_mspan2guess_field,
    }


# ============================================================
# 5. event role 组装（骨架版）
# ============================================================

def guess_role_for_ambiguous_type(
    event_type: str,
    raw_type: str,
    span_text: str,
    sentence_str: str,
    start: int,
    end: int,
) -> Optional[str]:
    """对 date/time 这类歧义类型做二次判别。
    这里先留成骨架，你可以后续接规则或分类器。
    """
    raw_type = raw_type.lower()

    if raw_type == "date":
        # TODO:
        # 规则示例：
        # - 如果附近有“出发/发车/起飞/入住”，判为 startDate
        # - 如果附近有“到达/抵达/退房”，判为 endDate
        return None

    if raw_type == "time":
        # TODO:
        # 规则示例：
        # - 如果附近有“发车/起飞”，判为 startTime
        # - 如果附近有“到达/抵达”，判为 endTime
        return None

    return normalize_type(raw_type)


def build_event_dict_from_mentions(
    doc_samples: List[Dict[str, Any]],
    event_type: str,
    use_only_stable_fields: bool = True,
) -> Dict[str, str]:
    """从 W2NER 预测结果中组装事件字典的骨架。
    默认只放稳定字段；date/time 先不强行塞进去。
    """
    event_dict: Dict[str, str] = {}
    allowed = ALLOWED_FIELDS.get(event_type, set())

    for sent_sample in doc_samples:
        sentence_str = "".join(sent_sample["sentence"])

        for ent in sent_sample.get("procnet_entities", []):
            b = ent["b"]
            e = ent["e"]
            raw_type = ent["type"]
            span_text = safe_get_text(sentence_str, ent)

            norm_field = normalize_type(raw_type)

            # 稳定字段：直接使用
            if norm_field is not None:
                if norm_field in allowed and norm_field not in event_dict:
                    event_dict[norm_field] = span_text
                continue

            # 歧义字段：二次判别
            if not use_only_stable_fields:
                guessed_field = guess_role_for_ambiguous_type(
                    event_type=event_type,
                    raw_type=raw_type,
                    span_text=span_text,
                    sentence_str=sentence_str,
                    start=b,
                    end=e,
                )
                if guessed_field is not None and guessed_field in allowed and guessed_field not in event_dict:
                    event_dict[guessed_field] = span_text

    return event_dict


# ============================================================
# 6. 主转换函数
# ============================================================

def convert_w2ner_output_to_procnet(
    input_path: str,
    output_path: str,
    infer_event_type: bool = False,
) -> None:
    """将 W2NER sentence-level 输出转换为 ProcNet doc-level 格式。

    参数：
    - infer_event_type=False：只构建 mention 层，事件先留空
    - infer_event_type=True：尝试根据样本中的字段或外部逻辑补 event_type，并组装 event_dict
    """
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    grouped = group_by_doc(samples)

    procnet_docs = []

    for doc_id, doc_samples in grouped.items():
        base_doc = build_mentions_for_doc(doc_samples)

        # 事件层：建议初期先留空，等 date/time 和 role 逻辑稳定后再开启
        recguid_eventname_eventdict_list = []

        if infer_event_type:
            # TODO:
            # 你可以从每条 sample 里取 event_type / intent / query_label
            # 这里先留骨架
            event_type = None

            if event_type is not None:
                event_dict = build_event_dict_from_mentions(
                    doc_samples=doc_samples,
                    event_type=event_type,
                    use_only_stable_fields=True,  # 初期建议只用稳定字段
                )
                if event_dict:
                    recguid_eventname_eventdict_list.append([0, event_type, event_dict])

        base_doc["recguid_eventname_eventdict_list"] = recguid_eventname_eventdict_list
        procnet_docs.append([doc_id, base_doc])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(procnet_docs, f, ensure_ascii=False, indent=2)


# ============================================================
# 7. 示例入口
# ============================================================

if __name__ == "__main__":
    convert_w2ner_output_to_procnet(
        input_path="mixed_data_with_queries_unfolded_output.json",
        output_path="procnet_from_w2ner_mentions.json",
        infer_event_type=False,  # 先只生成 mention 层
    )
```

---

## 五、这份脚本骨架的使用建议

### 建议第一阶段
先把这个脚本只用来做：

- 文档聚合
- mention 层构建
- 稳定 field 映射

也就是先产出：

- `sentences`
- `ann_mspan2dranges`
- `ann_mspan2guess_field`
- `recguid_eventname_eventdict_list = []`

这样你可以先确认：

1. mention 层是否稳定
2. ProcNet 是否能正常读取
3. 索引和文本校验是否全部通过

---

### 建议第二阶段
再单独增加：

- `event_type` 推断
- `date/time -> start/end` 判别
- event role 组装

这一步不建议一上来和 mention 层混在一起改。

---

## 六、一句话结论

**如果你的 W2NER 输出已经包含 `procnet_entities`，那它最合适的用途就是：直接作为 ProcNet 的 mention 候选输入；你下一步该做的是“文档聚合 + field 归一化 + 歧义 role 判别”，而不是重新发明一套实体抽取。**
