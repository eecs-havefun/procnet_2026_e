# W2NER ↔ ProcNet：Schema / 映射材料清单（已填充版）

> 填充日期：2026-04-01
> 数据来源：实际代码 + 实际数据文件扫描（非人工提供）
> 
> 数据来源说明：
> - W2NER 类型 schema：从 `W2NER/mixed_data_full_output.json`（2165 条预测记录）提取
> - ProcNet 事件 schema：从 `procnet_format/mixed_data_with_queries/train.json`（3360 篇文档）提取
> - 映射关系：从 `scripts/convert_data_v1b_to_procnet.py` 代码逻辑 + sidecar 实际数据推断

---

## 1. W2NER 实体类型 schema

**来源**：`W2NER/mixed_data_full_output.json`（W2NER 在 mixed_data 上的预测输出，5.2MB，2165 条记录，720 篇文档）

| # | W2NER type | 出现次数 | 语义说明 | 备注 |
|---|-----------|---------|---------|------|
| 1 | `address` | 152 | 地址 | 酒店事件中的入住地址 |
| 2 | `arrivalcity` | 167 | 到达城市 | 航班/火车到达城市 |
| 3 | `arrivalstation` | 290 | 到达站点 | 航班/火车到达站点 |
| 4 | `cardaddress` | 20 | 身份证地址 | 身份证事件 |
| 5 | `cardnumber` | 197 | 身份证号 | 身份证事件 |
| 6 | `city` | 168 | 城市 | 酒店事件中的城市 |
| 7 | `date` | 651 | 日期（泛化） | ⚠️ 合并了 startDate/endDate |
| 8 | `dateofbirth` | 157 | 出生日期 | 身份证事件 |
| 9 | `departurecity` | 163 | 出发城市 | 航班/火车出发城市 |
| 10 | `departurestation` | 326 | 出发站点 | 航班/火车出发站点 |
| 11 | `ethnicgroup` | 142 | 民族 | 身份证事件 |
| 12 | `gender` | 158 | 性别 | 身份证事件 |
| 13 | `idnumber` | 29 | 证件号 | 航班/酒店事件中的证件号 |
| 14 | `name` | 339 | 名称（泛化） | ⚠️ 可能是公司名/航司名/酒店名 |
| 15 | `orderapp` | 471 | 订购平台 | 如"美团""携程" |
| 16 | `ordernumber` | 43 | 订单号 | 航班/酒店/火车订单号 |
| 17 | `person` | 726 | 人名 | 乘客/入住人/身份证持有人 |
| 18 | `price` | 447 | 价格 | 票价/房费 |
| 19 | `roomtype` | 232 | 房型 | 酒店事件 |
| 20 | `seatclass` | 167 | 座位等级 | 如"经济舱""公务舱" |
| 21 | `seatnumber` | 353 | 座位号 | 如"35A" |
| 22 | `seattype` | 148 | 座位类型 | 如"二等座" |
| 23 | `status` | 272 | 状态 | 如"已锁定""待支付" |
| 24 | `ticketgate` | 88 | 检票口 | 火车事件 |
| 25 | `time` | 157 | 时间（泛化） | ⚠️ 合并了 startTime/endTime |
| 26 | `validfrom` | 174 | 有效期起 | 身份证事件 |
| 27 | `validto` | 156 | 有效期止 | 身份证事件 |
| 28 | `vehiclenumber` | 334 | 车次/航班号 | 如"G9512""PN9432" |

**大小写变体**：W2NER 输出中 type 字段为小写（如 `orderapp`），但 `procnet_entities` 中 `type` 字段为驼峰（如 `orderApp`）。实际 sidecar 中 `type` 字段为驼峰格式。

---

## 2. ProcNet 事件 schema

**来源**：`procnet_format/mixed_data_with_queries/train.json`（3360 篇文档）

### Event: `flight`（航班）

| 角色 | 出现次数 | 语义说明 |
|------|---------|---------|
| `person` | 845 | 乘客姓名 |
| `departureStation` | 805 | 出发机场 |
| `arrivalStation` | 802 | 到达机场 |
| `seatNumber` | 729 | 座位号 |
| `vehicleNumber` | 720 | 航班号 |
| `startDate` | 718 | 出发日期 |
| `seatClass` | 716 | 舱位等级 |
| `name` | 704 | 航司名称 |
| `orderApp` | 698 | 订购平台 |
| `price` | 687 | 票价 |
| `status` | 611 | 订单状态 |
| `departureCity` | 551 | 出发城市 |
| `arrivalCity` | 548 | 到达城市 |
| `endDate` | 373 | 到达日期 |
| `startTime` | 206 | 出发时间 |
| `endTime` | 188 | 到达时间 |
| `orderNumber` | 161 | 订单号 |
| `idNumber` | 75 | 证件号 |

### Event: `hotel`（酒店）

| 角色 | 出现次数 | 语义说明 |
|------|---------|---------|
| `person` | 843 | 入住人姓名 |
| `roomType` | 721 | 房型 |
| `startDate` | 718 | 入住日期 |
| `name` | 715 | 酒店名称 |
| `price` | 714 | 房费 |
| `orderApp` | 713 | 订购平台 |
| `city` | 672 | 城市 |
| `address` | 661 | 酒店地址 |
| `endDate` | 651 | 退房日期 |
| `status` | 544 | 订单状态 |
| `orderNumber` | 142 | 订单号 |
| `idNumber` | 58 | 证件号 |

### Event: `id_card`（身份证）

| 角色 | 出现次数 | 语义说明 |
|------|---------|---------|
| `cardNumber` | 847 | 身份证号 |
| `person` | 846 | 姓名 |
| `validTo` | 819 | 有效期止 |
| `dateOfBirth` | 779 | 出生日期 |
| `gender` | 774 | 性别 |
| `ethnicGroup` | 764 | 民族 |
| `validFrom` | 754 | 有效期起 |
| `cardAddress` | 257 | 身份证地址 |

### Event: `train`（火车）

| 角色 | 出现次数 | 语义说明 |
|------|---------|---------|
| `person` | 823 | 乘客姓名 |
| `departureStation` | 717 | 出发站 |
| `orderApp` | 705 | 订购平台 |
| `startDate` | 705 | 出发日期 |
| `vehicleNumber` | 705 | 车次号 |
| `seatNumber` | 695 | 座位号 |
| `arrivalStation` | 694 | 到达站 |
| `seatType` | 692 | 座位类型 |
| `price` | 692 | 票价 |
| `ticketGate` | 646 | 检票口 |
| `status` | 544 | 订单状态 |
| `endDate` | 494 | 到达日期 |
| `arrivalCity` | 454 | 到达城市 |
| `departureCity` | 432 | 出发城市 |
| `startTime` | 218 | 出发时间 |
| `endTime` | 180 | 到达时间 |
| `orderNumber` | 134 | 订单号 |
| `idNumber` | 69 | 证件号 |

---

## 3. W2NER type → ProcNet field 映射表

**映射规则来源**：`scripts/convert_data_v1b_to_procnet.py:292` — `role_name = entity_type`（直接透传，无映射字典）

### 3.1 直接匹配（驼峰归一化）

| W2NER type | ProcNet field | 映射方式 | 风险 |
|-----------|--------------|---------|------|
| `address` | `address` | 直接同名 | 低 |
| `arrivalcity` | `arrivalCity` | 驼峰归一化 | 低 |
| `arrivalstation` | `arrivalStation` | 驼峰归一化 | 低 |
| `cardaddress` | `cardAddress` | 驼峰归一化 | 低 |
| `cardnumber` | `cardNumber` | 驼峰归一化 | 低 |
| `city` | `city` | 直接同名 | 低 |
| `dateofbirth` | `dateOfBirth` | 驼峰归一化 | 低 |
| `departurecity` | `departureCity` | 驼峰归一化 | 低 |
| `departurestation` | `departureStation` | 驼峰归一化 | 低 |
| `ethnicgroup` | `ethnicGroup` | 驼峰归一化 | 低 |
| `gender` | `gender` | 直接同名 | 低 |
| `idnumber` | `idNumber` | 驼峰归一化 | 低 |
| `name` | `name` | 直接同名 | 中（语义过宽） |
| `orderapp` | `orderApp` | 驼峰归一化 | 低 |
| `ordernumber` | `orderNumber` | 驼峰归一化 | 低 |
| `person` | `person` | 直接同名 | 低 |
| `price` | `price` | 直接同名 | 低 |
| `roomtype` | `roomType` | 驼峰归一化 | 低 |
| `seatclass` | `seatClass` | 驼峰归一化 | 低 |
| `seatnumber` | `seatNumber` | 驼峰归一化 | 低 |
| `seattype` | `seatType` | 驼峰归一化 | 低 |
| `status` | `status` | 直接同名 | 低 |
| `ticketgate` | `ticketGate` | 驼峰归一化 | 低 |
| `validfrom` | `validFrom` | 驼峰归一化 | 低 |
| `validto` | `validTo` | 驼峰归一化 | 低 |
| `vehiclenumber` | `vehicleNumber` | 驼峰归一化 | 低 |

### 3.2 无法直接匹配（语义合并/分裂）

| W2NER type | ProcNet field(s) | 映射方式 | 风险 | 说明 |
|-----------|-----------------|---------|------|------|
| `date` | `startDate` / `endDate` | **语义分裂** | 🔴 高 | W2NER 把所有日期合并为 `date`，但 ProcNet 区分出发日期(`startDate`)和到达日期(`endDate`) |
| `time` | `startTime` / `endTime` | **语义分裂** | 🔴 高 | W2NER 把所有时间合并为 `time`，但 ProcNet 区分出发时间(`startTime`)和到达时间(`endTime`) |

### 3.3 W2NER 缺失的 ProcNet 角色

| ProcNet field | 在 W2NER 中出现次数 | 说明 |
|--------------|-------------------|------|
| `person` | 726（有） | ✅ W2NER 能预测 |
| `startDate` | 0 | ❌ W2NER 用 `date` 替代 |
| `endDate` | 0 | ❌ W2NER 用 `date` 替代 |
| `startTime` | 0 | ❌ W2NER 用 `time` 替代 |
| `endTime` | 0 | ❌ W2NER 用 `time` 替代 |
| `departureCity` | 163（有） | ✅ W2NER 能预测 |
| `arrivalCity` | 7（极少） | ⚠️ W2NER 几乎不预测 |
| `ticketGate` | 88（有） | ✅ W2NER 能预测 |
| `cardAddress` | 20（极少） | ⚠️ W2NER 很少预测 |
| `idNumber` | 29（极少） | ⚠️ W2NER 很少预测 |
| `orderNumber` | 43（有） | ✅ W2NER 能预测 |
| `city` | 168（有） | ✅ W2NER 能预测 |
| `address` | 152（有） | ✅ W2NER 能预测 |

---

## 4. 转换脚本中的映射代码

### 4.1 实体类型 → 事件角色（无映射，直接透传）

**文件**：`scripts/convert_data_v1b_to_procnet.py:285-293`

```python
# 构建事件参数字典（保留所有类型）
event_dict = {}
for entity in entities:
    entity_text = entity.get('value', '')
    entity_type = entity.get('entity', 'unknown')

    if entity_text:
        # 使用实体类型作为角色名 ← 直接透传，无映射
        role_name = entity_type
        event_dict[role_name] = entity_text
```

**关键**：`role_name = entity_type` — W2NER 的 entity type 直接作为 ProcNet 的事件角色名，**没有任何映射字典或转换逻辑**。

### 4.2 复合 key 构建

**文件**：`scripts/convert_data_v1b_to_procnet.py:260-278`

```python
# 创建唯一 key：包含文本、位置和类型
unique_key = f"{entity_text}#{sent_idx}_{start}_{end}#{entity_type}"

# 添加到提及列表
ann_valid_mspans.append(entity_text)

# 添加位置
drange = [sent_idx, start, end]
ann_mspan2dranges[unique_key].append(drange)
ann_valid_dranges.append(drange)

# 添加实体类型映射
ann_mspan2guess_field[unique_key] = entity_type
```

### 4.3 事件列表构建

**文件**：`scripts/convert_data_v1b_to_procnet.py:296-299`

```python
# 格式：[[0, "事件类型", {"角色": "实体提及", ...}]]
recguid_eventname_eventdict_list = []
if event_dict:
    recguid_eventname_eventdict_list.append([0, event_type, event_dict])
```

其中 `event_type` 来自 `intent` 的映射（`event_type_mapping.get(intent, intent)`）。

---

## 5. 同文本多类型 / 多角色冲突统计

**数据来源**：`procnet_format/mixed_data_with_queries/train.json`

| 指标 | 数值 |
|------|------|
| 总文档数 | 3,360 |
| 存在同文本多类型的文档 | 912（27.1%） |
| 典型冲突模式 | `"泰安"` → `departureCity` + `departureStation` |
| 典型冲突模式 | `"12月19日"` → `startDate` + `endDate` |

**处理方式**：复合 key 设计（`文本#sent_b_e#类型`）天然区分了同文本不同类型，**不会冲突**。每个 (文本, 位置, 类型) 组合是独立的 entry。

**event_dict 中的处理**：`event_dict[role_name] = entity_text` — 不同角色名指向同一文本值，**不会覆盖**（key 是角色名，不是文本）。

---

## 6. 完整 event 样例

### 文档：`doc_000003`（火车事件）

**原始文本**：
```
【美团】胡美女士您预订的G9512次列车（泰安站→六盘水威箐）4月20日15:10发车已锁定，
二等座待支付970.74元。请30分钟内完成支付，转发无效。检票口2，登录APP处理订单详情。
麻烦尽快，逾期将自动取消。
```

**W2NER 预测实体**（8 个）：
| 文本 | 位置 | W2NER type |
|------|------|-----------|
| 美团 | [1:3] | orderApp |
| 胡美 | ❌ 未预测 | person |
| G9512 | [12:17] | vehicleNumber |
| 泰安站 | [21:24] | departureStation |
| 4月20日 | [31:36] | date |
| 15:10 | [36:41] | time |
| 二等座 | [47:50] | seatType |
| 已锁定 | [43:46] | status |
| 970.74 | [53:59] | price |

**Gold 实体**（13 个）：
| 文本 | 位置 | type |
|------|------|------|
| 胡美 | [4:6] | person |
| 美团 | [1:3] | orderApp |
| 二等座 | [47:50] | seatType |
| 检票口2 | [77:81] → sent[2][0:4] | ticketGate |
| 泰安 | [21:23] | departureCity |
| 泰安 | [21:23] | departureStation |
| 六盘水 | [25:28] | arrivalCity |
| 威箐 | [28:30] | arrivalStation |
| 4月20日 | [31:36] | startDate |
| 15:10 | [36:41] | endTime |
| G9512 | [12:17] | vehicleNumber |
| 待支付 | [50:53] | status |
| 970.74 | [53:59] | price |

**转换后的 ProcNet 文档**：
```json
{
  "sentences": ["【美团】胡美女士您预订的G9512次列车...", "请30分钟内...", "检票口2，登录APP...", "麻烦尽快..."],
  "ann_mspan2dranges": {
    "胡美#0_4_6#person": [[0, 4, 6]],
    "美团#0_1_3#orderApp": [[0, 1, 3]],
    ...
  },
  "ann_mspan2guess_field": {
    "胡美#0_4_6#person": "person",
    "美团#0_1_3#orderApp": "orderApp",
    ...
  },
  "recguid_eventname_eventdict_list": [
    [0, "train", {
      "person": "胡美",
      "orderApp": "美团",
      "seatType": "二等座",
      "ticketGate": "检票口2",
      "departureCity": "泰安",
      "departureStation": "泰安",
      "arrivalCity": "六盘水",
      "arrivalStation": "威箐",
      "startDate": "4月20日",
      "endTime": "15:10",
      "vehicleNumber": "G9512",
      "status": "待支付",
      "price": "970.74"
    }]
  ]
}
```

---

## 7. 核心风险总结

### 🔴 高风险

| 风险 | 说明 | 影响范围 |
|------|------|---------|
| `date` → `startDate`/`endDate` 语义分裂 | W2NER 把所有日期合并为 `date`，无法区分出发/到达日期 | 3128 个 `date` 实体 |
| `time` → `startTime`/`endTime` 语义分裂 | W2NER 把所有时间合并为 `time`，无法区分出发/到达时间 | 520 个 `time` 实体 |
| `person` 预测缺失 | W2NER 对 `person` 的 recall 极低（0 vs 3357 gold） | 全部文档 |
| `arrivalCity` 预测极少 | W2NER 只预测了 7 个，gold 有 1002 个 | 大部分文档 |

### 🟡 中风险

| 风险 | 说明 | 影响范围 |
|------|------|---------|
| `name` 语义过宽 | 可能是航司名、酒店名、公司名，缺乏区分 | 339 个实体 |
| 同文本多角色 | 同一文本对应多个角色（如"泰安"→departureCity+departureStation） | 912 篇文档 |
| 句子分割 bug | `?` 作为句子结束标点导致含 `?` 的实体丢失 | 至少 1 篇文档 |

### 🟢 低风险

| 风险 | 说明 | 影响范围 |
|------|------|---------|
| 驼峰归一化 | `orderapp` → `orderApp` 等 | 全量，但规则稳定 |
| 复合 key 解析 | 正则 `^(.*)#(\d+)_(\d+)_(\d+)#([^#]+)$` 在极端情况下可能错误 | 仅当 span 文本含 `#数字_数字_数字#` 时 |

---

## 8. 映射规则修订建议

### 当前状态
```python
role_name = entity_type  # 直接透传
```

### 建议修订
```python
# W2NER type → ProcNet field 映射字典
TYPE_MAPPING = {
    # 驼峰归一化（稳定映射）
    'orderapp': 'orderApp',
    'seatclass': 'seatClass',
    'seatnumber': 'seatNumber',
    'seattype': 'seatType',
    'vehiclenumber': 'vehicleNumber',
    'departurestation': 'departureStation',
    'arrivalstation': 'arrivalStation',
    'departurecity': 'departureCity',
    'arrivalcity': 'arrivalCity',
    'dateofbirth': 'dateOfBirth',
    'cardnumber': 'cardNumber',
    'cardaddress': 'cardAddress',
    'ordernumber': 'orderNumber',
    'ethnicgroup': 'ethnicGroup',
    'validfrom': 'validFrom',
    'validto': 'validTo',
    'idnumber': 'idNumber',
    'roomtype': 'roomType',
    'ticketgate': 'ticketGate',
    
    # 语义分裂（需要根据上下文区分）
    'date': None,      # 需要根据事件类型和位置判断是 startDate 还是 endDate
    'time': None,      # 需要根据事件类型和位置判断是 startTime 还是 endTime
    
    # 直接同名
    'person': 'person',
    'name': 'name',
    'price': 'price',
    'status': 'status',
    'address': 'address',
    'city': 'city',
    'gender': 'gender',
}
```

### 对于 `date` 和 `time` 的处理建议

由于 W2NER 无法区分 `startDate`/`endDate` 和 `startTime`/`endTime`，有两种策略：

1. **保守策略**：将 `date` 和 `time` 映射到 `startDate` 和 `startTime`（默认取第一个），接受部分错误
2. **放弃策略**：不将 `date` 和 `time` 作为事件角色输入 ProcNet，让 ProcNet 通过其他方式推断

---

## 9. 一句话总结

**位置索引耦合已成立，类型映射基本稳定（26/28 个类型可直接驼峰归一化），核心风险集中在 `date`/`time` 的语义分裂和 W2NER 对 `person` 等关键类型的预测缺失。**
