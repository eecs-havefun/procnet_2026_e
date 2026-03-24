from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy


class DocEELabel:
    EVENT_TYPE = ["EquityFreeze", "EquityRepurchase", "EquityUnderweight", "EquityOverweight", "EquityPledge"]
    KEY_ENG_CHN = {"EventType": "事件",
                   "EquityFreeze": "股权冻结",
                   "EquityRepurchase": "股权回购",
                   "EquityUnderweight": "股票减持",
                   "EquityOverweight": "股票增持",
                   "EquityPledge": "股权质押",
                   "Pledger": "质押者",
                   "PledgedShares": "质押股份",
                   "Pledgee": "质权人",
                   "TotalHoldingShares": "总持股",
                   "TotalHoldingRatio": "总持股比率",
                   "TotalPledgedShares": "总质押股份",
                   "StartDate": "开始日期",
                   "EndDate": "结束日期",
                   "ReleasedDate": "发布日期",
                   "EquityHolder": "股权持有人",
                   "TradedShares": "交易股票",
                   "LaterHoldingShares": "后来控股股份",
                   "AveragePrice": "平均价格",
                   "CompanyName": "公司名",
                   "HighestTradingPrice": "最高交易价",
                   "LowestTradingPrice": "最低交易价",
                   "RepurchasedShares": "回购股份",
                   "ClosingDate": "截止日期",
                   "RepurchaseAmount": "回购金额",
                   "FrozeShares": "冻结股份",
                   "LegalInstitution": "法律机构",
                   "UnfrozeDate": "解冻日期",
                   "Null": "无"
                   }
    KEY_CHN_ENG = {v: k for k, v in KEY_ENG_CHN.items()}
    EVENT_SCHEMA = {
        "EquityFreeze": [
            "EquityHolder",
            "FrozeShares",
            "LegalInstitution",
            "TotalHoldingShares",
            "TotalHoldingRatio",
            "StartDate",
            "EndDate",
            "UnfrozeDate",
        ],
        "EquityRepurchase": [
            "CompanyName",
            "HighestTradingPrice",
            "LowestTradingPrice",
            "RepurchasedShares",
            "ClosingDate",
            "RepurchaseAmount",
        ],
        "EquityUnderweight": [
            "EquityHolder",
            "TradedShares",
            "StartDate",
            "EndDate",
            "LaterHoldingShares",
            "AveragePrice",
        ],
        "EquityOverweight": [
            "EquityHolder",
            "TradedShares",
            "StartDate",
            "EndDate",
            "LaterHoldingShares",
            "AveragePrice",
        ],
        "EquityPledge": [
            "Pledger",
            "PledgedShares",
            "Pledgee",
            "TotalHoldingShares",
            "TotalHoldingRatio",
            "TotalPledgedShares",
            "StartDate",
            "EndDate",
            "ReleasedDate",
        ],
    }


class PseudoDocEELabel:
    EVENT_SCHEMA = {
        '解除质押': ['质押方', '披露时间', '质权方', '质押物', '质押股票/股份数量', '事件时间', '质押物所属公司', '质押物占总股比', '质押物占持股比'],
        '股份回购': ['回购方', '披露时间', '回购股份数量', '每股交易价格', '占公司总股本比例', '交易金额', '回购完成时间'],
        '股东减持': ['股票简称', '披露时间', '交易股票/股份数量', '每股交易价格', '交易金额', '交易完成时间', '减持方', '减持部分占所持比例', '减持部分占总股本比例'],
        '亏损': ['公司名称', '披露时间', '财报周期', '净亏损', '亏损变化'],
        '中标': ['中标公司', '中标标的', '中标金额', '招标方', '中标日期', '披露日期'],
        '高管变动': ['高管姓名', '任职公司', '高管职位', '事件时间', '变动类型', '披露日期', '变动后职位', '变动后公司名称'],
        '企业破产': ['破产公司', '披露时间', '债务规模', '破产时间', '债权人'],
        '股东增持': ['股票简称', '披露时间', '交易股票/股份数量', '每股交易价格', '交易金额', '交易完成时间', '增持方', '增持部分占所持比例', '增持部分占总股本比例'],
        '被约谈': ['公司名称', '披露时间', '被约谈时间', '约谈机构'],
        '企业收购': ['收购方', '披露时间', '被收购方', '收购标的', '交易金额', '收购完成时间'],
        '公司上市': ['上市公司', '证券代码', '环节', '披露时间', '发行价格', '事件时间', '市值', '募资金额'],
        '企业融资': ['投资方', '披露时间', '被投资方', '融资金额', '融资轮次', '事件时间', '领投方'],
        '质押': ['质押方', '披露时间', '质权方', '质押物', '质押股票/股份数量', '事件时间', '质押物占总股比', '质押物所属公司', '质押物占持股比']
    }
    EVENT_TYPE = EVENT_SCHEMA.keys()
    KEY_ENG_CHN = {"EventType": "事件", "Null": "无"}
    for k, vs in EVENT_SCHEMA.items():
        for v in vs:
            KEY_ENG_CHN[v] = v
    KEY_CHN_ENG = {v: k for k, v in KEY_ENG_CHN.items()}


@dataclass
class DocEEEntity:
    span: str
    positions: List[list]  # [[sentence_index, start, end]]
    field: str


@dataclass
class DocEETypedEntity:
    """
    Disk-layer typed entity + ProcNet compatibility cache.

    约定：
    - sent_id / b / e / token_indices：句内坐标
    - doc_token_start / doc_token_end：文档级 token offset（如果已有）
    - procnet_span_key：后续给 ProcNet 后半段做对齐的兼容 key
    - fragment_sent_id / fragment_token_start / fragment_token_end：
      仅在 get_fragment() 后填充，供子片段编码时使用
    """
    doc_id: str
    sent_id: int
    b: int
    e: int
    type_id: Optional[int] = None
    type_name: Optional[str] = None
    token_indices: List[int] = field(default_factory=list)
    head: Optional[int] = None
    score: Optional[float] = None
    text: str = ""
    source: str = "w2ner"
    key: Optional[str] = None
    cluster_key: Optional[str] = None
    w2ner_key: Optional[str] = None
    procnet_span_key: Optional[Any] = None
    doc_token_start: Optional[int] = None
    doc_token_end: Optional[int] = None
    raw_w2ner: Optional[Dict[str, Any]] = None

    # fragment-time cache
    fragment_sent_id: Optional[int] = None
    fragment_token_start: Optional[int] = None
    fragment_token_end: Optional[int] = None

    def __post_init__(self):
        if not self.token_indices:
            self.token_indices = list(range(self.b, self.e))
        if self.head is None:
            self.head = self.token_indices[0] if self.token_indices else self.b
        if self.key is None:
            self.key = self.build_default_key()
        if self.w2ner_key is None:
            self.w2ner_key = self.key
        if self.procnet_span_key is None:
            self.procnet_span_key = (self.sent_id, self.b, self.e)
        if self.cluster_key is None:
            self.cluster_key = self.key
        self.validate()

    def build_default_key(self) -> str:
        type_piece = "None" if self.type_id is None else str(self.type_id)
        return f"{self.doc_id}:{self.sent_id}:{self.b}:{self.e}:{type_piece}"

    def validate(self):
        if self.b < 0 or self.e < self.b:
            raise ValueError(f"Invalid span [{self.b}, {self.e}) for doc {self.doc_id}")
        if self.token_indices:
            expected = list(range(self.b, self.e))
            if self.token_indices != expected:
                raise ValueError(
                    f"token_indices must match [b, e); got {self.token_indices}, expected {expected}"
                )
        if self.head is not None and self.token_indices and self.head not in self.token_indices:
            raise ValueError(f"head={self.head} not in token_indices={self.token_indices}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocEETypedEntity":
        payload = dict(data)
        if "type" in payload and "type_name" not in payload:
            payload["type_name"] = payload.pop("type")
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "sent_id": self.sent_id,
            "b": self.b,
            "e": self.e,
            "type_id": self.type_id,
            "type_name": self.type_name,
            "token_indices": list(self.token_indices),
            "head": self.head,
            "score": self.score,
            "text": self.text,
            "source": self.source,
            "key": self.key,
            "cluster_key": self.cluster_key,
            "w2ner_key": self.w2ner_key,
            "procnet_span_key": self.procnet_span_key,
            "doc_token_start": self.doc_token_start,
            "doc_token_end": self.doc_token_end,
            "raw_w2ner": self.raw_w2ner,
            "fragment_sent_id": self.fragment_sent_id,
            "fragment_token_start": self.fragment_token_start,
            "fragment_token_end": self.fragment_token_end,
        }

    def to_procnet_node(self, node_id: int) -> Dict[str, Any]:
        return {
            "node_id": node_id,
            "doc_id": self.doc_id,
            "sent_id": self.sent_id,
            "fragment_sent_id": self.fragment_sent_id,
            "w2ner_key": self.w2ner_key,
            "key": self.key,
            "cluster_key": self.cluster_key,
            "procnet_span_key": self.procnet_span_key,
            "b": self.b,
            "e": self.e,
            "token_indices": list(self.token_indices),
            "head": self.head,
            "type_id": self.type_id,
            "type_name": self.type_name,
            "score": self.score,
            "text": self.text,
            "source": self.source,
            "doc_token_start": self.doc_token_start,
            "doc_token_end": self.doc_token_end,
            "fragment_token_start": self.fragment_token_start,
            "fragment_token_end": self.fragment_token_end,
            "raw_w2ner": self.raw_w2ner,
        }


@dataclass
class DocEEDocumentExample:
    doc_id: str
    sentences: List[str]
    entities: List[DocEEEntity]
    events: List[Dict[str, str]]  # must have a key named EventType
    sentences_token: Optional[List[List[str]]] = None
    seq_BIO_tags: Optional[List[List[str]]] = None

    # W2NER -> ProcNet adapter fields
    typed_entities: List[DocEETypedEntity] = field(default_factory=list)
    entity_nodes: List[Dict[str, Any]] = field(default_factory=list)
    node_types: List[int] = field(default_factory=list)
    node_spans: List[Tuple[int, int, int]] = field(default_factory=list)  # (sent_id, b, e)
    num_nodes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        normalized_typed_entities = []
        for entity in self.typed_entities:
            if isinstance(entity, DocEETypedEntity):
                normalized_typed_entities.append(entity)
            else:
                normalized_typed_entities.append(DocEETypedEntity.from_dict(entity))
        self.typed_entities = normalized_typed_entities

        if self.entity_nodes or self.node_types or self.node_spans or self.num_nodes:
            self.num_nodes = len(self.entity_nodes)
        elif self.typed_entities:
            self.refresh_entity_node_cache()

    def copy(self):
        return copy.deepcopy(self)

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    def refresh_entity_node_cache(self):
        self.entity_nodes = [entity.to_procnet_node(node_id=i) for i, entity in enumerate(self.typed_entities)]
        self.node_types = [entity.type_id for entity in self.typed_entities]
        self.node_spans = [(entity.sent_id, entity.b, entity.e) for entity in self.typed_entities]
        self.num_nodes = len(self.typed_entities)

    def _filter_legacy_entities_for_fragment(self, start_sen: int, end_sen: int) -> List[DocEEEntity]:
        fragment_entities = []
        for entity in self.entities:
            new_positions = []
            for sent_id, start, end in entity.positions:
                if start_sen <= sent_id < end_sen:
                    new_positions.append([sent_id - start_sen, start, end])
            if new_positions:
                fragment_entities.append(
                    DocEEEntity(span=entity.span, positions=new_positions, field=entity.field)
                )
        return fragment_entities

    def _calc_fragment_token_offsets(self, start_sen: int, end_sen: int) -> Dict[int, int]:
        if self.sentences_token is None:
            return {}
        offset = 1  # reserve [CLS]
        mapping = {}
        for sent_id in range(start_sen, end_sen):
            mapping[sent_id] = offset
            offset += len(self.sentences_token[sent_id])
        return mapping

    def _filter_typed_entities_for_fragment(self, start_sen: int, end_sen: int) -> List[DocEETypedEntity]:
        token_offsets = self._calc_fragment_token_offsets(start_sen=start_sen, end_sen=end_sen)
        fragment_entities = []
        for entity in self.typed_entities:
            if start_sen <= entity.sent_id < end_sen:
                new_entity = copy.deepcopy(entity)
                new_entity.fragment_sent_id = entity.sent_id - start_sen
                if token_offsets:
                    sent_offset = token_offsets[entity.sent_id]
                    new_entity.fragment_token_start = sent_offset + entity.b
                    new_entity.fragment_token_end = sent_offset + entity.e
                fragment_entities.append(new_entity)
        return fragment_entities

    def get_fragment(self, start_sen: int, end_sen: int):
        new_example = self.copy()
        new_example.sentences = self.sentences[start_sen: end_sen]
        new_example.sentences_token = (
            self.sentences_token[start_sen: end_sen] if self.sentences_token is not None else None
        )
        new_example.seq_BIO_tags = (
            self.seq_BIO_tags[start_sen: end_sen] if self.seq_BIO_tags is not None else None
        )

        # 保持 legacy entity 与切分后的句子索引一致
        new_example.entities = self._filter_legacy_entities_for_fragment(
            start_sen=start_sen,
            end_sen=end_sen,
        )

        # 保持 typed_entities 在 fragment 中可直接使用
        new_example.typed_entities = self._filter_typed_entities_for_fragment(
            start_sen=start_sen,
            end_sen=end_sen,
        )
        new_example.refresh_entity_node_cache()

        new_example.extra = copy.deepcopy(self.extra)
        new_example.extra["fragment_start_sen"] = start_sen
        new_example.extra["fragment_end_sen"] = end_sen
        return new_example
