import logging
from typing import List, Dict, Any, Optional

from procnet.data_processor.basic_processor import BasicProcessor
from procnet.conf.global_config_manager import GlobalConfigManager
from procnet.data_example.DocEEexample import (
    DocEEDocumentExample,
    DocEEEntity,
    DocEELabel,
    PseudoDocEELabel,
)
from procnet.utils.util_data import UtilData


class DocEEProcessor(BasicProcessor):

    def __init__(
        self,
        read_pseudo_dataset: bool = False,
        use_procnet_pred_entities: bool = False,
        procnet_entity_field: str = "procnet_entities",
        procnet_type_id2field: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.data_path = GlobalConfigManager.get_dataset_path()

        if read_pseudo_dataset:
            self.data_path = GlobalConfigManager.get_pseudo_Doc2EDAG_path()

        self.use_procnet_pred_entities = use_procnet_pred_entities
        self.procnet_entity_field = procnet_entity_field
        self.procnet_type_id2field = procnet_type_id2field or {}

        logging.debug("Path: {}".format(self.data_path))
        self.train_path = self.data_path / "train.json"
        self.dev_path = self.data_path / "dev.json"
        self.test_path = self.data_path / "test.json"

        self.train_json = UtilData.read_raw_json_file(self.train_path)
        self.dev_json = UtilData.read_raw_json_file(self.dev_path)
        self.test_json = UtilData.read_raw_json_file(self.test_path)

        self.train_docs: List[DocEEDocumentExample] = self.parse_json_all(self.train_json)
        self.dev_docs: List[DocEEDocumentExample] = self.parse_json_all(self.dev_json)
        self.test_docs: List[DocEEDocumentExample] = self.parse_json_all(self.test_json)

        self.SCHEMA = DocEELabel.EVENT_SCHEMA
        self.SCHEMA_KEY_ENG_CHN = DocEELabel.KEY_ENG_CHN
        self.SCHEMA_KEY_CHN_ENG = DocEELabel.KEY_CHN_ENG

        if read_pseudo_dataset:
            self.SCHEMA = PseudoDocEELabel.EVENT_SCHEMA
            self.SCHEMA_KEY_ENG_CHN = PseudoDocEELabel.KEY_ENG_CHN
            self.SCHEMA_KEY_CHN_ENG = PseudoDocEELabel.KEY_CHN_ENG

    # ----------------------------
    # New helpers for procnet_entities
    # ----------------------------
    def _safe_extract_span(self, sentences: List[str], sent_idx: int, b: int, e: int) -> Optional[str]:
        if sent_idx < 0 or sent_idx >= len(sentences):
            return None
        sent = sentences[sent_idx]
        if b < 0 or e > len(sent) or b >= e:
            return None
        return sent[b:e]

    def _resolve_positions(self, ent: Dict[str, Any]) -> Optional[List[List[int]]]:
        """
        Accept multiple input styles:
        1) ent["positions"] = [[sent_idx, b, e], ...]
        2) ent["drange"] = [sent_idx, b, e]
        3) ent["sent_idx"], ent["b"], ent["e"]
        """
        if ent.get("positions") is not None:
            return ent["positions"]

        if ent.get("drange") is not None:
            drange = ent["drange"]
            return [drange]

        sent_idx = ent.get("sent_idx", ent.get("sentence_idx"))
        b = ent.get("b")
        e = ent.get("e")
        if sent_idx is not None and b is not None and e is not None:
            return [[int(sent_idx), int(b), int(e)]]

        return None

    def _normalize_procnet_entity(self, ent: Dict[str, Any], sentences: List[str]) -> Optional[Dict[str, Any]]:
        positions = self._resolve_positions(ent)
        if not positions:
            return None

        # Phase 1: only continuous entity; take the first mention
        sent_idx, b, e = positions[0]
        sent_idx, b, e = int(sent_idx), int(b), int(e)

        type_id = ent.get("type_id", ent.get("label_id", -1))
        type_id = int(type_id) if type_id is not None else -1

        field = ent.get("field", ent.get("type"))
        if field is None and type_id in self.procnet_type_id2field:
            field = self.procnet_type_id2field[type_id]

        span = ent.get("span")
        if span is None:
            span = self._safe_extract_span(sentences, sent_idx, b, e)

        cluster_key = ent.get("key", [b, e, type_id])  # keep your phase-1 convention
        global_key = ent.get("global_key", [sent_idx, b, e, type_id])  # avoid cross-sentence collision

        norm_ent = {
            "key": cluster_key,
            "cluster_key": cluster_key,
            "global_key": global_key,
            "token_indices": ent.get("token_indices", list(range(b, e))),
            "b": b,
            "e": e,
            "type_id": type_id,
            "field": field,        # optional string label
            "score": float(ent.get("score", 1.0)),
            "head": ent.get("head", None),
            "span": span,
            "positions": [[sent_idx, b, e]],   # ProcNet-compatible drange style
        }
        return norm_ent

    def _parse_procnet_entities(self, data: Dict[str, Any], sentences: List[str]) -> List[Dict[str, Any]]:
        raw_entities = data.get(self.procnet_entity_field, [])
        res = []
        for ent in raw_entities:
            one = self._normalize_procnet_entity(ent, sentences)
            if one is not None:
                res.append(one)

        res.sort(key=lambda x: (
            x["positions"][0][0],
            x["positions"][0][1],
            x["positions"][0][2],
            x["type_id"],
        ))
        return res

    # ----------------------------
    # Original logic + sidecar attach
    # ----------------------------
    def parse_json_one(self, json) -> DocEEDocumentExample:
        doc_id: str = json[0]
        data = json[1]

        sentences: List[str] = data["sentences"]
        ann_mspan2dranges: Dict[str, List[list]] = data["ann_mspan2dranges"]
        ann_mspan2guess_field: Dict[str, str] = data["ann_mspan2guess_field"]
        recguid_eventname_eventdict_list = data["recguid_eventname_eventdict_list"]

        assert len(ann_mspan2dranges) == len(ann_mspan2guess_field)

        # original gold entities
        entities = []
        for k, v in ann_mspan2dranges.items():
            entity = DocEEEntity(span=k, positions=v, field=ann_mspan2guess_field[k])
            entities.append(entity)

        # original gold events
        events = []
        for x in recguid_eventname_eventdict_list:
            event = {"EventType": x[1]}
            for k, v in x[2].items():
                event[k] = v
            events.append(event)

        doc = DocEEDocumentExample(
            doc_id=doc_id,
            sentences=sentences,
            entities=entities,
            events=events,
        )

        # keep original assertion
        for entity in doc.entities:
            for pos in entity.positions:
                assert entity.span == doc.sentences[pos[0]][pos[1]:pos[2]]

        # ---- new sidecar attach ----
        if self.use_procnet_pred_entities:
            doc.procnet_entities = self._parse_procnet_entities(data, sentences)
        else:
            doc.procnet_entities = []

        doc.has_procnet_entities = len(doc.procnet_entities) > 0
        doc.procnet_entity_source = "w2ner" if doc.has_procnet_entities else "gold_only"

        return doc

    def parse_json_all(self, json) -> List[DocEEDocumentExample]:
        docs = []
        for one in json:
            doc = self.parse_json_one(one)
            docs.append(doc)
        return docs
