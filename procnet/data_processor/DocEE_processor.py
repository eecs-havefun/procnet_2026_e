import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from procnet.data_processor.basic_processor import BasicProcessor
from procnet.conf.global_config_manager import GlobalConfigManager
from procnet.data_example.DocEEexample import (
    DocEEDocumentExample,
    DocEEEntity,
    DocEELabel,
    PseudoDocEELabel,
    DocEETypedEntity,
)
from procnet.utils.util_data import UtilData


class DocEEProcessor(BasicProcessor):
    def __init__(
        self,
        read_pseudo_dataset: bool = False,
        use_procnet_pred_entities: bool = False,
        procnet_entity_field: str = "procnet_entities",
        typed_entity_field: str = "typed_entities",
        procnet_type_id2field: Optional[Dict[int, str]] = None,
        typed_entities_dir: Optional[Union[str, Path]] = None,
        typed_entities_files: Optional[Dict[str, Union[str, Path]]] = None,
        sidecar_strict_doc_match: bool = False,
    ):
        super().__init__()
        self.data_path = GlobalConfigManager.get_dataset_path()
        if read_pseudo_dataset:
            self.data_path = GlobalConfigManager.get_pseudo_Doc2EDAG_path()

        self.use_procnet_pred_entities = use_procnet_pred_entities
        self.procnet_entity_field = procnet_entity_field
        self.typed_entity_field = typed_entity_field
        self.procnet_type_id2field = procnet_type_id2field or {}
        self.sidecar_strict_doc_match = sidecar_strict_doc_match

        logging.debug("Path: {}".format(self.data_path))
        self.train_path = self.data_path / "train.json"
        self.dev_path = self.data_path / "dev.json"
        self.test_path = self.data_path / "test.json"

        self.train_json = UtilData.read_raw_json_file(self.train_path)
        self.dev_json = UtilData.read_raw_json_file(self.dev_path)
        self.test_json = UtilData.read_raw_json_file(self.test_path)

        self.sidecar_paths = self._resolve_sidecar_paths(
            typed_entities_dir=typed_entities_dir,
            typed_entities_files=typed_entities_files,
        )
        self.sidecar_by_split = self._load_all_sidecars() if self.sidecar_paths else {}

        self.train_docs: List[DocEEDocumentExample] = self.parse_json_all(self.train_json, split_name="train")
        self.dev_docs: List[DocEEDocumentExample] = self.parse_json_all(self.dev_json, split_name="dev")
        self.test_docs: List[DocEEDocumentExample] = self.parse_json_all(self.test_json, split_name="test")

        self.SCHEMA = DocEELabel.EVENT_SCHEMA
        self.SCHEMA_KEY_ENG_CHN = DocEELabel.KEY_ENG_CHN
        self.SCHEMA_KEY_CHN_ENG = DocEELabel.KEY_CHN_ENG

        if read_pseudo_dataset:
            self.SCHEMA = PseudoDocEELabel.EVENT_SCHEMA
            self.SCHEMA_KEY_ENG_CHN = PseudoDocEELabel.KEY_ENG_CHN
            self.SCHEMA_KEY_CHN_ENG = PseudoDocEELabel.KEY_CHN_ENG

    # -------------------------------------------------------------------------
    # sidecar path / loading
    # -------------------------------------------------------------------------

    def _resolve_sidecar_paths(
        self,
        typed_entities_dir: Optional[Union[str, Path]],
        typed_entities_files: Optional[Dict[str, Union[str, Path]]],
    ) -> Dict[str, Path]:
        """
        支持两种传法：
        1) typed_entities_dir:
           目录下默认寻找：
             train_doc_typed_entities.jsonl
             dev_doc_typed_entities.jsonl
             test_doc_typed_entities.jsonl
           若不存在，再尝试：
             train.jsonl / dev.jsonl / test.jsonl
             train_typed_entities.jsonl / dev_typed_entities.jsonl / test_typed_entities.jsonl

        2) typed_entities_files:
           {"train": "...", "dev": "...", "test": "..."}
        """
        result: Dict[str, Path] = {}

        if typed_entities_files:
            for split_name, file_path in typed_entities_files.items():
                if file_path is None:
                    continue
                result[split_name] = Path(file_path)

        if typed_entities_dir is not None:
            base_dir = Path(typed_entities_dir)
            candidates = {
                "train": [
                    "train_doc_typed_entities.jsonl",
                    "train_typed_entities.jsonl",
                    "train.jsonl",
                ],
                "dev": [
                    "dev_doc_typed_entities.jsonl",
                    "dev_typed_entities.jsonl",
                    "dev.jsonl",
                ],
                "test": [
                    "test_doc_typed_entities.jsonl",
                    "test_typed_entities.jsonl",
                    "test.jsonl",
                ],
            }
            for split_name, names in candidates.items():
                if split_name in result:
                    continue
                for name in names:
                    p = base_dir / name
                    if p.exists():
                        result[split_name] = p
                        break

        existing = {}
        for split_name, p in result.items():
            if p.exists():
                existing[split_name] = p
            else:
                logging.warning("Typed entity sidecar for split=%s not found: %s", split_name, str(p))
        return existing

    def _load_all_sidecars(self) -> Dict[str, Dict[str, List[DocEETypedEntity]]]:
        loaded = {}
        for split_name, path in self.sidecar_paths.items():
            loaded[split_name] = self._load_typed_entity_jsonl(path)
            logging.info(
                "Loaded typed-entity sidecar for split=%s: %s docs from %s",
                split_name,
                len(loaded[split_name]),
                str(path),
            )
        return loaded

    def _load_typed_entity_jsonl(self, path: Path) -> Dict[str, List[DocEETypedEntity]]:
        """
        同时支持：
        A. doc-level:
           {"doc_id": "...", "typed_entities": [...]}
           {"doc_id": "...", "procnet_entities": [...]}

        B. sentence-level:
           {"doc_id": "...", "sent_id": 0, "sentence": "...", "procnet_entities": [...]}
           {"doc_id": "...", "sent_id": 0, "sentence": "...", "typed_entities": [...]}
        """
        doc_map: Dict[str, List[DocEETypedEntity]] = {}

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid json line in {path} line {line_idx}: {e}")

                doc_id = record.get("doc_id")
                if doc_id is None:
                    raise ValueError(f"Missing doc_id in {path} line {line_idx}")

                typed_entities = self._extract_typed_entities_from_sidecar_record(record)
                if not typed_entities:
                    doc_map.setdefault(doc_id, [])
                    continue

                bucket = doc_map.setdefault(doc_id, [])
                bucket.extend(typed_entities)

        for doc_id, entities in doc_map.items():
            entities.sort(key=lambda x: (x.sent_id, x.b, x.e, -1 if x.type_id is None else x.type_id, x.key))
            self._deduplicate_typed_entities_inplace(doc_id=doc_id, entities=entities)

        return doc_map

    def _extract_typed_entities_from_sidecar_record(self, record: Dict[str, Any]) -> List[DocEETypedEntity]:
        """
        record 可能是：
        1) doc-level，含 typed_entities / procnet_entities
        2) sentence-level，含 sent_id 且 typed_entities / procnet_entities 只对应当前句
        """
        raw_entities = record.get(self.typed_entity_field)
        if raw_entities is None:
            raw_entities = record.get(self.procnet_entity_field, [])

        if raw_entities is None:
            raw_entities = []

        doc_id = record["doc_id"]
        default_sent_id = record.get("sent_id", record.get("sentence_idx"))
        sentence_text = record.get("sentence")

        result = []
        for ent in raw_entities:
            norm = self._normalize_sidecar_entity(
                ent=ent,
                doc_id=doc_id,
                default_sent_id=default_sent_id,
                sentence_text=sentence_text,
            )
            if norm is not None:
                result.append(norm)
        return result

    def _deduplicate_typed_entities_inplace(self, doc_id: str, entities: List[DocEETypedEntity]):
        seen = set()
        deduped = []
        for ent in entities:
            uniq_key = (ent.key, ent.w2ner_key, ent.procnet_span_key)
            if uniq_key in seen:
                continue
            seen.add(uniq_key)
            deduped.append(ent)
        if len(deduped) != len(entities):
            logging.warning(
                "Deduplicated typed entities for doc_id=%s: %d -> %d",
                doc_id,
                len(entities),
                len(deduped),
            )
        entities[:] = deduped

    # -------------------------------------------------------------------------
    # helpers for span / type normalization
    # -------------------------------------------------------------------------

    def _safe_extract_span(self, sentences: List[str], sent_idx: int, b: int, e: int) -> Optional[str]:
        if sent_idx < 0 or sent_idx >= len(sentences):
            return None
        sent = sentences[sent_idx]
        if b < 0 or e > len(sent) or b >= e:
            return None
        return sent[b:e]

    def _safe_extract_span_from_sentence(self, sentence_text: Optional[str], b: int, e: int) -> Optional[str]:
        if sentence_text is None:
            return None
        if b < 0 or e > len(sentence_text) or b >= e:
            return None
        return sentence_text[b:e]

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

        sent_idx = ent.get("sent_idx", ent.get("sentence_idx", ent.get("sent_id")))
        b = ent.get("b")
        e = ent.get("e")
        if sent_idx is not None and b is not None and e is not None:
            return [[int(sent_idx), int(b), int(e)]]
        return None

    def _resolve_type_name(self, ent: Dict[str, Any], type_id: Optional[int]) -> Optional[str]:
        type_name = ent.get("type_name", ent.get("type", ent.get("field")))
        if type_name is None and type_id is not None and type_id in self.procnet_type_id2field:
            type_name = self.procnet_type_id2field[type_id]
        return type_name

    def _normalize_sidecar_entity(
        self,
        ent: Dict[str, Any],
        doc_id: str,
        default_sent_id: Optional[int] = None,
        sentence_text: Optional[str] = None,
    ) -> Optional[DocEETypedEntity]:
        """
        标准化成 DocEETypedEntity。
        兼容字段：
        - sent_id / sent_idx / sentence_idx
        - b / e
        - token_indices
        - type_id / label_id
        - type / type_name / field
        - key / cluster_key / w2ner_key / procnet_span_key
        - doc_token_start / doc_token_end
        - raw_w2ner
        """
        sent_id = ent.get("sent_id", ent.get("sent_idx", ent.get("sentence_idx", default_sent_id)))
        b = ent.get("b")
        e = ent.get("e")

        if sent_id is None or b is None or e is None:
            positions = self._resolve_positions(ent)
            if not positions:
                return None
            sent_id, b, e = positions[0]

        sent_id = int(sent_id)
        b = int(b)
        e = int(e)

        token_indices = ent.get("token_indices")
        if token_indices is None:
            token_indices = list(range(b, e))

        type_id = ent.get("type_id", ent.get("label_id"))
        type_id = int(type_id) if type_id is not None else None
        type_name = self._resolve_type_name(ent, type_id)

        text = ent.get("text", ent.get("span"))
        if text is None:
            text = self._safe_extract_span_from_sentence(sentence_text, b, e)

        key = ent.get("key")
        cluster_key = ent.get("cluster_key", key)
        w2ner_key = ent.get("w2ner_key", key)

        procnet_span_key = ent.get("procnet_span_key")
        if procnet_span_key is None:
            procnet_span_key = (sent_id, b, e)

        raw_w2ner = ent.get("raw_w2ner")
        if raw_w2ner is None and ent.get("source", "w2ner") == "w2ner":
            raw_w2ner = dict(ent)

        return DocEETypedEntity(
            doc_id=doc_id,
            sent_id=sent_id,
            b=b,
            e=e,
            type_id=type_id,
            type_name=type_name,
            token_indices=list(token_indices),
            head=ent.get("head"),
            score=float(ent.get("score", 1.0)) if ent.get("score") is not None else None,
            text=text or "",
            source=ent.get("source", "w2ner"),
            key=key,
            cluster_key=cluster_key,
            w2ner_key=w2ner_key,
            procnet_span_key=procnet_span_key,
            doc_token_start=ent.get("doc_token_start"),
            doc_token_end=ent.get("doc_token_end"),
            raw_w2ner=raw_w2ner,
        )

    # -------------------------------------------------------------------------
    # old procnet helper (kept for backward compatibility)
    # -------------------------------------------------------------------------

    def _normalize_procnet_entity(self, ent: Dict[str, Any], sentences: List[str]) -> Optional[Dict[str, Any]]:
        positions = self._resolve_positions(ent)
        if not positions:
            return None

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

        cluster_key = ent.get("key", [b, e, type_id])
        global_key = ent.get("global_key", [sent_idx, b, e, type_id])

        norm_ent = {
            "key": cluster_key,
            "cluster_key": cluster_key,
            "global_key": global_key,
            "token_indices": ent.get("token_indices", list(range(b, e))),
            "b": b,
            "e": e,
            "type_id": type_id,
            "field": field,
            "score": float(ent.get("score", 1.0)),
            "head": ent.get("head", None),
            "span": span,
            "positions": [[sent_idx, b, e]],
        }
        return norm_ent

    def _parse_procnet_entities(self, data: Dict[str, Any], sentences: List[str]) -> List[Dict[str, Any]]:
        raw_entities = data.get(self.procnet_entity_field, [])
        res = []
        for ent in raw_entities:
            one = self._normalize_procnet_entity(ent, sentences)
            if one is not None:
                res.append(one)
        res.sort(
            key=lambda x: (
                x["positions"][0][0],
                x["positions"][0][1],
                x["positions"][0][2],
                x["type_id"],
            )
        )
        return res

    # -------------------------------------------------------------------------
    # parsing
    # -------------------------------------------------------------------------

    def _attach_typed_entities_from_sidecar(
        self,
        doc: DocEEDocumentExample,
        split_name: Optional[str] = None,
    ) -> None:
        typed_entities = []
        if split_name is not None and split_name in self.sidecar_by_split:
            typed_entities = self.sidecar_by_split[split_name].get(doc.doc_id, [])

        if self.sidecar_strict_doc_match and split_name in self.sidecar_by_split:
            if doc.doc_id not in self.sidecar_by_split[split_name]:
                raise KeyError(
                    f"Missing typed-entity sidecar for split={split_name}, doc_id={doc.doc_id}"
                )

        doc.typed_entities = list(typed_entities)
        doc.refresh_entity_node_cache()

        doc.has_typed_entities = len(doc.typed_entities) > 0
        doc.typed_entity_source = "sidecar_jsonl" if doc.has_typed_entities else "none"

    def parse_json_one(self, json_item, split_name: Optional[str] = None) -> DocEEDocumentExample:
        doc_id: str = json_item[0]
        data = json_item[1]
        sentences: List[str] = data["sentences"]
        ann_mspan2dranges: Dict[str, List[list]] = data["ann_mspan2dranges"]
        ann_mspan2guess_field: Dict[str, str] = data["ann_mspan2guess_field"]
        recguid_eventname_eventdict_list = data["recguid_eventname_eventdict_list"]

        assert len(ann_mspan2dranges) == len(ann_mspan2guess_field)

        entities = []
        for k, v in ann_mspan2dranges.items():
            entity = DocEEEntity(span=k, positions=v, field=ann_mspan2guess_field[k])
            entities.append(entity)

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

        for entity in doc.entities:
            for pos in entity.positions:
                assert entity.span == doc.sentences[pos[0]][pos[1]:pos[2]]

        # old in-doc procnet_entities helper
        if self.use_procnet_pred_entities:
            doc.procnet_entities = self._parse_procnet_entities(data, sentences)
        else:
            doc.procnet_entities = []
        doc.has_procnet_entities = len(doc.procnet_entities) > 0
        doc.procnet_entity_source = "inline_json" if doc.has_procnet_entities else "gold_only"

        # new sidecar typed entities
        self._attach_typed_entities_from_sidecar(doc=doc, split_name=split_name)

        return doc

    def parse_json_all(self, json_data, split_name: Optional[str] = None) -> List[DocEEDocumentExample]:
        docs = []
        for one in json_data:
            doc = self.parse_json_one(one, split_name=split_name)
            docs.append(doc)
        return docs
