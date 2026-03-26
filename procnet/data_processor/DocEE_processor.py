import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from procnet.conf.global_config_manager import GlobalConfigManager
from procnet.data_example.DocEEexample import (
    DocEEDocumentExample,
    DocEEEntity,
    DocEELabel,
    DocEETypedEntity,
    PseudoDocEELabel,
)
from procnet.data_processor.basic_processor import BasicProcessor
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
        dataset_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__()

        self.data_path = Path(dataset_dir) if dataset_dir is not None else GlobalConfigManager.get_dataset_path()
        if dataset_dir is None and read_pseudo_dataset:
            self.data_path = GlobalConfigManager.get_pseudo_Doc2EDAG_path()

        self.use_procnet_pred_entities = use_procnet_pred_entities
        self.procnet_entity_field = procnet_entity_field
        self.typed_entity_field = typed_entity_field
        self.procnet_type_id2field = procnet_type_id2field or {}
        self.sidecar_strict_doc_match = sidecar_strict_doc_match

        logging.info("DocEEProcessor dataset path: %s", str(self.data_path))

        self.train_path = self.data_path / "train.json"
        self.dev_path = self.data_path / "dev.json"
        self.test_path = self.data_path / "test.json"

        self.train_json = UtilData.read_raw_json_file(self.train_path)
        self.dev_json = UtilData.read_raw_json_file(self.dev_path)
        self.test_json = UtilData.read_raw_json_file(self.test_path)

        self.sidecar_paths = self._resolve_sidecar_paths(typed_entities_dir, typed_entities_files)
        self.sidecar_by_split = self._load_all_sidecars() if self.sidecar_paths else {}

        self.train_docs = self.parse_json_all(self.train_json, split_name="train")
        self.dev_docs = self.parse_json_all(self.dev_json, split_name="dev")
        self.test_docs = self.parse_json_all(self.test_json, split_name="test")

        self.SCHEMA = DocEELabel.EVENT_SCHEMA
        self.SCHEMA_KEY_ENG_CHN = DocEELabel.KEY_ENG_CHN
        self.SCHEMA_KEY_CHN_ENG = DocEELabel.KEY_CHN_ENG
        if read_pseudo_dataset:
            self.SCHEMA = PseudoDocEELabel.EVENT_SCHEMA
            self.SCHEMA_KEY_ENG_CHN = PseudoDocEELabel.KEY_ENG_CHN
            self.SCHEMA_KEY_CHN_ENG = PseudoDocEELabel.KEY_CHN_ENG

    # ------------------------------------------------------------------
    # sidecar discovery / loading
    # ------------------------------------------------------------------

    def _sidecar_filename_candidates(self, split_name: str) -> List[str]:
        return [
            f"{split_name}_doc_typed_entities.jsonl",
            f"{split_name}_typed_entities.jsonl",
            f"{split_name}_doc_procnet_entities.jsonl",
            f"{split_name}_procnet_entities.jsonl",
            f"{split_name}_doc_pred_entities.jsonl",
            f"{split_name}_pred_entities.jsonl",
            f"{split_name}_doc_entities.jsonl",
            f"{split_name}_entities.jsonl",
            f"tmp_{split_name}_doc_typed_entities.jsonl",
            f"tmp_{split_name}_typed_entities.jsonl",
            f"{split_name}.jsonl",
        ]

    def _iter_sidecar_dirs(self, typed_entities_dir: Optional[Union[str, Path]]) -> Iterable[Path]:
        candidates: List[Path] = []

        if typed_entities_dir is not None:
            candidates.append(Path(typed_entities_dir))

        for env_key in (
            "DOCEE_TYPED_ENTITIES_DIR",
            "PROCNET_TYPED_ENTITIES_DIR",
            "W2NER_TYPED_ENTITIES_DIR",
        ):
            env_value = os.environ.get(env_key)
            if env_value:
                candidates.append(Path(env_value))

        data_parent = self.data_path.parent
        module_root = Path(__file__).resolve().parents[2]
        cwd = Path.cwd()

        base_dirs = [data_parent, module_root, cwd, cwd.parent]
        for base_dir in base_dirs:
            candidates.extend([base_dir / "tmp_sidecar", base_dir])

        seen = set()
        for path in candidates:
            try:
                resolved = path.expanduser().resolve()
            except Exception:
                resolved = path.expanduser()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists() and resolved.is_dir():
                yield resolved

    def _resolve_sidecar_paths(
        self,
        typed_entities_dir: Optional[Union[str, Path]],
        typed_entities_files: Optional[Dict[str, Union[str, Path]]],
    ) -> Dict[str, Path]:
        result: Dict[str, Path] = {}

        if typed_entities_files:
            for split_name, path in typed_entities_files.items():
                if path is None:
                    continue
                candidate = Path(path)
                if candidate.exists():
                    result[split_name] = candidate
                else:
                    logging.warning("Typed entity sidecar for split=%s not found: %s", split_name, str(candidate))

        missing_splits = [split for split in ("train", "dev", "test") if split not in result]
        if not missing_splits:
            return result

        for base_dir in self._iter_sidecar_dirs(typed_entities_dir):
            for split_name in list(missing_splits):
                for filename in self._sidecar_filename_candidates(split_name):
                    candidate = base_dir / filename
                    if candidate.exists():
                        result[split_name] = candidate
                        missing_splits.remove(split_name)
                        logging.info("Loaded sidecar path for split=%s: %s", split_name, str(candidate))
                        break
            if not missing_splits:
                break

        return result

    def _load_all_sidecars(self) -> Dict[str, Dict[str, List[DocEETypedEntity]]]:
        loaded = {}
        for split_name, path in self.sidecar_paths.items():
            loaded[split_name] = self._load_typed_entity_jsonl(path)
            logging.info(
                "Loaded typed-entity sidecar for split=%s: %d docs from %s",
                split_name,
                len(loaded[split_name]),
                str(path),
            )
        return loaded

    def _load_typed_entity_jsonl(self, path: Path) -> Dict[str, List[DocEETypedEntity]]:
        doc_map: Dict[str, List[DocEETypedEntity]] = {}

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"Invalid json line in {path} line {line_idx}: {exc}")

                doc_id = record.get("doc_id")
                if doc_id is None:
                    raise ValueError(f"Missing doc_id in {path} line {line_idx}")

                entities = self._extract_typed_entities_from_sidecar_record(record)
                bucket = doc_map.setdefault(doc_id, [])
                bucket.extend(entities)

        for doc_id, entities in doc_map.items():
            entities.sort(key=lambda x: (x.sent_id, x.b, x.e, -1 if x.type_id is None else x.type_id, x.key))
            self._deduplicate_typed_entities_inplace(doc_id, entities)

        return doc_map

    def _extract_typed_entities_from_sidecar_record(self, record: Dict[str, Any]) -> List[DocEETypedEntity]:
        raw_entities = record.get(self.typed_entity_field)
        if raw_entities is None:
            raw_entities = record.get(self.procnet_entity_field, [])
        raw_entities = raw_entities or []

        default_sent_id = record.get("sent_id", record.get("sentence_idx"))
        sentence_text = record.get("sentence")
        doc_id = record["doc_id"]

        result = []
        for ent in raw_entities:
            norm = self._normalize_sidecar_entity(ent, doc_id, default_sent_id, sentence_text)
            if norm is not None:
                result.append(norm)
        return result

    def _deduplicate_typed_entities_inplace(self, doc_id: str, entities: List[DocEETypedEntity]) -> None:
        seen = set()
        deduped = []
        for ent in entities:
            uniq_key = (ent.key, ent.w2ner_key, ent.procnet_span_key)
            if uniq_key in seen:
                continue
            seen.add(uniq_key)
            deduped.append(ent)
        if len(deduped) != len(entities):
            logging.warning("Deduplicated typed entities for doc_id=%s: %d -> %d", doc_id, len(entities), len(deduped))
        entities[:] = deduped

    # ------------------------------------------------------------------
    # normalization
    # ------------------------------------------------------------------

    def _safe_extract_span(self, sentences: List[str], sent_idx: int, b: int, e: int) -> Optional[str]:
        if sent_idx < 0 or sent_idx >= len(sentences):
            return None
        sentence = sentences[sent_idx]
        if b < 0 or e > len(sentence) or b >= e:
            return None
        return sentence[b:e]

    def _safe_extract_span_from_sentence(self, sentence_text: Optional[str], b: int, e: int) -> Optional[str]:
        if sentence_text is None or b < 0 or e > len(sentence_text) or b >= e:
            return None
        return sentence_text[b:e]

    def _resolve_positions(self, ent: Dict[str, Any]) -> Optional[List[List[int]]]:
        if ent.get("positions") is not None:
            return ent["positions"]
        if ent.get("drange") is not None:
            return [ent["drange"]]

        sent_idx = ent.get("sent_idx", ent.get("sentence_idx", ent.get("sent_id")))
        b = ent.get("b")
        e = ent.get("e")
        if sent_idx is None or b is None or e is None:
            return None
        return [[int(sent_idx), int(b), int(e)]]

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
        sent_id = ent.get("sent_id", ent.get("sent_idx", ent.get("sentence_idx", default_sent_id)))
        b = ent.get("b")
        e = ent.get("e")

        if sent_id is None or b is None or e is None:
            positions = self._resolve_positions(ent)
            if not positions:
                return None
            sent_id, b, e = positions[0]

        sent_id, b, e = int(sent_id), int(b), int(e)
        token_indices = ent.get("token_indices", list(range(b, e)))

        type_id = ent.get("type_id", ent.get("label_id"))
        type_id = int(type_id) if type_id is not None else None
        type_name = self._resolve_type_name(ent, type_id)

        text = ent.get("text", ent.get("span"))
        if text is None:
            text = self._safe_extract_span_from_sentence(sentence_text, b, e)

        key = ent.get("key")
        cluster_key = ent.get("cluster_key", key)
        w2ner_key = ent.get("w2ner_key", key)
        procnet_span_key = ent.get("procnet_span_key", (sent_id, b, e))

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

    def _normalize_procnet_entity(self, ent: Dict[str, Any], sentences: List[str]) -> Optional[Dict[str, Any]]:
        positions = self._resolve_positions(ent)
        if not positions:
            return None

        sent_idx, b, e = [int(x) for x in positions[0]]
        type_id = ent.get("type_id", ent.get("label_id", -1))
        type_id = int(type_id) if type_id is not None else -1

        field = ent.get("field", ent.get("type"))
        if field is None and type_id in self.procnet_type_id2field:
            field = self.procnet_type_id2field[type_id]

        span = ent.get("span") or self._safe_extract_span(sentences, sent_idx, b, e)
        cluster_key = ent.get("key", [b, e, type_id])
        global_key = ent.get("global_key", [sent_idx, b, e, type_id])

        return {
            "key": cluster_key,
            "cluster_key": cluster_key,
            "global_key": global_key,
            "token_indices": ent.get("token_indices", list(range(b, e))),
            "b": b,
            "e": e,
            "type_id": type_id,
            "field": field,
            "score": float(ent.get("score", 1.0)),
            "head": ent.get("head"),
            "span": span,
            "positions": [[sent_idx, b, e]],
        }

    def _typed_entity_to_procnet_entity(self, ent: DocEETypedEntity) -> Dict[str, Any]:
        sent_id = int(ent.sent_id)
        b = int(ent.b)
        e = int(ent.e)
        type_id = -1 if ent.type_id is None else int(ent.type_id)

        field = ent.type_name
        if field is None and type_id in self.procnet_type_id2field:
            field = self.procnet_type_id2field[type_id]

        key = ent.cluster_key or ent.key or [b, e, type_id]
        procnet_span_key = ent.procnet_span_key or [sent_id, b, e]

        return {
            "key": key,
            "cluster_key": key,
            "global_key": [sent_id, b, e, type_id],
            "procnet_span_key": procnet_span_key,
            "token_indices": list(ent.token_indices or list(range(b, e))),
            "b": b,
            "e": e,
            "type_id": type_id,
            "field": field,
            "score": float(ent.score) if ent.score is not None else 1.0,
            "head": ent.head,
            "span": ent.text or "",
            "positions": [[sent_id, b, e]],
        }

    # ------------------------------------------------------------------
    # parsing / sidecar attach
    # ------------------------------------------------------------------

    def _normalize_doc_id_for_sidecar_match(self, doc_id: Optional[str]) -> Optional[str]:
        if doc_id is None:
            return None
        normalized = str(doc_id).strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        for suffix in (".json", ".jsonl", ".txt"):
            if lowered.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                lowered = normalized.lower()
        return lowered

    def _get_sidecar_entities(
        self,
        doc_id: str,
        split_name: Optional[str] = None,
    ) -> Tuple[List[DocEETypedEntity], Optional[str]]:
        if not self.sidecar_by_split:
            return [], None

        split_order: List[str] = []
        if split_name is not None and split_name in self.sidecar_by_split:
            split_order.append(split_name)
        split_order.extend([one for one in self.sidecar_by_split if one not in split_order])

        for one_split in split_order:
            doc_map = self.sidecar_by_split.get(one_split, {})
            if doc_id in doc_map:
                return list(doc_map[doc_id]), one_split

        normalized_doc_id = self._normalize_doc_id_for_sidecar_match(doc_id)
        if normalized_doc_id is None:
            return [], None

        for one_split in split_order:
            doc_map = self.sidecar_by_split.get(one_split, {})
            for candidate_doc_id, entities in doc_map.items():
                if self._normalize_doc_id_for_sidecar_match(candidate_doc_id) == normalized_doc_id:
                    logging.warning(
                        "Relaxed typed-entity sidecar doc match: requested=%s matched=%s split=%s",
                        doc_id,
                        candidate_doc_id,
                        one_split,
                    )
                    return list(entities), one_split

        return [], None

    def _merge_procnet_entities(self, existing: List[Dict[str, Any]], extra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged = list(existing or [])
        seen = {
            tuple(one.get("positions", [[-1, -1, -1]])[0]) + (one.get("type_id", -1),)
            for one in merged
        }
        for one in extra:
            uniq = tuple(one["positions"][0]) + (one.get("type_id", -1),)
            if uniq in seen:
                continue
            seen.add(uniq)
            merged.append(one)

        merged.sort(
            key=lambda x: (
                x["positions"][0][0],
                x["positions"][0][1],
                x["positions"][0][2],
                x.get("type_id", -1),
            )
        )
        return merged

    def _attach_typed_entities_from_sidecar(
        self,
        doc: DocEEDocumentExample,
        split_name: Optional[str] = None,
    ) -> None:
        typed_entities, matched_split = self._get_sidecar_entities(doc.doc_id, split_name)

        if self.sidecar_strict_doc_match and split_name in self.sidecar_by_split and not typed_entities:
            raise KeyError(f"Missing typed-entity sidecar for split={split_name}, doc_id={doc.doc_id}")

        doc.typed_entities = list(typed_entities)
        doc.has_typed_entities = bool(doc.typed_entities)
        doc.typed_entity_source = f"sidecar_jsonl:{matched_split}" if doc.has_typed_entities and matched_split else "none"

        had_inline_procnet = bool(getattr(doc, "procnet_entities", []))
        if self.use_procnet_pred_entities and doc.has_typed_entities:
            extra_procnet_entities = [self._typed_entity_to_procnet_entity(ent) for ent in doc.typed_entities]
            doc.procnet_entities = self._merge_procnet_entities(getattr(doc, "procnet_entities", []), extra_procnet_entities)
            if doc.procnet_entities:
                if had_inline_procnet:
                    doc.procnet_entity_source = (
                        f"inline_json+typed_entities:{matched_split}" if matched_split else "inline_json+typed_entities"
                    )
                else:
                    doc.procnet_entity_source = f"typed_entities:{matched_split}" if matched_split else "typed_entities"

        doc.has_procnet_entities = bool(getattr(doc, "procnet_entities", []))
        if not doc.has_procnet_entities:
            doc.procnet_entity_source = getattr(doc, "procnet_entity_source", None) or "gold_only"

        if hasattr(doc, "refresh_entity_node_cache"):
            doc.refresh_entity_node_cache()

    def _unwrap_json_item(self, json_item):
        if isinstance(json_item, (list, tuple)):
            if len(json_item) == 2 and isinstance(json_item[1], dict):
                return json_item[0], json_item[1]
            raise ValueError(f"Unsupported list/tuple json_item format: {type(json_item)} / len={len(json_item)}")

        if isinstance(json_item, dict):
            if "doc_id" in json_item and "sentences" in json_item:
                return json_item["doc_id"], json_item

            if "doc_id" in json_item and "data" in json_item and isinstance(json_item["data"], dict):
                return json_item["doc_id"], json_item["data"]

            if len(json_item) == 1:
                doc_id, data = next(iter(json_item.items()))
                if isinstance(data, dict):
                    return doc_id, data

            raise ValueError(f"Unsupported dict json_item keys: {list(json_item.keys())[:10]}")

        raise TypeError(f"Unsupported json_item type: {type(json_item)}")

    def _parse_procnet_entities(self, data: Dict[str, Any], sentences: List[str]) -> List[Dict[str, Any]]:
        result = []
        for ent in data.get(self.procnet_entity_field, []) or []:
            norm = self._normalize_procnet_entity(ent, sentences)
            if norm is not None:
                result.append(norm)
        result.sort(key=lambda x: (x["positions"][0][0], x["positions"][0][1], x["positions"][0][2], x["type_id"]))
        return result

    def parse_json_one(self, json_item, split_name: Optional[str] = None) -> DocEEDocumentExample:
        doc_id, data = self._unwrap_json_item(json_item)
        sentences: List[str] = data["sentences"]
        ann_mspan2dranges: Dict[str, List[list]] = data["ann_mspan2dranges"]
        ann_mspan2guess_field: Dict[str, str] = data["ann_mspan2guess_field"]
        recguid_eventname_eventdict_list = data["recguid_eventname_eventdict_list"]

        assert len(ann_mspan2dranges) == len(ann_mspan2guess_field)

        entities = [
            DocEEEntity(span=span, positions=positions, field=ann_mspan2guess_field[span])
            for span, positions in ann_mspan2dranges.items()
        ]

        events = []
        for _, event_name, event_dict in recguid_eventname_eventdict_list:
            event = {"EventType": event_name}
            event.update(event_dict)
            events.append(event)

        doc = DocEEDocumentExample(doc_id=doc_id, sentences=sentences, entities=entities, events=events)

        for entity in doc.entities:
            for sent_idx, b, e in entity.positions:
                assert entity.span == doc.sentences[sent_idx][b:e]

        doc.procnet_entities = self._parse_procnet_entities(data, sentences) if self.use_procnet_pred_entities else []
        doc.has_procnet_entities = bool(doc.procnet_entities)
        doc.procnet_entity_source = "inline_json" if doc.has_procnet_entities else "gold_only"

        self._attach_typed_entities_from_sidecar(doc, split_name)
        return doc

    def parse_json_all(self, json_data, split_name: Optional[str] = None) -> List[DocEEDocumentExample]:
        iterable = list(json_data.items()) if isinstance(json_data, dict) else json_data
        return [self.parse_json_one(one, split_name=split_name) for one in iterable]

    def get_docs_with_typed_entities(self, split_name: str = "test") -> List[DocEEDocumentExample]:
        docs = {"train": self.train_docs, "dev": self.dev_docs, "test": self.test_docs}.get(split_name, self.test_docs)
        return [doc for doc in docs if getattr(doc, "has_typed_entities", False)]
