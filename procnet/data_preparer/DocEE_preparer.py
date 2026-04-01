
import logging
from typing import List, Dict, Any, Iterable, Tuple

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from procnet.data_preparer.basic_preparer import BasicPreparer
from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_example.DocEEexample import DocEEDocumentExample
from procnet.conf.DocEE_conf import DocEEConfig
from procnet.utils.util_string import UtilString


class DocEEPreparer(BasicPreparer):

    def __init__(self, config: DocEEConfig, processor: DocEEProcessor):
        super().__init__(model_name=config.model_name)
        self.config = config

        # stage-1 compatibility flags:
        # these attributes can be added to DocEE_conf.py later, but are optional now.
        self.use_procnet_entity_nodes = getattr(config, "use_procnet_entity_nodes", False)
        self.return_procnet_entity_nodes = getattr(config, "return_procnet_entity_nodes", False)
        self.keep_gold_bio = getattr(config, "keep_gold_bio", True)
        self.procnet_entity_key_mode = getattr(config, "procnet_entity_key_mode", "token_ids")
        self.procnet_entity_include_type_in_key = getattr(
            config, "procnet_entity_include_type_in_key", False
        )

        self.seq_label_BIO_tag_set = set()
        self.seq_label_category_set = set()
        self.event_type_label_set = set()
        self.event_role_label_set = set()

        self.SCHEMA = processor.SCHEMA
        self.SCHEMA_KEY_CHN_ENG = processor.SCHEMA_KEY_CHN_ENG
        self.SCHEMA_KEY_ENG_CHN = processor.SCHEMA_KEY_ENG_CHN

        self.all_docs: List[List[DocEEDocumentExample]] = [
            processor.train_docs, processor.dev_docs, processor.test_docs
        ]

        [self.tokenize_sentences(x) for x in self.all_docs]
        [
            [self.longer_sentence_process_simple_cut(doc, self.config.max_len) for doc in docs]
            for docs in self.all_docs
        ]

        # keep original BIO path for now
        [[self.seq_label_BIO_tags_generate(doc) for doc in one_docs] for one_docs in self.all_docs]

        # ----------------------------
        # original vocab build
        # ----------------------------
        self.seq_bio_index_to_cate = ["Null"] + sorted(list(self.seq_label_category_set))
        self.seq_bio_cate_to_index = {
            self.seq_bio_index_to_cate[i]: i for i in range(len(self.seq_bio_index_to_cate))
        }

        self.seq_BIO_index_to_tag = (
            ["O"]
            + [x + "-B" for x in self.seq_bio_index_to_cate[1:]]
            + [x + "-I" for x in self.seq_bio_index_to_cate[1:]]
        )
        self.seq_BIO_tag_to_index = {
            self.seq_BIO_index_to_tag[i]: i for i in range(len(self.seq_BIO_index_to_tag))
        }

        self.seq_bio_tag_index_to_cate_index = {0: 0}
        for cate in self.seq_bio_index_to_cate[1:]:
            cate_index = self.seq_bio_cate_to_index[cate]
            b_index = self.seq_BIO_tag_to_index[cate + "-B"]
            i_index = self.seq_BIO_tag_to_index[cate + "-I"]
            self.seq_bio_tag_index_to_cate_index[b_index] = cate_index
            self.seq_bio_tag_index_to_cate_index[i_index] = cate_index

        event_type_label_set_from_data = set()
        event_role_label_set_from_data = set()
        for docs in self.all_docs:
            for doc in docs:
                for event in doc.events:
                    for k, v in event.items():
                        if k == "EventType":
                            event_type_label_set_from_data.add(v)
                        else:
                            event_role_label_set_from_data.add(k)

        for k, v in self.SCHEMA.items():
            self.event_type_label_set.add(k)
            [self.event_role_label_set.add(x) for x in v]

        if self.event_type_label_set != event_type_label_set_from_data:
            logging.warning(
                "event schema type and data not same with schema: {} and: data {}".format(
                    self.event_type_label_set, event_type_label_set_from_data
                )
            )
        if self.event_role_label_set != event_role_label_set_from_data:
            logging.warning(
                "event schema role and data not same with schema: {} and: data {}".format(
                    self.event_role_label_set, event_role_label_set_from_data
                )
            )

        self.event_type_index_to_type = ["Null"] + sorted(list(self.event_type_label_set))
        self.event_type_type_to_index = {
            self.event_type_index_to_type[i]: i for i in range(len(self.event_type_index_to_type))
        }

        self.event_role_index_to_relation = ["Null"] + sorted(list(self.event_role_label_set))
        self.event_role_relation_to_index = {
            self.event_role_index_to_relation[i]: i
            for i in range(len(self.event_role_index_to_relation))
        }

        self.event_schema_index = {}
        for k, v in self.SCHEMA.items():
            new_v = [self.event_role_relation_to_index[x] for x in v]
            new_k = self.event_type_type_to_index[k]
            self.event_schema_index[new_k] = new_v

        self.train_docs = self.all_docs[0]
        self.dev_docs = self.all_docs[1]
        self.test_docs = self.all_docs[2]

        # ----------------------------
        # procnet typed-entity vocab
        # ----------------------------
        self.procnet_type_ids = self._collect_procnet_type_ids()
        self.procnet_type_index_to_id = [-1] + self.procnet_type_ids
        self.procnet_type_id_to_index = {
            type_id: idx for idx, type_id in enumerate(self.procnet_type_index_to_id)
        }

        self._log_procnet_sidecar_coverage()

        pos_event_num_total = 0
        for doc in self.train_docs:
            pos_event_num_total += len(doc.events)
        all_event_num_total = config.proxy_slot_num * len(self.train_docs)
        neg_event_num_total = all_event_num_total - pos_event_num_total
        self.pos_event_ratio_total = pos_event_num_total / all_event_num_total
        self.neg_event_ratio_total = neg_event_num_total / all_event_num_total

        neg_bio_num = 0
        total_bio_num = 0
        for doc in self.train_docs:
            for seq in doc.seq_BIO_tags:
                for x in seq:
                    if x == "O":
                        neg_bio_num += 1
                    total_bio_num += 1
        pos_bio_num = total_bio_num - neg_bio_num
        self.pos_bio_ratio_total = pos_bio_num / total_bio_num
        self.neg_bio_ratio_total = neg_bio_num / total_bio_num

    def tokenize_sentences(self, docs: List[DocEEDocumentExample]):
        for doc in docs:
            doc.sentences_token = [self.my_tokenize(x) for x in doc.sentences]

    def my_tokenize(self, s: str) -> List[str]:
        return UtilString.character_tokenize(s)

    def find_end_pos_for_max_len(self, doc: DocEEDocumentExample, start: int, max_len: int) -> int:
        acc_len = 0
        end = start
        for i in range(start, len(doc.sentences_token)):
            acc_len += len(doc.sentences_token[i])
            if acc_len > max_len:
                break
            else:
                end = i + 1
        if start == end:
            raise Exception(
                "A sentence is more than max_len len! which is {}".format(
                    [len(x) for x in doc.sentences_token]
                )
            )
        assert sum([len(x) for x in doc.sentences_token[start:end]]) <= max_len
        return end

    def seq_label_BIO_tags_generate(self, doc: DocEEDocumentExample, mode: str = "BIO"):
        BIO_tags = [["O"] * len(sentence) for sentence in doc.sentences_token]
        for entity in doc.entities:
            cate = entity.field
            self.seq_label_category_set.add(cate)
            B_tag = cate + "-B"
            I_tag = cate + "-I"
            for pos in entity.positions:
                if mode == "BIO":
                    BIO_tags[pos[0]][pos[1]] = B_tag
                    BIO_tags[pos[0]][pos[1] + 1: pos[2]] = [I_tag] * (pos[2] - pos[1] - 1)
                    self.seq_label_BIO_tag_set.add(B_tag)
                    self.seq_label_BIO_tag_set.add(I_tag)
        doc.seq_BIO_tags = BIO_tags

    # ----------------------------
    # typed-entity helpers
    # ----------------------------
    def _collect_procnet_type_ids(self) -> List[int]:
        type_ids = set()
        for docs in self.all_docs:
            for doc in docs:
                for ent in self._get_doc_sidecar_entities(doc):
                    type_id = ent.get("type_id", None)
                    if type_id is None:
                        continue
                    try:
                        type_ids.add(int(type_id))
                    except Exception:
                        continue
        return sorted(type_ids)

    def _get_doc_sidecar_entities(self, doc: DocEEDocumentExample) -> List[Dict[str, Any]]:
        """
        Normalize whatever the processor/doc example currently exposes.

        Supported sources:
        - doc.procnet_entities              (dict list)
        - doc.typed_entities                (dataclass / dict list)
        - doc.entity_nodes                  (dict list)
        """
        raw_entities = None
        for attr_name in ["procnet_entities", "typed_entities", "entity_nodes"]:
            if hasattr(doc, attr_name):
                candidate = getattr(doc, attr_name)
                if candidate:
                    raw_entities = candidate
                    break
        if raw_entities is None:
            return []
        return [self._entity_like_to_dict(x) for x in raw_entities]

    def _entity_like_to_dict(self, ent: Any) -> Dict[str, Any]:
        if isinstance(ent, dict):
            return dict(ent)
        if hasattr(ent, "to_dict") and callable(ent.to_dict):
            return dict(ent.to_dict())
        if hasattr(ent, "__dict__"):
            return dict(ent.__dict__)
        raise TypeError("Unsupported typed entity object type: {}".format(type(ent)))

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _normalize_positions(self, ent: Dict[str, Any]) -> List[List[int]]:
        """
        Convert entity position payload into a normalized list of [sent_idx, b, e].
        """
        positions = ent.get("positions", None)
        if positions:
            normalized_positions = []
            for pos in positions:
                if pos is None or len(pos) < 3:
                    continue
                normalized_positions.append([
                    self._safe_int(pos[0]),
                    self._safe_int(pos[1]),
                    self._safe_int(pos[2]),
                ])
            if normalized_positions:
                return normalized_positions

        sent_idx = ent.get("sent_id", ent.get("sent_idx", None))
        b = ent.get("b", ent.get("start", ent.get("token_start", None)))
        e = ent.get("e", ent.get("end", ent.get("token_end", None)))
        if sent_idx is None or b is None or e is None:
            return []
        return [[self._safe_int(sent_idx), self._safe_int(b), self._safe_int(e)]]

    def _iter_entity_mentions(self, ent: Dict[str, Any]) -> Iterable[Tuple[int, int, int, int]]:
        positions = self._normalize_positions(ent)
        for pos_idx, pos in enumerate(positions):
            if len(pos) < 3:
                continue
            sent_idx, b, e = pos[:3]
            if e <= b:
                continue
            yield pos_idx, sent_idx, b, e

    def _sentence_start_offsets(self, sentence_tokens: List[List[str]]) -> List[int]:
        """
        Offsets in the flattened fragment sequence that includes a leading [CLS].
        Example:
            sent0 len=5 -> starts at 1
            sent1 len=7 -> starts at 6
        """
        starts = []
        cur = 1
        for sent in sentence_tokens:
            starts.append(cur)
            cur += len(sent)
        return starts

    def _build_procnet_span_key(
        self,
        token_ids: List[int],
        sent_idx: int,
        b: int,
        e: int,
        type_id: int,
    ):
        if self.procnet_entity_key_mode == "position":
            base_key = (sent_idx, b, e)
        else:
            base_key = tuple(token_ids)

        if self.procnet_entity_include_type_in_key:
            if isinstance(base_key, tuple):
                return base_key + (type_id,)
            return tuple(list(base_key) + [type_id])
        return base_key

    def _get_entity_type_name(self, ent: Dict[str, Any]):
        return ent.get(
            "field",
            ent.get(
                "type_name",
                ent.get(
                    "type",
                    ent.get("label", None)
                )
            )
        )

    def _count_doc_sidecar_entities(self, doc: DocEEDocumentExample) -> int:
        try:
            return len(self._get_doc_sidecar_entities(doc))
        except Exception:
            return 0

    def _log_procnet_sidecar_coverage(self):
        split_docs = [
            ("train", self.all_docs[0]),
            ("dev", self.all_docs[1]),
            ("test", self.all_docs[2]),
        ]
        for split_name, docs in split_docs:
            docs_with_entities = 0
            entity_total = 0
            for doc in docs:
                ent_count = self._count_doc_sidecar_entities(doc)
                if ent_count > 0:
                    docs_with_entities += 1
                    entity_total += ent_count
            logging.info(
                "procnet sidecar coverage split=%s docs_with_entities=%s/%s entities=%s",
                split_name,
                docs_with_entities,
                len(docs),
                entity_total,
            )

    def find_dataset_item_by_doc_id(self, target_doc_id: str):
        if not target_doc_id:
            return None, None, None

        tokenizer = self.get_auto_tokenizer()

        class MyDataSet(Dataset):
            def __init__(this, split_name: str, examples: List[DocEEDocumentExample], preparer, tokenizer):
                this.split_name = split_name
                this.examples = examples
                this.preparer = preparer
                this.tokenizer = tokenizer

            def __len__(this):
                return len(this.examples)

            def __getitem__(this, index):
                example = this.examples[index]
                total_sentence_nums = len(example.sentences_token)

                sub_examples = []
                sub_ranges = []

                start = 0
                end = 0
                while end < total_sentence_nums:
                    end = self.find_end_pos_for_max_len(
                        doc=example, start=start, max_len=self.config.max_len
                    )
                    sub_example = example.get_fragment(start_sen=start, end_sen=end)
                    sub_examples.append(sub_example)
                    sub_ranges.append((start, end))
                    start = end

                doc_id = example.doc_id
                input_ids = []
                input_att_masks = []
                BIO_ids = []
                procnet_entity_nodes = []

                for sub_example, (frag_start, frag_end) in zip(sub_examples, sub_ranges):
                    input_token = [this.tokenizer.cls_token]
                    for x in sub_example.sentences_token:
                        input_token += x
                    input_id = this.tokenizer.convert_tokens_to_ids(input_token)
                    input_ids.append(torch.LongTensor(input_id))

                    input_att_mask = [1] * len(input_id)
                    input_att_masks.append(torch.LongTensor(input_att_mask))

                    BIO_tags = ["O"]
                    for x in sub_example.seq_BIO_tags:
                        BIO_tags += x
                    BIO_id = [self.seq_BIO_tag_to_index[x] for x in BIO_tags]
                    BIO_ids.append(torch.LongTensor(BIO_id))

                    if this.preparer.return_procnet_entity_nodes or this.preparer.use_procnet_entity_nodes:
                        one_fragment_nodes = self.build_procnet_entity_nodes_for_fragment(
                            example=example,
                            start_sen=frag_start,
                            end_sen=frag_end,
                            tokenizer=this.tokenizer,
                        )
                    else:
                        one_fragment_nodes = []
                    procnet_entity_nodes.append(one_fragment_nodes)

                events_label = []
                for event in example.events:
                    event_label = {}
                    for k, v in event.items():
                        if k == "EventType":
                            event_label[k] = self.event_type_type_to_index[v]
                        else:
                            if v is not None:
                                v_id = this.tokenizer.convert_tokens_to_ids(self.my_tokenize(v))
                                event_label[tuple(v_id)] = self.event_role_relation_to_index[k]
                    events_label.append(event_label)

                if this.preparer.return_procnet_entity_nodes:
                    return (
                        doc_id,
                        input_ids,
                        input_att_masks,
                        BIO_ids,
                        events_label,
                        procnet_entity_nodes,
                    )

                return doc_id, input_ids, input_att_masks, BIO_ids, events_label

        split_specs = [
            ("train", self.train_docs),
            ("dev", self.dev_docs),
            ("test", self.test_docs),
        ]
        for split_name, docs in split_specs:
            dataset = MyDataSet(split_name, docs, self, tokenizer)
            for idx, doc in enumerate(docs):
                if getattr(doc, "doc_id", None) == target_doc_id:
                    return split_name, idx, dataset[idx]
        return None, None, None

    def longer_sentence_process_simple_cut(self, doc: DocEEDocumentExample, max_len: int):
        all_short = True
        for sentence in doc.sentences_token:
            if len(sentence) > max_len:
                all_short = False
                break
        if all_short:
            return

        cut_record = []
        for i in range(len(doc.sentences_token)):
            if len(doc.sentences_token[i]) > max_len:
                doc.sentences_token[i] = doc.sentences_token[i][:max_len]
                cut_record.append(i)

        # original gold entities
        for entity in doc.entities:
            all_valid = True
            for pos in entity.positions:
                if pos[0] in cut_record and pos[2] > max_len:
                    all_valid = False
                    break
            if not all_valid:
                new_positions = []
                for pos in entity.positions:
                    if pos[0] in cut_record and pos[2] > max_len:
                        continue
                    else:
                        new_positions.append(pos)
                entity.positions = new_positions

        # sidecar typed entities / procnet entities
        for attr_name in ["procnet_entities", "typed_entities", "entity_nodes"]:
            if not hasattr(doc, attr_name):
                continue
            raw_entities = getattr(doc, attr_name)
            if not raw_entities:
                continue

            new_entities = []
            for raw_ent in raw_entities:
                ent = self._entity_like_to_dict(raw_ent)
                kept_positions = []
                for pos in self._normalize_positions(ent):
                    sent_idx, b, e = pos
                    if sent_idx in cut_record and e > max_len:
                        continue
                    kept_positions.append([sent_idx, b, e])
                if not kept_positions:
                    continue

                if isinstance(raw_ent, dict):
                    new_ent = dict(raw_ent)
                    new_ent["positions"] = kept_positions
                    if len(kept_positions) == 1:
                        new_ent["sent_id"] = kept_positions[0][0]
                        new_ent["b"] = kept_positions[0][1]
                        new_ent["e"] = kept_positions[0][2]
                    new_entities.append(new_ent)
                else:
                    if hasattr(raw_ent, "positions"):
                        raw_ent.positions = kept_positions
                    new_entities.append(raw_ent)

            setattr(doc, attr_name, new_entities)

    # ----------------------------
    # fragment-level typed entity nodes
    # ----------------------------
    def build_procnet_entity_nodes_for_fragment(
        self,
        example: DocEEDocumentExample,
        start_sen: int,
        end_sen: int,
        tokenizer,
    ) -> List[Dict[str, Any]]:
        """
        Build fragment-local node dicts from sidecar typed entities.

        Important design choices:
        - one node per mention/span, not per cluster
        - keep document-level ids and add fragment-local ids
        - preserve both:
            * procnet_span_key : key later ProcNet components can align on
            * w2ner_key        : provenance / debugging key
        - provide flat_b / flat_e so the model can directly slice hidden states
          from the flattened fragment sequence without reconstructing offsets again
        """
        nodes = []
        fragment_sentence_tokens = example.sentences_token[start_sen:end_sen]
        fragment_sentence_starts = self._sentence_start_offsets(fragment_sentence_tokens)

        node_id = -1
        for ent in self._get_doc_sidecar_entities(example):
            type_id = self._safe_int(ent.get("type_id", -1), -1)
            type_index = self.procnet_type_id_to_index.get(type_id, 0)
            field = self._get_entity_type_name(ent)

            for mention_idx, sent_idx, b, e in self._iter_entity_mentions(ent):
                if sent_idx < start_sen or sent_idx >= end_sen:
                    continue

                local_sent_idx = sent_idx - start_sen
                sent_tokens = example.sentences_token[sent_idx]

                if b < 0 or e > len(sent_tokens) or e <= b:
                    continue

                span_tokens = sent_tokens[b:e]
                token_ids = tokenizer.convert_tokens_to_ids(span_tokens)

                sentence_flat_start = fragment_sentence_starts[local_sent_idx]
                flat_b = sentence_flat_start + b
                flat_e = sentence_flat_start + e
                flat_token_indices = list(range(flat_b, flat_e))

                procnet_span_key = self._build_procnet_span_key(
                    token_ids=token_ids,
                    sent_idx=sent_idx,
                    b=b,
                    e=e,
                    type_id=type_id,
                )
                w2ner_key = (sent_idx, b, e, type_id)

                node_id += 1
                node = {
                    # stable ids / provenance
                    "node_id": node_id,
                    "cluster_key": ent.get("cluster_key", (b, e, type_id)),
                    "global_key": ent.get("global_key", (sent_idx, b, e, type_id)),
                    "local_key": (local_sent_idx, b, e, type_id),
                    "w2ner_key": ent.get("w2ner_key", w2ner_key),
                    "procnet_span_key": procnet_span_key,
                    "mention_index": mention_idx,

                    # fragment and position info
                    "fragment_start_sen": start_sen,
                    "fragment_end_sen": end_sen,
                    "global_sent_idx": sent_idx,
                    "sent_idx": local_sent_idx,
                    "fragment_sent_id": local_sent_idx,
                    "b": b,
                    "e": e,
                    "flat_b": flat_b,
                    "flat_e": flat_e,

                    # typed info
                    "type_id": type_id,
                    "type_index": type_index,
                    "field": field,
                    "type_name": field,

                    # confidence / structure
                    "score": ent.get("score", 1.0),
                    "head": ent.get("head", None),

                    # content
                    "text": ent.get("text", ent.get("span", None)),
                    "span": ent.get("span", ent.get("text", None)),
                    "token_indices": ent.get("token_indices", list(range(b, e))),
                    "flat_token_indices": flat_token_indices,
                    "token_ids": token_ids,
                    "span_tokens": span_tokens,

                    # keep raw payload for debugging
                    "raw_w2ner": ent.get("raw_w2ner", ent.get("raw", None)),
                }
                nodes.append(node)

        nodes.sort(
            key=lambda x: (
                x["global_sent_idx"],
                x["b"],
                x["e"],
                x["type_id"],
                x["mention_index"],
            )
        )

        # reassign node ids after sort for stable ordering inside fragment
        for idx, node in enumerate(nodes):
            node["node_id"] = idx

        return nodes

    def get_loader_for_flattened_fragment_before_event(self):
        tokenizer = self.get_auto_tokenizer()

        class MyDataSet(Dataset):

            def __init__(this, split_name: str, examples: List[DocEEDocumentExample], preparer, tokenizer):
                this.split_name = split_name
                this.examples = examples
                this.preparer = preparer
                this.tokenizer = tokenizer

            def __len__(this):
                return len(this.examples)

            def __getitem__(this, index):
                example = this.examples[index]
                total_sentence_nums = len(example.sentences_token)

                sub_examples = []
                sub_ranges = []

                start = 0
                end = 0
                while end < total_sentence_nums:
                    end = self.find_end_pos_for_max_len(
                        doc=example, start=start, max_len=self.config.max_len
                    )
                    sub_example = example.get_fragment(start_sen=start, end_sen=end)
                    sub_examples.append(sub_example)
                    sub_ranges.append((start, end))
                    start = end

                doc_id = example.doc_id
                input_ids = []
                input_att_masks = []
                BIO_ids = []
                procnet_entity_nodes = []

                for sub_example, (frag_start, frag_end) in zip(sub_examples, sub_ranges):
                    input_token = [this.tokenizer.cls_token]
                    for x in sub_example.sentences_token:
                        input_token += x
                    input_id = this.tokenizer.convert_tokens_to_ids(input_token)
                    input_ids.append(torch.LongTensor(input_id))

                    input_att_mask = [1] * len(input_id)
                    input_att_masks.append(torch.LongTensor(input_att_mask))

                    BIO_tags = ["O"]
                    for x in sub_example.seq_BIO_tags:
                        BIO_tags += x
                    BIO_id = [self.seq_BIO_tag_to_index[x] for x in BIO_tags]
                    BIO_ids.append(torch.LongTensor(BIO_id))

                    # build fragment-local typed entity nodes
                    if this.preparer.return_procnet_entity_nodes or this.preparer.use_procnet_entity_nodes:
                        one_fragment_nodes = self.build_procnet_entity_nodes_for_fragment(
                            example=example,
                            start_sen=frag_start,
                            end_sen=frag_end,
                            tokenizer=this.tokenizer,
                        )
                    else:
                        one_fragment_nodes = []
                    procnet_entity_nodes.append(one_fragment_nodes)

                # original event label path kept as the default for full backward compatibility.
                # The later trainer/model thin-adapter can consume procnet_entity_nodes and choose
                # node["procnet_span_key"] as the actual span alignment key.
                events_label = []
                for event in example.events:
                    event_label = {}
                    for k, v in event.items():
                        if k == "EventType":
                            event_label[k] = self.event_type_type_to_index[v]
                        else:
                            if v is not None:
                                v_id = this.tokenizer.convert_tokens_to_ids(self.my_tokenize(v))
                                event_label[tuple(v_id)] = self.event_role_relation_to_index[k]
                    events_label.append(event_label)

                if this.preparer.return_procnet_entity_nodes:
                    return (
                        doc_id,
                        input_ids,
                        input_att_masks,
                        BIO_ids,
                        events_label,
                        procnet_entity_nodes,
                    )

                return doc_id, input_ids, input_att_masks, BIO_ids, events_label

        train_dataset = MyDataSet("train", self.train_docs, self, tokenizer)
        dev_dataset = MyDataSet("dev", self.dev_docs, self, tokenizer)
        test_dataset = MyDataSet("test", self.test_docs, self, tokenizer)

        logging.info(
            "dataset {} train, {} dev, {} test".format(
                len(train_dataset), len(dev_dataset), len(test_dataset)
            )
        )

        def my_collate_fn(batch):
            assert len(batch) == 1
            batch = batch[0]

            if self.return_procnet_entity_nodes:
                doc_ids = batch[0]
                input_ids = batch[1]
                input_att_mask = batch[2]
                BIO_ids = batch[3]
                events_labels = batch[4]
                procnet_entity_nodes = batch[5]
                return (
                    doc_ids,
                    input_ids,
                    input_att_mask,
                    BIO_ids,
                    events_labels,
                    procnet_entity_nodes,
                )

            doc_ids = batch[0]
            input_ids = batch[1]
            input_att_mask = batch[2]
            BIO_ids = batch[3]
            events_labels = batch[4]
            return doc_ids, input_ids, input_att_mask, BIO_ids, events_labels

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=my_collate_fn,
            shuffle=self.config.data_loader_shuffle,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=1,
            collate_fn=my_collate_fn,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=my_collate_fn,
            shuffle=False,
        )
        return train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader
