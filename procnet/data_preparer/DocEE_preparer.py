import logging
from typing import List, Dict, Any

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

        # keep this stage backward-compatible:
        # no need to edit DocEE_conf.py immediately
        self.use_procnet_entity_nodes = getattr(config, "use_procnet_entity_nodes", False)
        self.return_procnet_entity_nodes = getattr(config, "return_procnet_entity_nodes", False)
        self.keep_gold_bio = getattr(config, "keep_gold_bio", True)

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
        [[self.longer_sentence_process_simple_cut(doc, self.config.max_len) for doc in docs] for docs in self.all_docs]

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
            self.event_role_index_to_relation[i]: i for i in range(len(self.event_role_index_to_relation))
        }

        self.event_schema_index = {}
        for k, v in self.SCHEMA.items():
            new_v = [self.event_role_relation_to_index[x] for x in v]
            new_k = self.event_type_type_to_index[k]
            self.event_schema_index[new_k] = new_v

        # ----------------------------
        # NEW: procnet type vocab
        # ----------------------------
        self.procnet_type_ids = sorted({
            ent["type_id"]
            for docs in self.all_docs
            for doc in docs
            for ent in getattr(doc, "procnet_entities", [])
        })
        self.procnet_type_index_to_id = [-1] + self.procnet_type_ids
        self.procnet_type_id_to_index = {
            type_id: idx for idx, type_id in enumerate(self.procnet_type_index_to_id)
        }

        self.train_docs = self.all_docs[0]
        self.dev_docs = self.all_docs[1]
        self.test_docs = self.all_docs[2]

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

        # NEW: procnet sidecar entities
        if hasattr(doc, "procnet_entities"):
            new_procnet_entities = []
            for ent in doc.procnet_entities:
                sent_idx, b, e = ent["positions"][0]
                if sent_idx in cut_record and e > max_len:
                    continue
                new_procnet_entities.append(ent)
            doc.procnet_entities = new_procnet_entities

    # ----------------------------
    # NEW: fragment-level typed entity nodes
    # ----------------------------
    def build_procnet_entity_nodes_for_fragment(
        self,
        example: DocEEDocumentExample,
        start_sen: int,
        end_sen: int,
        tokenizer,
    ) -> List[Dict[str, Any]]:
        nodes = []

        for ent in getattr(example, "procnet_entities", []):
            sent_idx, b, e = ent["positions"][0]

            if sent_idx < start_sen or sent_idx >= end_sen:
                continue

            local_sent_idx = sent_idx - start_sen
            sent_tokens = example.sentences_token[sent_idx]
            span_tokens = sent_tokens[b:e]
            span_token_ids = tokenizer.convert_tokens_to_ids(span_tokens)

            node = {
                # stable ids
                "cluster_key": ent["cluster_key"],                # [b, e, type_id]
                "global_key": ent["global_key"],                  # [sent_idx, b, e, type_id]
                "local_key": [local_sent_idx, b, e, ent["type_id"]],

                # span position
                "global_sent_idx": sent_idx,
                "sent_idx": local_sent_idx,
                "b": b,
                "e": e,

                # typed info
                "type_id": ent["type_id"],
                "type_index": self.procnet_type_id_to_index.get(ent["type_id"], 0),
                "field": ent.get("field", None),

                # confidence / structural info
                "score": ent.get("score", 1.0),
                "head": ent.get("head", None),

                # content
                "span": ent.get("span", None),
                "token_indices": ent.get("token_indices", list(range(b, e))),
                "token_ids": span_token_ids,
            }
            nodes.append(node)

        nodes.sort(key=lambda x: (
            x["global_sent_idx"],
            x["b"],
            x["e"],
            x["type_id"],
        ))
        return nodes

    def get_loader_for_flattened_fragment_before_event(self):
        tokenizer = self.get_auto_tokenizer()

        class MyDataSet(Dataset):

            def __init__(this, examples: List[DocEEDocumentExample], preparer, tokenizer):
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

                    # NEW: build fragment-local typed entity nodes
                    one_fragment_nodes = self.build_procnet_entity_nodes_for_fragment(
                        example=example,
                        start_sen=frag_start,
                        end_sen=frag_end,
                        tokenizer=this.tokenizer,
                    )
                    procnet_entity_nodes.append(one_fragment_nodes)

                # original event label
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

                # IMPORTANT:
                # stage-1 default: keep original 5-tuple for full backward compatibility
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

        train_dataset = MyDataSet(self.train_docs, self, tokenizer)
        dev_dataset = MyDataSet(self.dev_docs, self, tokenizer)
        test_dataset = MyDataSet(self.test_docs, self, tokenizer)

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
