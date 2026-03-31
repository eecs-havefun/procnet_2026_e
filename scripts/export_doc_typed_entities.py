from pathlib import Path

# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export W2NER sentence-level predictions to doc-level typed entities for ProcNet.

Assumptions based on your current output.json sample:
- Each prediction record contains:
  {
    "doc_id": ...,
    "sent_id": ...,
    "sentence": [...],
    "procnet_entities": [
      {
        "token_indices": [...],
        "b": 0,
        "e": 12,
        "type_id": 3,
        "type": "companyname",
        "score": 0.941056,
        "head": 0,
        "text": "维维食品饮料股份有限公司"
      }
    ]
  }

Important:
- This script treats `e` as an exclusive end index because your sample has:
    b = 0, e = 12, token_indices = [0..11]
- It validates that token_indices == list(range(b, e)) by default.
- It reads BOTH the source split file and the prediction file, aligns by (doc_id, sent_id),
  and aggregates sentence-level entities into doc-level typed_entities.

Typical usage:
python export_doc_typed_entities.py \
  --source_json ./data/mydata_doc_drop450/dev.json \
  --pred_json ./output/dev_output.json \
  --output_jsonl ./exports/dev_doc_typed_entities.jsonl \
  --report_json ./exports/dev_export_report.json

And similarly for test:
python export_doc_typed_entities.py \
  --source_json ./data/mydata_doc_drop450/test.json \
  --pred_json ./output/test_output.json \
  --output_jsonl ./exports/test_doc_typed_entities.jsonl \
  --report_json ./exports/test_export_report.json
"""

import argparse
import json
from collections import defaultdict


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sentence_to_text(sentence):
    if isinstance(sentence, list):
        return "".join(sentence)
    return str(sentence)


def make_pred_index(pred_rows):
    pred_index = {}
    dup_keys = []

    for i, row in enumerate(pred_rows):
        key = (row.get("doc_id"), int(row.get("sent_id")))
        if key in pred_index:
            dup_keys.append({"key": [row.get("doc_id"), int(row.get("sent_id"))], "row_index": i})
        pred_index[key] = row

    return pred_index, dup_keys


def normalize_entity(ent, doc_id, sent_id, sentence_tokens, strict_range_check=True):
    b = int(ent["b"])
    e = int(ent["e"])  # treated as exclusive
    type_id = int(ent["type_id"])

    token_indices = ent.get("token_indices")
    if token_indices is None:
        token_indices = list(range(b, e))
    token_indices = [int(x) for x in token_indices]

    if strict_range_check:
        expected = list(range(b, e))
        if token_indices != expected:
            raise ValueError(
                f"token_indices mismatch for {(doc_id, sent_id)}: "
                f"got {token_indices}, expected {expected}, with b={b}, e={e}"
            )

    head = ent.get("head", b)
    head = int(head)

    text = ent.get("text")
    if text is None:
        text = "".join(sentence_tokens[b:e])

    norm = {
        "key": f"{doc_id}:{sent_id}:{b}:{e}:{type_id}",
        "cluster_key": f"{doc_id}:{sent_id}:{b}:{e}:{type_id}",
        "doc_id": doc_id,
        "sent_id": int(sent_id),
        "token_indices": token_indices,
        "b": b,
        "e": e,
        "type_id": type_id,
        "type": ent.get("type"),
        "score": float(ent.get("score", 1.0)),
        "head": head,
        "text": text,
        "source": "w2ner",
    }
    return norm


def validate_entity(ent, sentence_tokens):
    n = len(sentence_tokens)
    errors = []

    b = ent["b"]
    e = ent["e"]
    token_indices = ent["token_indices"]
    head = ent["head"]

    if not (0 <= b <= e <= n):
        errors.append(f"span_out_of_range:b={b},e={e},n={n}")

    if token_indices and (min(token_indices) < 0 or max(token_indices) >= n):
        errors.append(f"token_indices_out_of_range:{token_indices}")

    if token_indices and head not in token_indices:
        errors.append(f"head_not_in_token_indices:head={head},token_indices={token_indices}")

    recovered = "".join(sentence_tokens[b:e])
    if ent["text"] != recovered:
        errors.append(f"text_mismatch:ent_text={ent['text']},recovered={recovered}")

    return errors


def export_doc_level(source_rows, pred_rows, strict_sentence_match=True, strict_range_check=True):
    pred_index, dup_pred_keys = make_pred_index(pred_rows)

    docs = defaultdict(lambda: {"doc_id": None, "sentences": [], "typed_entities": []})
    missing_pred = []
    sentence_mismatches = []
    validation_errors = []

    for i, src in enumerate(source_rows):
        doc_id = src["doc_id"]
        sent_id = int(src["sent_id"])
        sentence_tokens = src["sentence"]
        src_text = sentence_to_text(sentence_tokens)

        pred = pred_index.get((doc_id, sent_id))
        if pred is None:
            missing_pred.append({"doc_id": doc_id, "sent_id": sent_id})
            pred = {
                "doc_id": doc_id,
                "sent_id": sent_id,
                "sentence": sentence_tokens,
                "procnet_entities": [],
            }

        pred_text = sentence_to_text(pred.get("sentence", []))
        if strict_sentence_match and pred_text != src_text:
            sentence_mismatches.append({
                "doc_id": doc_id,
                "sent_id": sent_id,
                "source_text": src_text,
                "pred_text": pred_text,
            })

        doc_obj = docs[doc_id]
        doc_obj["doc_id"] = doc_id
        doc_obj["sentences"].append({
            "sent_id": sent_id,
            "sentence": sentence_tokens,
            "text": src_text,
        })

        for ent in pred.get("procnet_entities", []):
            norm = normalize_entity(
                ent=ent,
                doc_id=doc_id,
                sent_id=sent_id,
                sentence_tokens=sentence_tokens,
                strict_range_check=strict_range_check,
            )
            errs = validate_entity(norm, sentence_tokens)
            if errs:
                validation_errors.append({
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "entity_key": norm["key"],
                    "errors": errs,
                })
            doc_obj["typed_entities"].append(norm)

    # sort each doc by sent_id
    doc_rows = []
    duplicate_entity_keys = []
    for doc_id, obj in docs.items():
        obj["sentences"] = sorted(obj["sentences"], key=lambda x: x["sent_id"])
        obj["typed_entities"] = sorted(
            obj["typed_entities"],
            key=lambda x: (x["sent_id"], x["b"], x["e"], x["type_id"])
        )

        seen = set()
        deduped = []
        for ent in obj["typed_entities"]:
            if ent["key"] in seen:
                duplicate_entity_keys.append({"doc_id": doc_id, "entity_key": ent["key"]})
                continue
            seen.add(ent["key"])
            deduped.append(ent)
        obj["typed_entities"] = deduped

        obj["num_sentences"] = len(obj["sentences"])
        obj["num_typed_entities"] = len(obj["typed_entities"])
        doc_rows.append(obj)

    doc_rows.sort(key=lambda x: x["doc_id"])

    report = {
        "num_source_sentences": len(source_rows),
        "num_pred_sentences": len(pred_rows),
        "num_docs": len(doc_rows),
        "num_missing_pred_sentences": len(missing_pred),
        "num_sentence_mismatches": len(sentence_mismatches),
        "num_duplicate_pred_sentence_keys": len(dup_pred_keys),
        "num_duplicate_entity_keys": len(duplicate_entity_keys),
        "num_validation_errors": len(validation_errors),
        "sample_missing_pred": missing_pred[:20],
        "sample_sentence_mismatches": sentence_mismatches[:20],
        "sample_duplicate_pred_sentence_keys": dup_pred_keys[:20],
        "sample_duplicate_entity_keys": duplicate_entity_keys[:20],
        "sample_validation_errors": validation_errors[:20],
    }

    return doc_rows, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_json", required=True, help="Filtered dev.json or test.json")
    parser.add_argument("--pred_json", required=True, help="W2NER output.json for that split")
    parser.add_argument("--output_jsonl", required=True, help="Output doc-level jsonl")
    parser.add_argument("--report_json", required=True, help="Export report json")
    parser.add_argument("--no_strict_sentence_match", action="store_true")
    parser.add_argument("--no_strict_range_check", action="store_true")
    args = parser.parse_args()

    source_rows = load_json(args.source_json)
    pred_rows = load_json(args.pred_json)

    doc_rows, report = export_doc_level(
        source_rows=source_rows,
        pred_rows=pred_rows,
        strict_sentence_match=not args.no_strict_sentence_match,
        strict_range_check=not args.no_strict_range_check,
    )

    dump_jsonl(args.output_jsonl, doc_rows)
    dump_json(args.report_json, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
