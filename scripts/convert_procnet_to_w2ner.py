from pathlib import Path

# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict-alignment ProcNet -> W2NER converter.

What this script fixes compared with the minimal version
-------------------------------------------------------
1) Strict span-text alignment:
   - For each drange [sent_idx, start, end), verify that:
         sentence_text[start:end] == recovered_entity_text
   - This is the most important "strict alignment" check.

2) Keep reversible metadata:
   - sample_id
   - doc_id
   - sent_id
   - text
   - sentence
   - ner
   - entities
   These fields make later doc-level reconstruction much safer.

3) Preserve same-span multi-type entities by default:
   - Do NOT collapse entities that share the same span but have different types.
   - Deduplication, if used, is by (span, type), not by text only.

4) Optional role folding for W2NER training:
   - Your current W2NER repo can handle nested/overlap spans,
     but it does NOT natively support same-span multi-type labels on the same head-tail cell.
   - If you want a "scheme 1" baseline, you can fold:
         startDate/endDate -> date
         startTime/endTime -> time
     by passing --fold_role_types.

Input
-----
Each input doc is expected to be:
[
  "doc_000000",
  {
    "sentences": [...],
    "ann_mspan2dranges": {...},
    "ann_mspan2guess_field": {...},
    ...
  }
]

Important compatibility note
----------------------------
This script supports two styles of mspan keys:

A) Plain mention text:
   ann_mspan2dranges = {"王启春": [[0, 11, 14]]}
   ann_mspan2guess_field = {"王启春": "person"}

B) Unique key used to preserve duplicate entities:
   ann_mspan2dranges = {"11月11日#0_13_19#startDate": [[0, 13, 19]]}
   ann_mspan2guess_field = {"11月11日#0_13_19#startDate": "startDate"}

For case B, the visible entity text is recovered from the key prefix before "#".

Usage
-----
Directory mode:
    python convert_procnet_to_w2ner_strict.py \
        --input_dir /path/to/procnet \
        --output_dir /path/to/w2ner

Single file mode:
    python convert_procnet_to_w2ner_strict.py \
        --input /path/to/train.json \
        --output /path/to/train.w2ner.json

Recommended strict run:
    python convert_procnet_to_w2ner_strict.py \
        --input_dir /path/to/procnet \
        --output_dir /path/to/w2ner \
        --keep_duplicates \
        --keep_empty_sent \
        --strict_alignment \
        --write_manifest

Scheme-1 baseline (fold role labels):
    python convert_procnet_to_w2ner_strict.py \
        --input_dir /path/to/procnet \
        --output_dir /path/to/w2ner_folded \
        --keep_duplicates \
        --fold_role_types
"""

import os
import json
import argparse
from collections import defaultdict


ROLE_FOLD_MAP = {
    "startDate": "date",
    "endDate": "date",
    "startTime": "time",
    "endTime": "time",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_sentence_to_text(sentence):
    """
    Normalize a sentence to raw text string.
    Supports either:
      - str
      - list[str] / char list
    """
    if isinstance(sentence, str):
        return sentence
    if isinstance(sentence, list):
        return "".join(sentence)
    raise TypeError(f"Unsupported sentence type: {type(sentence)}")


def normalize_sentence_to_char_list(sentence):
    """
    W2NER side uses character-level list by default.
    This keeps drange offsets aligned with the sentence coordinate space.
    """
    if isinstance(sentence, list):
        return sentence
    return list(sentence)


def build_sample_id(doc_id, sent_id):
    return f"{doc_id}__sent_{sent_id}"


def maybe_fold_type(type_name, fold_role_types=False, custom_fold_map=None):
    if not fold_role_types:
        return type_name
    fold_map = custom_fold_map or ROLE_FOLD_MAP
    return fold_map.get(type_name, type_name)


def recover_entity_text_from_mspan_key(mspan_key):
    """
    Recover visible mention text from ProcNet mspan key.

    Supports:
      - plain text key: "王启春"
      - unique key: "11月11日#0_13_19#startDate"

    Caveat:
      if original entity text itself contains "#", this simple split would be ambiguous.
      In your current upstream script, the generated unique key format is:
          f"{entity_text}#{sent_idx}_{start}_{end}#{entity_type}"
      so this recovery is compatible with that pipeline.
    """
    if not isinstance(mspan_key, str):
        return str(mspan_key)

    if "#" in mspan_key:
        return mspan_key.split("#", 1)[0]
    return mspan_key


def warn_or_raise(message, strict_alignment=True):
    if strict_alignment:
        raise ValueError(message)
    print(f"[WARN] {message}")


def validate_span_against_text(doc_id, sent_idx, sentence_text, start, end, mention_text, strict_alignment=True):
    """
    Strict alignment check:
        sentence_text[start:end] == mention_text
    """
    if not isinstance(start, int) or not isinstance(end, int):
        warn_or_raise(
            f"{doc_id} sent={sent_idx}: non-int span [{start}, {end}] for mention={mention_text!r}",
            strict_alignment=strict_alignment,
        )
        return False

    if not (0 <= start < end <= len(sentence_text)):
        warn_or_raise(
            f"{doc_id} sent={sent_idx}: invalid span [{start}, {end}) with sent_len={len(sentence_text)} "
            f"for mention={mention_text!r}",
            strict_alignment=strict_alignment,
        )
        return False

    extracted = sentence_text[start:end]
    if extracted != mention_text:
        warn_or_raise(
            f"{doc_id} sent={sent_idx}: span-text mismatch: "
            f"expected={mention_text!r}, got={extracted!r}, span=[{start}, {end})",
            strict_alignment=strict_alignment,
        )
        return False

    return True


def build_entities_for_doc(
    procnet_doc,
    *,
    fold_role_types=False,
    custom_fold_map=None,
    strict_alignment=True,
):
    """
    Convert one ProcNet doc into:
        sent_id -> [entity_verbose, ...]
    """
    doc_id, data = procnet_doc
    sentences = data.get("sentences", [])
    ann_mspan2dranges = data.get("ann_mspan2dranges", {})
    ann_mspan2guess_field = data.get("ann_mspan2guess_field", {})

    if not isinstance(sentences, list):
        raise TypeError(f"{doc_id}: 'sentences' must be list")
    if not isinstance(ann_mspan2dranges, dict):
        raise TypeError(f"{doc_id}: 'ann_mspan2dranges' must be dict")
    if not isinstance(ann_mspan2guess_field, dict):
        raise TypeError(f"{doc_id}: 'ann_mspan2guess_field' must be dict")

    sent_entities = defaultdict(list)

    for mspan_key, dranges in ann_mspan2dranges.items():
        orig_type = ann_mspan2guess_field.get(mspan_key)
        if orig_type is None:
            warn_or_raise(
                f"{doc_id}: missing type in ann_mspan2guess_field for mspan_key={mspan_key!r}",
                strict_alignment=strict_alignment,
            )
            continue

        folded_type = maybe_fold_type(
            orig_type,
            fold_role_types=fold_role_types,
            custom_fold_map=custom_fold_map,
        )

        mention_text = recover_entity_text_from_mspan_key(mspan_key)

        if not isinstance(dranges, list):
            warn_or_raise(
                f"{doc_id}: dranges for mspan_key={mspan_key!r} must be list, got {type(dranges)}",
                strict_alignment=strict_alignment,
            )
            continue

        for dr in dranges:
            if not (isinstance(dr, list) and len(dr) == 3):
                warn_or_raise(
                    f"{doc_id}: bad drange for mspan_key={mspan_key!r}: {dr!r}",
                    strict_alignment=strict_alignment,
                )
                continue

            sent_idx, start, end = dr

            if not isinstance(sent_idx, int) or not (0 <= sent_idx < len(sentences)):
                warn_or_raise(
                    f"{doc_id}: sent_idx out of range for mspan_key={mspan_key!r}: {dr!r}",
                    strict_alignment=strict_alignment,
                )
                continue

            sentence_text = normalize_sentence_to_text(sentences[sent_idx])

            if not validate_span_against_text(
                doc_id=doc_id,
                sent_idx=sent_idx,
                sentence_text=sentence_text,
                start=start,
                end=end,
                mention_text=mention_text,
                strict_alignment=strict_alignment,
            ):
                continue

            sent_entities[sent_idx].append(
                {
                    "start": start,
                    "end": end,  # right-open interval [start, end)
                    "text": mention_text,
                    "type_name": folded_type,
                    "orig_type_name": orig_type,
                    "mspan_key": mspan_key,
                    "doc_id": doc_id,
                    "sent_idx": sent_idx,
                }
            )

    # Stable sort for reproducibility
    for sent_idx in list(sent_entities.keys()):
        sent_entities[sent_idx].sort(
            key=lambda x: (
                x["start"],
                x["end"],
                x["type_name"],
                x["orig_type_name"],
                x["text"],
                x["mspan_key"],
            )
        )

    return sent_entities


def deduplicate_entities(entities, keep_duplicates=True):
    """
    Keep duplicates by default.

    IMPORTANT:
    - same span + different type should be preserved
    - deduplication key is (start, end, type_name, orig_type_name, text)
    - never deduplicate by text only
    """
    if keep_duplicates:
        return entities

    seen = set()
    output = []
    for ent in entities:
        key = (
            ent["start"],
            ent["end"],
            ent["type_name"],
            ent["orig_type_name"],
            ent["text"],
        )
        if key not in seen:
            seen.add(key)
            output.append(ent)
    return output


def entities_to_w2ner_ner(entities):
    ner = []
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        if not (isinstance(start, int) and isinstance(end, int) and end > start):
            continue
        ner.append(
            {
                "index": list(range(start, end)),
                "type": ent["type_name"],
            }
        )
    return ner


def convert_one_doc(
    procnet_doc,
    *,
    keep_empty_sent=True,
    keep_duplicates=True,
    fold_role_types=False,
    custom_fold_map=None,
    strict_alignment=True,
    preserve_orig_type_in_entities=True,
):
    """
    Input:
        procnet_doc = [
            doc_id,
            {
                "sentences": [...],
                "ann_mspan2dranges": {...},
                "ann_mspan2guess_field": {...},
                ...
            }
        ]

    Output:
        [
            {
                "sample_id": ...,
                "doc_id": ...,
                "sent_id": 0,
                "text": "...",
                "sentence": [...],
                "ner": [...],
                "entities": [...]
            },
            ...
        ]
    """
    doc_id, data = procnet_doc
    sentences = data.get("sentences", [])

    sent_entities = build_entities_for_doc(
        procnet_doc,
        fold_role_types=fold_role_types,
        custom_fold_map=custom_fold_map,
        strict_alignment=strict_alignment,
    )

    results = []

    for sent_id, sentence in enumerate(sentences):
        sent_text = normalize_sentence_to_text(sentence)
        sent_chars = normalize_sentence_to_char_list(sent_text)
        entities = sent_entities.get(sent_id, [])
        entities = deduplicate_entities(entities, keep_duplicates=keep_duplicates)

        if not preserve_orig_type_in_entities:
            slim_entities = []
            for ent in entities:
                slim_entities.append(
                    {
                        "start": ent["start"],
                        "end": ent["end"],
                        "text": ent["text"],
                        "type_name": ent["type_name"],
                        "doc_id": ent["doc_id"],
                        "sent_idx": ent["sent_idx"],
                    }
                )
            entities_out = slim_entities
        else:
            entities_out = entities

        # Additional strict consistency check:
        # Because sentence is char-level, it should reconstruct back to text exactly.
        if "".join(sent_chars) != sent_text:
            warn_or_raise(
                f"{doc_id} sent={sent_id}: sentence char list does not reconstruct to original text",
                strict_alignment=strict_alignment,
            )

        ner = entities_to_w2ner_ner(entities)

        if (not keep_empty_sent) and len(ner) == 0:
            continue

        results.append(
            {
                "sample_id": build_sample_id(doc_id, sent_id),
                "doc_id": doc_id,
                "sent_id": sent_id,
                "text": sent_text,
                "sentence": sent_chars,
                "ner": ner,
                "entities": entities_out,
            }
        )

    return results


def build_manifest(converted):
    """
    Helpful debug summary.
    """
    num_docs = len({x["doc_id"] for x in converted})
    num_samples = len(converted)
    num_nonempty = sum(1 for x in converted if x["ner"])
    num_entities = sum(len(x["ner"]) for x in converted)

    return {
        "num_docs": num_docs,
        "num_samples": num_samples,
        "num_nonempty_samples": num_nonempty,
        "num_entities": num_entities,
        "note": "Manifest is for debugging only; do not feed it into W2NER training.",
    }


def convert_split(
    input_path,
    output_path,
    *,
    keep_empty_sent=True,
    keep_duplicates=True,
    fold_role_types=False,
    strict_alignment=True,
    preserve_orig_type_in_entities=True,
    write_manifest=False,
):
    raw_data = load_json(input_path)
    converted = []

    for one_doc in raw_data:
        converted.extend(
            convert_one_doc(
                one_doc,
                keep_empty_sent=keep_empty_sent,
                keep_duplicates=keep_duplicates,
                fold_role_types=fold_role_types,
                strict_alignment=strict_alignment,
                preserve_orig_type_in_entities=preserve_orig_type_in_entities,
            )
        )

    dump_json(converted, output_path)

    if write_manifest:
        manifest_path = output_path + ".manifest.json"
        dump_json(build_manifest(converted), manifest_path)

    print(f"[OK] {input_path} -> {output_path}, num_samples={len(converted)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ProcNet raw doc-level annotations into strict-alignment W2NER sentence-level samples."
    )

    # Keep compatibility with the old directory-based interface,
    # while also supporting single-file mode.
    parser.add_argument("--input_dir", help="ProcNet raw data directory, containing train/dev/test.json")
    parser.add_argument("--output_dir", help="Output W2NER directory")
    parser.add_argument("--input", help="Path to one ProcNet raw JSON file")
    parser.add_argument("--output", help="Path to one W2NER JSON file")

    # Behavior flags
    parser.add_argument("--drop_empty_sent", action="store_true", help="Drop sentences without entities")
    parser.add_argument(
        "--keep_empty_sent",
        action="store_true",
        help="Explicitly keep empty sentences (wins over --drop_empty_sent if both are set)",
    )
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Keep duplicate entities. Recommended. Same-span different-type entities should usually be preserved.",
    )
    parser.add_argument(
        "--strict_alignment",
        action="store_true",
        help="Enable strict alignment checks and raise on mismatch. Recommended for debugging / first full run.",
    )
    parser.add_argument(
        "--no_strict_alignment",
        action="store_true",
        help="Disable strict alignment raises; only warn and skip bad spans.",
    )
    parser.add_argument(
        "--fold_role_types",
        action="store_true",
        help="Fold startDate/endDate->date and startTime/endTime->time for W2NER scheme-1 baseline.",
    )
    parser.add_argument(
        "--slim_entities",
        action="store_true",
        help="Do not keep orig_type_name/mspan_key in entities; keep only a slimmer entity record.",
    )
    parser.add_argument(
        "--write_manifest",
        action="store_true",
        help="Write a small manifest JSON next to each output file.",
    )

    args = parser.parse_args()

    # Defaults chosen for strict-alignment workflow:
    # - keep empty sentences
    # - keep duplicates
    # - strict alignment on
    if args.keep_empty_sent:
        keep_empty_sent = True
    else:
        keep_empty_sent = not args.drop_empty_sent

    keep_duplicates = True if args.keep_duplicates else True
    # In the strict-alignment version, preserving duplicates is the safer default.

    strict_alignment = True
    if args.no_strict_alignment:
        strict_alignment = False
    elif args.strict_alignment:
        strict_alignment = True

    preserve_orig_type_in_entities = not args.slim_entities

    # Single-file mode
    if args.input and args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        convert_split(
            args.input,
            args.output,
            keep_empty_sent=keep_empty_sent,
            keep_duplicates=keep_duplicates,
            fold_role_types=args.fold_role_types,
            strict_alignment=strict_alignment,
            preserve_orig_type_in_entities=preserve_orig_type_in_entities,
            write_manifest=args.write_manifest,
        )
        print("[DONE] ProcNet raw -> W2NER format")
        return

    # Directory mode
    if args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        found_any = False
        for split in ["train", "dev", "test"]:
            in_path = os.path.join(args.input_dir, f"{split}.json")
            if not os.path.exists(in_path):
                continue

            found_any = True
            out_path = os.path.join(args.output_dir, f"{split}.json")
            convert_split(
                in_path,
                out_path,
                keep_empty_sent=keep_empty_sent,
                keep_duplicates=keep_duplicates,
                fold_role_types=args.fold_role_types,
                strict_alignment=strict_alignment,
                preserve_orig_type_in_entities=preserve_orig_type_in_entities,
                write_manifest=args.write_manifest,
            )

        if not found_any:
            raise FileNotFoundError(
                f"No split file found in {args.input_dir}. Expected at least one of train.json/dev.json/test.json"
            )

        print("[DONE] ProcNet raw -> W2NER format")
        return

    raise ValueError(
        "Use either --input/--output for single-file mode, or --input_dir/--output_dir for directory mode."
    )


if __name__ == "__main__":
    main()
