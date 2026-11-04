from collections import Counter
from collections.abc import Collection
from typing import TypedDict


class CandidateLabel(TypedDict):
    label: str
    occurrence: int
    annotation_id: int


class CandidateWord(TypedDict):
    """Typed assignment fields added to the original OCR word metadata."""

    ocr_index: int
    candidates: list[CandidateLabel]


SECTIONS = ("invoice", "payment_instructions", "subtotal")
TARGET_FIELDS = {
    "invoice.client_name",
    "invoice.client_address",
    "invoice.seller_name",
    "invoice.seller_address",
    "invoice.invoice_number",
    "invoice.invoice_date",
    "invoice.due_date",
    "payment_instructions.payment_terms",
    "payment_instructions.account_number",
    "subtotal.tax",
    "subtotal.total",
}

NAME_TO_ADDRESS = {
    "invoice.client_name": "invoice.client_address",
    "invoice.seller_name": "invoice.seller_address",
}


def word_inside_box(word, bbox, page):
    left, top, right, bottom = bbox
    word_left, word_top, word_right, word_bottom = word["bbox"]

    center_x = (word_left + word_right) / 2
    center_y = (word_top + word_bottom) / 2

    return (
        word["page"] == page and left <= center_x <= right and top <= center_y <= bottom
    )


def find_spans(row):
    ocr_text = row["ocr_text"]
    text_with_boxes = row["text_with_boxes"]
    spans = []

    for section in SECTIONS:
        for name, field in row["json_data"].get(section, {}).items():
            if field is None:
                continue

            path = f"{section}.{name}"
            if path not in TARGET_FIELDS:
                continue

            for occurrence, (bbox, page) in enumerate(
                zip(field["bbox"], field["pages"], strict=True)
            ):
                previous_index = None

                for word_index, word in enumerate(text_with_boxes):
                    if not word_inside_box(word, bbox, page):
                        continue

                    if previous_index is not None and word_index == previous_index + 1:
                        spans[-1]["end"] = int(word["end"])
                    else:
                        spans.append(
                            {
                                "start": int(word["start"]),
                                "end": int(word["end"]),
                                "label": path,
                                "page": int(page),
                                "occurrence": occurrence,
                            }
                        )

                    previous_index = word_index

    spans.sort(key=lambda span: span["start"])

    for span in spans:
        span["text"] = ocr_text[span["start"] : span["end"]]

    return spans


def assign_candidate_labels(row, target_fields: Collection[str] | None = None):
    """Project selected annotation fields onto OCR words using their boxes."""
    if target_fields is None:
        target_fields = TARGET_FIELDS
    words = row["text_with_boxes"]

    # Preserve the original word order and metadata.
    word_candidates: list[CandidateWord] = [
        {
            **word,
            "ocr_index": index,
            "candidates": [],
        }
        for index, word in enumerate(words)
    ]

    annotations = []

    for section in SECTIONS:
        for name, field in row["json_data"].get(section, {}).items():
            path = f"{section}.{name}"

            if field is None or path not in target_fields:
                continue

            for occurrence, (bbox, page) in enumerate(
                zip(field["bbox"], field["pages"], strict=True)
            ):
                annotation_id = len(annotations)
                matched_words = 0

                for word_index, word in enumerate(words):
                    if not word_inside_box(word, bbox, page):
                        continue

                    word_candidates[word_index]["candidates"].append(
                        {
                            "label": path,
                            "occurrence": occurrence,
                            "annotation_id": annotation_id,
                        }
                    )

                    matched_words += 1

                annotations.append(
                    {
                        "annotation_id": annotation_id,
                        "field": path,
                        "occurrence": occurrence,
                        "page": int(page),
                        "expected": field["source_values"][occurrence],
                        "matched_words": matched_words,
                    }
                )

    return word_candidates, annotations


def resolve_name_address_nesting(word_candidates):
    resolved_words = []

    for word in word_candidates:
        labels = {candidate["label"] for candidate in word["candidates"]}

        excluded_addresses = {
            address for name, address in NAME_TO_ADDRESS.items() if name in labels
        }

        resolved_words.append(
            {
                **word,
                "candidates": [
                    candidate.copy()
                    for candidate in word["candidates"]
                    if candidate["label"] not in excluded_addresses
                ],
            }
        )

    return resolved_words


def deduplicate_candidate_labels(resolved_words):
    deduplicated_words = []

    for word in resolved_words:
        by_label = {}

        for candidate in word["candidates"]:
            label = candidate["label"]

            grouped = by_label.setdefault(
                label,
                {
                    "label": label,
                    "sources": [],
                },
            )

            source = {
                "annotation_id": candidate["annotation_id"],
                "occurrence": candidate["occurrence"],
            }

            if source not in grouped["sources"]:
                grouped["sources"].append(source)

        deduplicated_words.append(
            {
                **word,
                "candidates": list(by_label.values()),
            }
        )

    return deduplicated_words


def build_spans(ocr_text, deduplicated_words):
    spans = []
    ignored_regions = []
    previous_key = None
    previous_index = None

    for word in deduplicated_words:
        candidates = word["candidates"]

        if len(candidates) != 1:
            if len(candidates) > 1:
                ignored_regions.append(
                    {
                        "start": int(word["start"]),
                        "end": int(word["end"]),
                        "page": int(word["page"]),
                    }
                )

            previous_key = None
            previous_index = None
            continue

        candidate = candidates[0]

        source_key = tuple(
            sorted(
                (source["annotation_id"], source["occurrence"])
                for source in candidate["sources"]
            )
        )

        key = (
            candidate["label"],
            int(word["page"]),
            source_key,
        )

        start = int(word["start"])
        end = int(word["end"])

        if (
            previous_index is not None
            and key == previous_key
            and word["ocr_index"] == previous_index + 1
            and not ocr_text[spans[-1]["end"] : start].strip()
        ):
            spans[-1]["end"] = end
        else:
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": candidate["label"],
                    "page": int(word["page"]),
                    "sources": [source.copy() for source in candidate["sources"]],
                }
            )

        previous_key = key
        previous_index = word["ocr_index"]

    for span in spans:
        span["text"] = ocr_text[span["start"] : span["end"]]

    return spans, ignored_regions


def find_remaining_issues(deduplicated_words, annotations):
    issues = []
    supervised_counts = Counter()

    if not annotations:
        return [{"status": "no_selected_annotations"}]

    for word in deduplicated_words:
        candidates = word["candidates"]

        if len(candidates) > 1:
            issues.append(
                {
                    "status": "overlap",
                    "page": int(word["page"]),
                    "start": int(word["start"]),
                    "end": int(word["end"]),
                    "text": word["text"],
                    "labels": [candidate["label"] for candidate in candidates],
                }
            )

        elif len(candidates) == 1:
            # Count this word once for each supporting annotation.
            supervised_counts.update(
                {source["annotation_id"] for source in candidates[0]["sources"]}
            )

    for annotation in annotations:
        annotation_id = annotation["annotation_id"]

        if annotation["matched_words"] == 0:
            status = "missing_span"
        elif supervised_counts[annotation_id] == 0:
            status = "no_supervised_words"
        else:
            continue

        issues.append(
            {
                **annotation,
                "status": status,
                "supervised_words": supervised_counts[annotation_id],
            }
        )

    return issues
