from copy import deepcopy

import pytest

from app.extraction.bert.pytorch.docile.spans import (
    assign_candidate_labels,
    build_spans,
    deduplicate_candidate_labels,
    find_remaining_issues,
    find_spans,
    resolve_name_address_nesting,
    word_inside_box,
)


@pytest.fixture
def row():
    return {
        "ocr_text": "Acme Road Acme",
        "text_with_boxes": [
            {
                "text": "Acme",
                "start": 0,
                "end": 4,
                "page": 0,
                "bbox": [0.1, 0.1, 0.2, 0.2],
            },
            {
                "text": "Road",
                "start": 5,
                "end": 9,
                "page": 0,
                "bbox": [0.1, 0.3, 0.2, 0.4],
            },
            {
                "text": "Acme",
                "start": 10,
                "end": 14,
                "page": 1,
                "bbox": [0.1, 0.1, 0.2, 0.2],
            },
        ],
        "json_data": {
            "invoice": {
                "seller_name": {
                    "bbox": [[0, 0, 0.3, 0.25]] * 2,
                    "pages": [1, 0],
                    "source_values": ["Acme", "Acme"],
                },
                "seller_address": {
                    "bbox": [[0, 0, 0.3, 0.5]],
                    "pages": [0],
                    "source_values": ["Acme Road"],
                },
            }
        },
    }


def test_projection_uses_pages_occurrences_and_name_priority_without_mutating(row):
    original = deepcopy(row)
    raw = find_spans(row)
    assert [
        (s["page"], s["occurrence"]) for s in raw if s["label"] == "invoice.seller_name"
    ] == [(0, 1), (1, 0)]
    assert not word_inside_box(row["text_with_boxes"][0], [0, 0, 1, 1], 1)
    candidates, annotations = assign_candidate_labels(row)
    resolved = deduplicate_candidate_labels(resolve_name_address_nesting(candidates))
    spans, ignored = build_spans(row["ocr_text"], resolved)
    assert [s["text"] for s in spans] == ["Acme", "Road", "Acme"]
    assert not ignored and not find_remaining_issues(resolved, annotations)
    assert row == original
    assert spans[0]["sources"] == [{"annotation_id": 1, "occurrence": 1}]


def test_duplicate_same_label_keeps_annotation_sources_and_real_conflicts_are_masked(
    row,
):
    name = row["json_data"]["invoice"]["seller_name"]
    name["bbox"].append(name["bbox"][1])
    name["pages"].append(0)
    name["source_values"].append("Acme")
    candidates, annotations = assign_candidate_labels(row, {"invoice.seller_name"})
    deduplicated = deduplicate_candidate_labels(candidates)
    spans, ignored = build_spans(row["ocr_text"], deduplicated)
    assert len(spans[0]["sources"]) == 2 and not ignored
    assert not find_remaining_issues(deduplicated, annotations)
    row["json_data"]["invoice"]["client_name"] = deepcopy(name)
    candidates, annotations = assign_candidate_labels(row)
    resolved = deduplicate_candidate_labels(resolve_name_address_nesting(candidates))
    spans, ignored = build_spans(row["ocr_text"], resolved)
    assert len(ignored) == 2
    assert [s["text"] for s in spans] == ["Road"]
    assert "overlap" in {
        issue["status"] for issue in find_remaining_issues(resolved, annotations)
    }


def test_missing_annotations_and_invalid_parallel_arrays_are_visible(row):
    row["json_data"]["invoice"]["seller_name"]["pages"] = [5, 6]
    candidates, annotations = assign_candidate_labels(row)
    resolved = deduplicate_candidate_labels(resolve_name_address_nesting(candidates))
    issues = find_remaining_issues(resolved, annotations)
    assert sum(issue["status"] == "missing_span" for issue in issues) == 2
    row["json_data"]["invoice"]["seller_name"]["pages"] = []
    with pytest.raises(ValueError, match="shorter"):
        assign_candidate_labels(row)


@pytest.mark.parametrize(
    "gap,page,source,expected_count",
    [(" ", 0, 0, 1), (" x ", 0, 0, 2), (" ", 1, 0, 2), (" ", 0, 1, 2)],
)
def test_merging_requires_whitespace_same_page_and_same_annotation(
    gap, page, source, expected_count
):
    text = f"Acme{gap}Road"
    words = [
        {
            "ocr_index": 0,
            "start": 0,
            "end": 4,
            "page": 0,
            "candidates": [
                {
                    "label": "invoice.seller_name",
                    "sources": [{"annotation_id": 0, "occurrence": 0}],
                }
            ],
        },
        {
            "ocr_index": 1,
            "start": 4 + len(gap),
            "end": len(text),
            "page": page,
            "candidates": [
                {
                    "label": "invoice.seller_name",
                    "sources": [{"annotation_id": source, "occurrence": 0}],
                }
            ],
        },
    ]
    spans, ignored = build_spans(text, words)
    assert len(spans) == expected_count and not ignored
    assert all(text[s["start"] : s["end"]] == s["text"] for s in spans)
