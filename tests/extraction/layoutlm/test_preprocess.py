import json
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from app.extraction.enums import FIELD_TO_ENTITY
from app.extraction.layoutlm.preprocess import AssignWords, InvoicePreprocessor
from app.extraction.schema import InvoiceLabelSchema


def test_preparation_orders_words_normalizes_boxes_and_keeps_pages(document_row):
    records = InvoicePreprocessor(FIELD_TO_ENTITY).prepare_document(document_row)
    assert records[0]["words"] == ["A", "BB", "C"]
    assert records[0]["boxes"] == [
        [100, 100, 200, 200],
        [300, 100, 500, 200],
        [100, 400, 200, 500],
    ]
    assert [record["page"] for record in records] == [0, 1]
    assert records[1]["words"] == ["Other"]


def test_schema_paths_and_entity_names_are_used(document_row):
    schema = InvoiceLabelSchema({"invoice.seller_name": "SELLER_NAME"})
    records = InvoicePreprocessor(
        schema.field_to_entity, schema.label2id
    ).prepare_document(document_row)
    assert [schema.id2label[index] for index in records[0]["labels"]] == [
        "B-SELLER_NAME",
        "I-SELLER_NAME",
        "O",
    ]
    assert records[0]["pdf_path"] == document_row.pdf_path
    assert records[1]["labels"] == [schema.label2id["O"]]


def test_custom_mapping_defines_selected_fields_and_labels(document_row):
    preprocessor = InvoicePreprocessor({"invoice.seller_name": "VENDOR"})
    records = preprocessor.prepare_document(document_row)
    assert preprocessor.label2id == {"O": 0, "B-VENDOR": 1, "I-VENDOR": 2}
    assert records[0]["labels"] == [1, 2, 0]
    assert records[1]["labels"] == [0]


@pytest.mark.parametrize("min_coverage", [0, -0.1, 1.1])
def test_invalid_coverage_threshold_is_rejected(min_coverage):
    with pytest.raises(ValueError, match="min_coverage"):
        InvoicePreprocessor(FIELD_TO_ENTITY, min_coverage=min_coverage)


def test_explicit_label_ids_must_include_all_selected_entities():
    with pytest.raises(ValueError, match="label2id must include"):
        InvoicePreprocessor({"invoice.seller_name": "VENDOR"}, label2id={"O": 0})


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("missing", "annotation_without_assigned_words"),
        ("overlap", "superseded_by_nested_annotation"),
        ("annotation_page", "Invalid annotation page"),
        ("word_page", "Invalid OCR page"),
        ("lengths", "Annotation lengths differ"),
        ("box", "Invalid normalized coordinates"),
        ("nonfinite_box", "Invalid box"),
        ("empty", "no OCR words"),
    ],
)
def test_bad_documents_are_rejected_as_a_whole(document_row, failure, reason):
    bad = deepcopy(document_row)
    bad.document_id = "bad"
    data = json.loads(bad.json_data)
    annotation = data["invoice"]["seller_name"]
    if failure == "missing":
        annotation["bbox"] = [[0.7, 0.7, 0.9, 0.9]]
    elif failure == "overlap":
        data["invoice"]["client_name"] = deepcopy(annotation)
    elif failure == "annotation_page":
        annotation["pages"] = [2]
    elif failure == "word_page":
        bad.text_with_boxes[0]["page"] = 2
    elif failure == "lengths":
        annotation["pages"] = []
    elif failure == "box":
        bad.text_with_boxes[0]["bbox"] = [-1, 0, 0.5, 0.5]
    elif failure == "nonfinite_box":
        bad.text_with_boxes[0]["bbox"] = [0, 0, float("nan"), 0.5]
    else:
        data["invoice"] = {}
        bad.text_with_boxes = []
    bad.json_data = json.dumps(data)
    records, rejected = InvoicePreprocessor(FIELD_TO_ENTITY).run(
        pd.DataFrame([vars(document_row), vars(bad)])
    )
    assert {record["document_id"] for record in records} == {document_row.document_id}
    assert rejected["document_id"].tolist() == ["bad"]
    assert reason in rejected.iloc[0]["reason"]


def test_bio_restarts_after_background_and_page_boundary(document_row):
    row = SimpleNamespace(**vars(document_row))
    data = json.loads(row.json_data)
    data["invoice"]["seller_name"] = {
        "pages": [0, 1],
        "bbox": [[0.1, 0.1, 0.6, 0.2]] * 2,
    }
    row.json_data = json.dumps(data)
    row.text_with_boxes = [
        {"text": "A", "start": 0, "page": 0, "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"text": "gap", "start": 2, "page": 0, "bbox": [0.7, 0.1, 0.9, 0.2]},
        {"text": "BB", "start": 6, "page": 0, "bbox": [0.3, 0.1, 0.5, 0.2]},
        {"text": "C", "start": 9, "page": 1, "bbox": [0.1, 0.1, 0.2, 0.2]},
    ]
    schema = InvoiceLabelSchema()
    records = InvoicePreprocessor(FIELD_TO_ENTITY).prepare_document(row)
    assert [schema.id2label[index] for index in records[0]["labels"]] == [
        "B-SELLER_NAME",
        "O",
        "B-SELLER_NAME",
    ]
    assert schema.id2label[records[1]["labels"][0]] == "B-SELLER_NAME"


def test_nested_boxes_assign_name_and_address_without_rejecting(document_row):
    data = json.loads(document_row.json_data)
    data["invoice"]["seller_name"]["bbox"] = [[0.1, 0.1, 0.2, 0.2]]
    data["invoice"]["seller_address"] = {
        "pages": [0],
        "bbox": [[0.1, 0.1, 0.6, 0.2]],
    }
    document_row.json_data = json.dumps(data)
    original = deepcopy(vars(document_row))
    preprocessor = InvoicePreprocessor(FIELD_TO_ENTITY)
    records, rejected = preprocessor.run(pd.DataFrame([vars(document_row)]))
    assert rejected.empty
    labels = [
        preprocessor.label2id[label]
        for label in ("B-SELLER_NAME", "B-SELLER_ADDRESS", "O")
    ]
    assert records[0]["labels"] == labels
    assert vars(document_row) == original


@pytest.mark.parametrize(
    ("wide_coverage", "small_coverage", "expected_owner"),
    [(1.0, 1.0, 0), (0.83, 0.81, 1), (0.89, 0.81, 0)],
)
def test_assignment_ranks_coverage_then_area_then_annotation_id(
    wide_coverage, small_coverage, expected_owner
):
    row = pd.Series(
        {
            "json_data": {"metadata": {"page_count": 1}},
            "text_with_boxes": [
                {"text": "word", "start": 0, "page": 0, "bbox": [0, 0.1, 1, 0.2]},
            ],
            "annotations": [
                {"page": 0, "box": [0, 0.1, wide_coverage, 0.2]},
                {"page": 0, "box": [0, 0.1, small_coverage, 0.2]},
            ],
        }
    )
    pages, matched = AssignWords(0.5).process(row)
    assert pages[0][0]["owner"] == expected_owner
    assert matched == {expected_owner}


def test_unassigned_nested_annotation_is_reported_as_rejected(document_row):
    data = json.loads(document_row.json_data)
    data["invoice"]["seller_address"] = {
        "pages": [0],
        "bbox": [[0, 0, 0.7, 0.3]],
    }
    document_row.json_data = json.dumps(data)
    records, rejected = InvoicePreprocessor(FIELD_TO_ENTITY).run(
        pd.DataFrame([vars(document_row)])
    )
    assert records == []
    assert rejected.document_id.tolist() == [document_row.document_id]
    assert "superseded_by_nested_annotation: SELLER_ADDRESS" in rejected.iloc[0].reason
