import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from app.config import Settings
from app.datasets.docile.converter import DocileAnnotationConverter
from app.datasets.docile.corrections import DocileAnnotationCorrections
from app.datasets.docile.downloader import DocileDatasetDownloader
from app.datasets.docile.loader import DocileDatasetLoader
from app.datasets.docile.preparation import DocileInvoicesPreparation
from app.extraction.bert.pytorch.docile.builder import DocileRecordBuilder
from commands import cli


def correction(**changes):
    return {
        "document_id": "doc-0",
        "page": 0,
        "expected_fieldtype": "vendor_name",
        "expected_text": "Acme Corporation",
        "replacement_fieldtype": "vendor_address",
        "reason": "Manually verified fixture annotation.",
        **changes,
    }


def write_rules(tmp_path, rules):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return path


def annotation():
    return {
        "field_extractions": [
            {
                "fieldtype": "vendor_name",
                "page": 0,
                "text": "Acme Corporation",
                "bbox": [0.1, 0.1, 0.6, 0.2],
            }
        ],
        "metadata": {"source": "fixture"},
    }


def test_corrections_change_only_field_type_and_preserve_originals(tmp_path):
    rules = DocileAnnotationCorrections.from_file(write_rules(tmp_path, [correction()]))
    original = annotation()
    snapshot = deepcopy(original)
    corrected = rules.apply("doc-0", original)
    expected = deepcopy(original)
    expected["field_extractions"][0]["fieldtype"] = "vendor_address"
    assert corrected == expected
    assert original == snapshot
    corrected["metadata"]["source"] = "modified"
    assert original == snapshot
    assert rules.apply("another-document", original) == original


@pytest.mark.parametrize("matches", [0, 2])
def test_stale_or_ambiguous_rules_fail_without_mutating_original(tmp_path, matches):
    rules = DocileAnnotationCorrections.from_file(write_rules(tmp_path, [correction()]))
    original = annotation()
    original["field_extractions"] *= matches
    snapshot = deepcopy(original)
    with pytest.raises(
        ValueError,
        match=f"doc-0, page 0, vendor_name: expected 1 annotation, found {matches}",
    ):
        rules.apply("doc-0", original)
    assert original == snapshot


def test_bbox_selector_disambiguates_repeated_text_and_duplicate_rules_fail(tmp_path):
    original = annotation()
    second = deepcopy(original["field_extractions"][0])
    second["bbox"] = [0.1, 0.3, 0.6, 0.4]
    original["field_extractions"].append(second)
    rule = correction(expected_bbox=second["bbox"])
    rules = DocileAnnotationCorrections.from_file(write_rules(tmp_path, [rule]))
    corrected = rules.apply("doc-0", original)
    assert [field["fieldtype"] for field in corrected["field_extractions"]] == [
        "vendor_name",
        "vendor_address",
    ]
    duplicates = DocileAnnotationCorrections.from_file(
        write_rules(tmp_path, [rule, rule])
    )
    with pytest.raises(ValueError, match="Multiple corrections"):
        duplicates.apply("doc-0", original)


@pytest.mark.parametrize(
    "rules",
    [
        {},
        [correction(page=-1)],
        [correction(page="0")],
        [correction(reason="")],
        [correction(replacement_text="Changed")],
        [correction(expected_bbox=[0, 0, 1])],
        [correction(expected_bbox=[0.6, 0.1, 0.1, 0.2])],
    ],
)
def test_invalid_manifest_is_rejected(tmp_path, rules):
    with pytest.raises(ValueError):
        DocileAnnotationCorrections.from_file(write_rules(tmp_path, rules))


def test_reviewed_address_rules_preserve_the_logo_annotation():
    path = Path(__file__).resolve().parents[3] / "config/docile_corrections.json"
    rules = DocileAnnotationCorrections.from_file(path)
    fields = [
        {
            "fieldtype": "vendor_name",
            "text": "CUMUL\nMUSKEGON",
            "page": 0,
            "bbox": [0.08, 0.06, 0.21, 0.10],
        }
    ]
    for rule in rules.rules:
        assert rule.expected_bbox is not None
        fields.append(
            {
                "fieldtype": rule.expected_fieldtype,
                "text": rule.expected_text,
                "page": rule.page,
                "bbox": list(rule.expected_bbox),
            }
        )
    original = {"field_extractions": fields}
    snapshot = deepcopy(original)
    corrected = rules.apply("26f0e3a78e26450b9b47dd81", original)
    converted = DocileAnnotationConverter().convert(corrected)
    assert original == snapshot
    assert corrected["field_extractions"][0] == original["field_extractions"][0]
    assert converted["invoice"]["seller_name"]["source_values"] == ["CUMUL\nMUSKEGON"]
    address = converted["invoice"]["seller_address"]
    assert address["pages"] == [0, 1]
    assert (
        address["source_values"] == ["3646 Momentum Place\nChicago, IL 60689-5336"] * 2
    )


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_cli_rebuilds_cached_data_with_corrections_and_audit(
    docile_archive,
    tmp_path,
    monkeypatch,
    capsys,
    extension,
):
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None, DATA_PATH="unused", DEBUG_ARTIFACTS_PATH="unused"
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    output = tmp_path / f"prepared.{extension}"
    args = [
        "datasets",
        "download",
        "docile",
        "--token",
        "dummy",
        "--destination-path",
        str(docile_archive.parent),
        "--output-path",
        str(output),
        "--minimum-archive-bytes",
        "0",
    ]
    cli.main(args)
    source_path = docile_archive.parent / "annotations/doc-0.json"
    original_bytes = source_path.read_bytes()
    manifest = write_rules(tmp_path, [correction()])
    monkeypatch.setattr(
        DocileDatasetDownloader,
        "download",
        Mock(side_effect=AssertionError("Already extracted")),
    )
    cli.main([*args, "--corrections-path", str(manifest)])
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(output)
    assert source_path.read_bytes() == original_bytes
    assert (
        DocileDatasetLoader(docile_archive.parent)
        .load("doc-0")
        .json_data["invoice"]["seller_name"]["value"]
        == "Acme Corporation"
    )
    frame = pd.read_parquet(output) if extension == "parquet" else pd.read_csv(output)
    audit = json.loads(
        frame.loc[frame.document_id == "doc-0", "annotation_corrections"].iloc[0]
    )
    assert audit[0]["reason"] == correction()["reason"]
    assert audit[0]["replacement_fieldtype"] == "vendor_address"
    frame["json_data"] = frame["json_data"].map(json.loads)
    if extension == "csv":
        frame["text_with_boxes"] = frame["text_with_boxes"].map(json.loads)
    records, rejected = DocileRecordBuilder().build(frame)
    assert not rejected
    corrected_record = next(
        record for record in records if record["document_id"] == "doc-0"
    )
    assert corrected_record["entities"][0]["label"] == "invoice.seller_address"
    prepared_bytes = output.read_bytes()
    write_rules(tmp_path, [correction(expected_text="Stale annotation text")])
    with pytest.raises(RuntimeError, match="doc-0.*expected 1 annotation, found 0"):
        cli.main([*args, "--corrections-path", str(manifest)])
    assert output.read_bytes() == prepared_bytes
    assert source_path.read_bytes() == original_bytes


def test_invalid_correction_path_fails_before_download(tmp_path, monkeypatch):
    download = Mock(side_effect=AssertionError("Must not download"))
    monkeypatch.setattr(DocileDatasetDownloader, "download", download)
    preparation = DocileInvoicesPreparation(
        DocileDatasetDownloader("dummy", target_dir=tmp_path / "raw", unzip=True),
        tmp_path / "output.csv",
        corrections_path=tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError, match="missing.json"):
        preparation.run()
    download.assert_not_called()
