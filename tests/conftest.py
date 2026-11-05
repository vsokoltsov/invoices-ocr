import json
from zipfile import ZipFile

import pytest


@pytest.fixture
def docile_archive(tmp_path):
    """A real archive with official splits and deliberately unsorted OCR words."""
    root = tmp_path / "docile"
    root.mkdir()
    archive_path = root / "annotated-trainval.zip"
    ids = [f"doc-{index}" for index in range(12)]
    with ZipFile(archive_path, "w") as archive:
        for split, documents in {
            "train": ids[:10],
            "val": ids[10:],
            "trainval": ids,
        }.items():
            archive.writestr(f"{split}.json", json.dumps(documents))
        for document_id in ids:
            fields = [
                {
                    "fieldtype": "vendor_name",
                    "page": 0,
                    "bbox": [0.1, 0.1, 0.6, 0.2],
                    "text": "Acme Corporation",
                }
            ]
            item = {
                "fieldtype": "line_item_description",
                "page": 0,
                "bbox": [0.1, 0.4, 0.6, 0.5],
                "text": "prefix",
                "line_item_id": 1,
            }
            annotation = {
                "field_extractions": fields,
                "line_item_extractions": [item],
                "line_item_headers": [item],
                "metadata": {"language": "eng"},
            }
            words = [
                {
                    "value": "Corporation",
                    "geometry": [[0.3, 0.1], [0.6, 0.2]],
                    "snapped_geometry": [[0.31, 0.11], [0.59, 0.19]],
                    "confidence": 0.9,
                },
                {"value": "Acme", "geometry": [[0.1, 0.1], [0.25, 0.2]]},
            ]
            ocr = {
                "pages": [{"page_idx": 0, "blocks": [{"lines": [{"words": words}]}]}]
            }
            archive.writestr(f"annotations/{document_id}.json", json.dumps(annotation))
            archive.writestr(f"ocr/{document_id}.json", json.dumps(ocr))
    return archive_path
