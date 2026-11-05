from collections.abc import Collection
from dataclasses import dataclass


@dataclass
class DocileItemAnnotations:
    """Select item-value boxes while retaining document-local item membership."""

    target_fields: Collection[str]

    def iter_annotations(self, json_data):
        for item in json_data.get("items", []):
            item_id = int(item["line_item_id"])
            for name, field in item["fields"].items():
                path = f"items.{name}"
                if field is None or path not in self.target_fields:
                    continue
                for occurrence, (bbox, page, expected) in enumerate(
                    zip(
                        field["bbox"],
                        field["pages"],
                        field["source_values"],
                        strict=True,
                    )
                ):
                    yield {
                        "field": path,
                        "line_item_id": item_id,
                        "occurrence": occurrence,
                        "bbox": bbox,
                        "page": int(page),
                        "expected": expected,
                    }
