from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from app.datasets.docile.models import BBox


class AnnotationCorrection(BaseModel):
    """Select one original field and change its type without changing geometry."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    document_id: str = Field(min_length=1)
    page: int = Field(ge=0)
    expected_fieldtype: str = Field(min_length=1)
    expected_text: str
    replacement_fieldtype: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    expected_bbox: BBox | None = None

    @field_validator("expected_bbox")
    @classmethod
    def validate_bbox(cls, bbox: BBox | None) -> BBox | None:
        if bbox is not None:
            left, top, right, bottom = bbox
            if not (
                all(isfinite(value) and 0 <= value <= 1 for value in bbox)
                and left < right
                and top < bottom
            ):
                raise ValueError("expected_bbox must be a valid normalized box")
        return bbox

    def matches(self, field: dict) -> bool:
        return (
            field.get("fieldtype") == self.expected_fieldtype
            and field.get("page") == self.page
            and field.get("text") == self.expected_text
            and (
                self.expected_bbox is None
                or tuple(field.get("bbox", ())) == self.expected_bbox
            )
        )


@dataclass(frozen=True)
class DocileAnnotationCorrections:
    rules: tuple[AnnotationCorrection, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> "DocileAnnotationCorrections":
        """Read a JSON list; reject malformed rules and unknown attributes."""
        path = Path(path).expanduser()
        rules = TypeAdapter(list[AnnotationCorrection]).validate_json(path.read_bytes())
        return cls(tuple(rules))

    def for_document(self, document_id: str) -> tuple[AnnotationCorrection, ...]:
        return tuple(rule for rule in self.rules if rule.document_id == document_id)

    def apply(self, document_id: str, annotation: dict) -> dict:
        """Return a corrected copy, requiring one match per applicable rule.

        Match against original annotations so rule order cannot change selectors.
        Reject multiple corrections for the same field. Rules for documents not
        being loaded are left unused, which permits preparing individual splits.
        """
        fields = annotation.get("field_extractions", [])
        replacements = {}
        for rule in self.for_document(document_id):
            matches = [
                index for index, field in enumerate(fields) if rule.matches(field)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Correction for {document_id}, page {rule.page}, "
                    f"{rule.expected_fieldtype}: expected 1 annotation, found {len(matches)}"
                )
            index = matches[0]
            if index in replacements:
                raise ValueError(
                    f"Multiple corrections select annotation {index} in {document_id}"
                )
            replacements[index] = rule.replacement_fieldtype

        corrected = deepcopy(annotation)
        for index, fieldtype in replacements.items():
            corrected["field_extractions"][index]["fieldtype"] = fieldtype
        return corrected
