from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class DocileAnnotationConverter:
    FIELD_MAPPING: ClassVar[dict[str, tuple[str, str]]] = {
        "document_id": ("invoice", "invoice_number"),
        "date_issue": ("invoice", "invoice_date"),
        "date_due": ("invoice", "due_date"),
        "vendor_name": ("invoice", "seller_name"),
        "vendor_address": ("invoice", "seller_address"),
        "customer_billing_name": ("invoice", "client_name"),
        "customer_billing_address": ("invoice", "client_address"),
        "account_num": ("payment_instructions", "account_number"),
        "payment_terms": ("payment_instructions", "payment_terms"),
        "amount_total_gross": ("subtotal", "total"),
        "amount_total_tax": ("subtotal", "tax"),
    }

    ITEM_MAPPING: ClassVar[dict[str, str]] = {
        "line_item_description": "description",
        "line_item_quantity": "quantity",
        "line_item_units_of_measure": "units_of_measure",
        "line_item_code": "code",
        "line_item_unit_price_gross": "unit_price",
        "line_item_amount_gross": "total_price",
    }

    @staticmethod
    def _add_field(
        destination: dict,
        name: str,
        field: dict,
        *,
        join_fragments: bool = False,
    ) -> None:
        entry = destination.setdefault(
            name,
            {
                "value": field["text"],
                "bbox": [],
                "pages": [],
                "source_values": [],
            },
        )

        entry["bbox"].append(list(field["bbox"]))
        entry["pages"].append(field["page"])
        entry["source_values"].append(field["text"])

        if join_fragments:
            entry["value"] = "\n".join(dict.fromkeys(entry["source_values"]))

    def convert(self, annotation: dict[str, Any]) -> dict[str, Any]:
        """Convert annotations without discarding unmapped fields or locations."""
        result: dict[str, Any] = {
            "invoice": {},
            "payment_instructions": {},
            "subtotal": {},
            "other_fields": {},
            "items": [],
            "line_item_headers": {},
            "metadata": annotation.get("metadata", {}),
        }

        for field in annotation.get("field_extractions", []):
            fieldtype = field["fieldtype"]

            section, name = self.FIELD_MAPPING.get(
                fieldtype,
                ("other_fields", fieldtype),
            )

            self._add_field(result[section], name, field)

        items_by_id = {}

        for field in annotation.get("line_item_extractions", []):
            item_id = field["line_item_id"]

            item = items_by_id.setdefault(
                item_id,
                {
                    "line_item_id": item_id,
                    "fields": {},
                },
            )

            fieldtype = field["fieldtype"]
            name = self.ITEM_MAPPING.get(
                fieldtype,
                fieldtype.removeprefix("line_item_"),
            )

            self._add_field(
                item["fields"],
                name,
                field,
                join_fragments=name == "description",
            )

        result["items"] = list(items_by_id.values())

        for field in annotation.get("line_item_headers", []):
            fieldtype = field["fieldtype"]
            name = self.ITEM_MAPPING.get(
                fieldtype,
                fieldtype.removeprefix("line_item_"),
            )

            self._add_field(
                result["line_item_headers"],
                name,
                field,
            )

        return result
