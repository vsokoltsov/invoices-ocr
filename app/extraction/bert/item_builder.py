from dataclasses import dataclass, field

from app.extraction.bert.builder import InvoiceRecordBuilder
from app.extraction.bert.schema import InvoiceLabelSchema
from app.extraction.bert.spans import assign_candidate_labels
from app.extraction.enums import ITEM_FIELD_TO_ENTITY


@dataclass
class ItemRecordBuilder(InvoiceRecordBuilder):
    """Build item-only field spans; item IDs are metadata, not BIO classes."""

    label_schema: InvoiceLabelSchema = field(
        default_factory=lambda: InvoiceLabelSchema(ITEM_FIELD_TO_ENTITY)
    )

    def _assign_candidates(self, row):
        return assign_candidate_labels(
            row,
            target_fields=self.label_schema.field_to_entity.keys(),
            line_items_only=True,
        )
