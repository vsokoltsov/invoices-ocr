from dataclasses import dataclass, field

from app.extraction.bert.preprocess import InvoicePreprocessor
from app.extraction.bert.spans import assign_candidate_labels
from app.extraction.enums import ITEM_FIELD_TO_ENTITY
from app.extraction.schema import InvoiceLabelSchema


@dataclass
class ItemPreprocessor(InvoicePreprocessor):
    """Build item-only field spans; item IDs are metadata, not BIO classes."""

    label_schema: InvoiceLabelSchema = field(
        default_factory=lambda: InvoiceLabelSchema(ITEM_FIELD_TO_ENTITY)
    )

    def _assign_candidates(self, row):
        """Select item annotations without changing item membership.

        Args:
            row: Document with decoded OCR and annotation data.

        Returns:
            Candidate word labels and item annotation metadata.
        """
        return assign_candidate_labels(
            row,
            target_fields=self.label_schema.field_to_entity.keys(),
            line_items_only=True,
        )
