from collections import defaultdict
from dataclasses import dataclass

from app.extraction.bert.enums import EntityType
from app.extraction.bert.models import (
    DocumentOutput,
    InvoiceData,
    PaymentInstructions,
    PredictedEntity,
    SubtotalData,
)


@dataclass(frozen=True, slots=True)
class DocumentOutputFormatter:
    """Convert flat predicted entities into the document output schema."""

    def format(self, entities: list[PredictedEntity]) -> DocumentOutput:
        """Apply each document field's aggregation rule to predicted entities."""

        grouped = self._group_entities(entities)

        return DocumentOutput(
            invoice=InvoiceData(
                invoice_number=self._first(grouped, EntityType.INVOICE_NUMBER),
                invoice_date=self._first(grouped, EntityType.INVOICE_DATE),
                invoice_due_date=self._first(grouped, EntityType.INVOICE_DUE_DATE),
                seller_name=self._first(grouped, EntityType.SELLER_NAME),
                seller_address=self._multiline(grouped, EntityType.SELLER_ADDRESS),
                client_name=self._first(grouped, EntityType.CLIENT_NAME),
                client_address=self._multiline(grouped, EntityType.CLIENT_ADDRESS),
            ),
            subtotal=SubtotalData(
                tax=self._last(grouped, EntityType.TAX),
                total=self._last(grouped, EntityType.TOTAL),
                discount=self._last(grouped, EntityType.DISCOUNT),
            ),
            payment_instructions=PaymentInstructions(
                due_date=self._first(grouped, EntityType.PAYMENT_DUE_DATE),
                account_number=self._first(grouped, EntityType.ACCOUNT_NUMBER),
                bank_name=self._first(grouped, EntityType.BANK_NAME),
                payment_method=self._first(grouped, EntityType.PAYMENT_METHOD),
            ),
            items=None,
        )

    def format_json(
        self,
        entities: list[PredictedEntity],
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize formatted predictions while excluding absent optional fields."""

        return self.format(entities).model_dump_json(
            exclude_none=True,
            indent=indent,
        )

    @staticmethod
    def _group_entities(
        entities: list[PredictedEntity],
    ) -> dict[EntityType, list[PredictedEntity]]:
        grouped: dict[EntityType, list[PredictedEntity]] = defaultdict(list)

        for entity in sorted(entities, key=lambda item: item.start):
            grouped[entity.entity].append(entity)

        return dict(grouped)

    @staticmethod
    def _first(
        grouped: dict[EntityType, list[PredictedEntity]],
        entity_type: EntityType,
    ) -> str | None:
        candidates = grouped.get(entity_type, [])
        return candidates[0].text.strip() if candidates else None

    @staticmethod
    def _last(
        grouped: dict[EntityType, list[PredictedEntity]],
        entity_type: EntityType,
    ) -> str | None:
        candidates = grouped.get(entity_type, [])
        return candidates[-1].text.strip() if candidates else None

    @staticmethod
    def _multiline(
        grouped: dict[EntityType, list[PredictedEntity]],
        entity_type: EntityType,
    ) -> str | None:
        values = [
            candidate.text.strip()
            for candidate in grouped.get(entity_type, [])
            if candidate.text.strip()
        ]
        return "\n".join(values) if values else None
