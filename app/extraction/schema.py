from dataclasses import dataclass, field

from app.extraction.enums import FIELD_TO_ENTITY


@dataclass(frozen=True)
class InvoiceLabelSchema:
    """Build deterministic BIO labels for invoice fields.

    BERT and LayoutLM use the same label order. Frozen attributes prevent
    reassignment, but the mapping objects themselves remain mutable.

    Attributes:
        field_to_entity: Ordered field-path to entity-name mapping copied at construction.
        entity_to_field: Reverse mapping for decoded predictions.
        labels: O followed by the B/I pair for each field in mapping order.
        label2id: BIO label to integer ID mapping.
        id2label: Integer ID to BIO label mapping.
    """

    field_to_entity: dict[str, str] = field(default_factory=lambda: FIELD_TO_ENTITY)

    entity_to_field: dict[str, str] = field(init=False)
    labels: tuple[str, ...] = field(init=False)
    label2id: dict[str, int] = field(init=False)
    id2label: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        # Copy the input so changes to the caller's mapping do not affect it.
        """Copy and validate field names, then derive BIO label mappings.

        Raises:
            ValueError: An entity name is duplicated, or a field/entity name is empty or untrimmed.
        """
        mapping = dict(self.field_to_entity)
        entities = list(mapping.values())

        if len(entities) != len(set(entities)):
            raise ValueError("Each field must have a unique entity name")

        if any(
            not name or name != name.strip() for name in (*mapping.keys(), *entities)
        ):
            raise ValueError(
                "Field paths and entity names must be non-empty and trimmed"
            )

        labels = ("O",) + tuple(
            f"{prefix}-{entity}" for entity in entities for prefix in ("B", "I")
        )

        object.__setattr__(self, "field_to_entity", mapping)
        object.__setattr__(
            self,
            "entity_to_field",
            {entity: path for path, entity in mapping.items()},
        )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self,
            "label2id",
            {label: index for index, label in enumerate(labels)},
        )
        object.__setattr__(
            self,
            "id2label",
            dict(enumerate(labels)),
        )

    @property
    def num_labels(self) -> int:
        """Return the output dimension required by a token classifier.

        Returns:
            One background label plus two labels per configured field.
        """
        return len(self.labels)
