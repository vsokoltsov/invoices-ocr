
from collections import defaultdict
from collections.abc import Iterable

from app.extraction.bert.enums import EntityType
from app.extraction.bert.spans import EntitySpan


def normalize_with_mapping(text):
    normalized = []
    original_positions = []

    previous_was_space = True

    for index, character in enumerate(str(text)):
        if character.isalnum():
            normalized.append(character.lower())
            original_positions.append(index)
            previous_was_space = False

        elif not previous_was_space:
            normalized.append(" ")
            original_positions.append(index)
            previous_was_space = True

    if normalized and normalized[-1] == " ":
        normalized.pop()
        original_positions.pop()

    return "".join(normalized), original_positions


def group_spans_by_entity(
    spans: Iterable[EntitySpan],
) -> dict[
    EntityType,
    tuple[EntitySpan, ...],
]:
    grouped: dict[
        EntityType,
        list[EntitySpan],
    ] = defaultdict(list)

    for span in spans:
        grouped[span.entity].append(span)

    return {
        entity: tuple(
            sorted(
                entity_spans,
                key=lambda span: span.start,
            )
        )
        for entity, entity_spans
        in grouped.items()
    }

def money_digits(value: object) -> str:
    return "".join(
        character
        for character in str(value)
        if character.isdigit()
    )