import re

from app.extraction.bert.enums import OCR_CONFUSION_PAIRS, MatchType
from app.extraction.bert.fields.field import Field
from app.extraction.bert.spans import CandidateSpan


class IdentifierField(Field):
    def compact(self) -> str:
        return self.compact_value(self)

    @staticmethod
    def compact_value(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", value).upper()

    def find_matches(
        self,
        text: str,
    ) -> list[CandidateSpan]:
        characters = re.findall(
            r"[A-Za-z0-9]",
            self,
        )

        if not characters:
            return []

        pattern = (
            r"(?<![A-Za-z0-9])"
            + r"[^A-Za-z0-9]*".join(
                re.escape(character)
                for character in characters
            )
            + r"(?![A-Za-z0-9])"
        )

        return [
            CandidateSpan(
                start=match.start(),
                end=match.end(),
            )
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        ]

    def classify_match(
        self,
        observed: str,
    ) -> MatchType | None:
        expected = str(self)

        if expected == observed:
            return MatchType.EXACT

        expected_compact = self.compact_value(expected)
        observed_compact = self.compact_value(observed)

        if expected_compact == observed_compact:
            return MatchType.NORMALIZED

        if self.identifiers_are_equivalent(
            expected_compact,
            observed_compact,
        ):
            return MatchType.OCR_EQUIVALENT

        if (
            expected_compact
            and observed_compact
            and (
                expected_compact in observed_compact
                or observed_compact in expected_compact
            )
        ):
            return MatchType.PARTIAL

        return None

    @staticmethod
    def identifiers_are_equivalent(
        expected: str,
        observed: str,
    ) -> bool:
        if len(expected) != len(observed):
            return False

        return all(
            IdentifierField.characters_are_equivalent(
                expected_character,
                observed_character,
            )
            for expected_character, observed_character in zip(
                expected,
                observed,
            )
        )

    @staticmethod
    def characters_are_equivalent(
        expected: str,
        observed: str,
    ) -> bool:
        if expected == observed:
            return True

        return frozenset({
            expected,
            observed,
        }) in OCR_CONFUSION_PAIRS