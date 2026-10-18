import re
from app.extraction.bert.fields.identifier import IdentifierField
from app.extraction.bert.spans import CandidateSpan

class AccountNumberField(
    IdentifierField
):
    OCR_ALIASES = (
        "IBAN",
        "Account number",
        "Account no",
    )

    @classmethod
    def alias_pattern(
        cls,
    ) -> re.Pattern:
        alias_patterns = [
            r"\s+".join(
                re.escape(word)
                for word in alias.split()
            )
            for alias in cls.OCR_ALIASES
        ]

        aliases = "|".join(alias_patterns)

        return re.compile(
            rf"""
            \b(?:{aliases})\b
            \.?
            \s*[:#-]?\s*
            (?P<value>[A-Z0-9-]{{12,34}})
            \b
            """,
            flags=re.IGNORECASE | re.VERBOSE,
        )


    def find_alias_candidates(
        self,
        text: str,
    ) -> list[CandidateSpan]:
        candidates = []

        for match in self.alias_pattern().finditer(text):
            candidates.append(
                CandidateSpan(
                    start=match.start("value"),
                    end=match.end("value"),
                )
            )

        return candidates

    def find_matches(
        self,
        text: str,
    ) -> list[CandidateSpan]:
        exact_matches = (
            super().find_matches(text)
        )

        if exact_matches:
            return exact_matches

        return self.find_alias_candidates(
            text
        )