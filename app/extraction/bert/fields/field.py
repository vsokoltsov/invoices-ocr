import re
from pydantic_core import core_schema
import unicodedata
from app.extraction.bert.spans import CandidateSpan
from app.extraction.bert.utils import normalize_with_mapping

class Field(str):
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type,
        handler,
    ):
        return (
            core_schema
            .no_info_after_validator_function(
                cls,
                core_schema.str_schema(),
            )
        )

    def normalized(self) -> str:
        value = unicodedata.normalize(
            "NFKC",
            self,
        )

        return re.sub(
            r"\s+",
            " ",
            value,
        ).strip()
    
    def alignment_parts(
        self,
    ) -> tuple["Field", ...]:
        return (self,)

    def find_matches(
        self,
        text: str,
    ) -> list[CandidateSpan]:
        normalized_text, positions = (
            normalize_with_mapping(text)
        )
        normalized_value, _ = (
            normalize_with_mapping(self)
        )

        if not normalized_value:
            return []

        matches = []
        search_start = 0

        while True:
            normalized_start = normalized_text.find(
                normalized_value,
                search_start,
            )

            if normalized_start == -1:
                break

            normalized_end = (
                normalized_start
                + len(normalized_value)
            )

            matches.append(
                CandidateSpan(
                    start=positions[normalized_start],
                    end=(
                        positions[normalized_end - 1]
                        + 1
                    ),
                )
            )

            search_start = normalized_start + 1

        return matches