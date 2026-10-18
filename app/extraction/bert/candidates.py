import re
from dataclasses import dataclass

from app.extraction.bert.enums import EntityType
from app.extraction.bert.spans import CandidateSpan


@dataclass
class CandidatesSelector:
    def _select_unique_candidate(
        self,
        text: str,
        candidates: list[CandidateSpan],
    ) -> CandidateSpan | None:
        if len(candidates) != 1:
            return None

        return candidates[0]

    def _select_final_total_candidate(
        self,
        text: str,
        candidates: list[CandidateSpan],
    ) -> CandidateSpan | None:
        if not candidates:
            return None

        anchors = list(
            re.finditer(
                r"\btotal\b",
                text,
                flags=re.IGNORECASE,
            )
        )

        if anchors:
            final_anchor = anchors[-1].start()

            candidates_after_total = [
                candidate
                for candidate in candidates
                if candidate.start > final_anchor
            ]

            if candidates_after_total:
                return candidates_after_total[-1]

        return candidates[-1]

    def select(
        self,
        text: str,
        entity_type: EntityType,
        candidates: list[CandidateSpan]
    ) -> CandidateSpan | None:
        if entity_type in {EntityType.TAX, EntityType.TOTAL}:
            return self._select_final_total_candidate(
                text=text,
                candidates=candidates
            )
        return self._select_unique_candidate(
            text=text,
            candidates=candidates
        )