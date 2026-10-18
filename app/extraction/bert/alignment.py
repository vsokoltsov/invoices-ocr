import re
from dataclasses import dataclass
from app.extraction.bert.spans import (
    FinancialSummarySpans,
    CandidateSpan,
    EntitySpan
)
from app.extraction.bert.candidates import CandidatesSelector
from app.extraction.bert.models import InvoiceTarget

@dataclass
class FinancialSummaryAligner:
    MONEY_PATTERN = re.compile(
        r"""
        (?<![\dA-Za-z])
        (
            (?:
                \d{1,3}
                (?:[ \u00A0.'’]\d{3})+
                |
                \d+
            )
            [,.]
            \d{2}
        )
        (?!\d)
        """,
        flags=re.VERBOSE,
    )

    TOTAL_PATTERN = re.compile(
        r"\btotal\b",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        lookahead: int = 200,
    ):
        self.lookahead = lookahead

    def find(
        self,
        text: str,
    ) -> FinancialSummarySpans | None:
        total_anchors = list(
            self.TOTAL_PATTERN.finditer(text)
        )

        # Start from the final occurrence because "total" can
        # appear in product descriptions or other sections.
        for anchor in reversed(total_anchors):
            region_end = min(
                len(text),
                anchor.end() + self.lookahead,
            )

            amounts = list(
                self.MONEY_PATTERN.finditer(
                    text,
                    anchor.end(),
                    region_end,
                )
            )

            if len(amounts) < 3:
                continue

            net_match = amounts[0]
            tax_match = amounts[1]
            total_match = amounts[2]

            return FinancialSummarySpans(
                net=CandidateSpan(
                    start=net_match.start(),
                    end=net_match.end(),
                ),
                tax=CandidateSpan(
                    start=tax_match.start(),
                    end=tax_match.end(),
                ),
                total=CandidateSpan(
                    start=total_match.start(),
                    end=total_match.end(),
                ),
            )

        return None

@dataclass
class DocumentAligner:
    financial_summary_aligner: FinancialSummaryAligner
    selector: CandidatesSelector

    def find_spans(self, text: str, target: InvoiceTarget) -> list[EntitySpan]:
        
        financial_summary = (
            self.financial_summary_aligner.find(text)
        )
        spans: list[EntitySpan] = []
        occupied: list[CandidateSpan] = []

        for entity, value in target.extract_fields():

            for value_part in value.alignment_parts():
                financial_candidate = (
                    financial_summary.get(entity)
                    if financial_summary is not None
                    else None
                )
                if financial_candidate is not None:
                    # For TAX and TOTAL, the position in the final
                    # summary row is more reliable than ordinary
                    # text matching.
                    candidates = [financial_candidate]
                else:
                    candidates = value_part.find_matches(text)

                available_candidates = [
                    candidate
                    for candidate in candidates
                    if not any(
                        candidate.overlaps(existing)
                        for existing in occupied
                    )
                ]

                if not available_candidates:
                    continue

                if financial_candidate is not None:
                    # The financial aligner already selected the
                    # semantic position.
                    selected = available_candidates[0]
                else:
                    selected = self.selector.select(
                        text=text,
                        entity_type=entity,
                        candidates=available_candidates
                    )

                if selected is None:
                    continue

                spans.append(
                    EntitySpan(
                        start=selected.start,
                        end=selected.end,
                        entity=entity,
                        # Preserve the OCR-observed text rather than
                        # copying the expected JSON value.
                        value=text[
                            selected.start:selected.end
                        ],
                    )
                )

                occupied.append(selected)

        return sorted(
            spans,
            key=lambda span: span.start,
        )