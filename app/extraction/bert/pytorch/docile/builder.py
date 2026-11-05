from dataclasses import dataclass, field

import pandas as pd

from app.extraction.bert.pytorch.docile.spans import (
    assign_candidate_labels,
    build_spans,
    deduplicate_candidate_labels,
    find_remaining_issues,
    resolve_name_address_nesting,
    resolve_total_amount_due_overlap,
)
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema


@dataclass
class DocileRecordBuilder:
    """Build character-span records for InvoiceDataset using the selected schema.

    Reject documents with unmatched annotations, unresolved labels, or invalid
    offsets. Keep page and annotation metadata alongside the training fields.
    """

    label_schema: InvoiceLabelSchema = field(default_factory=InvoiceLabelSchema)

    def build(self, df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        records = []
        rejected = []

        for row_index, row in df.iterrows():
            word_candidates, annotations = assign_candidate_labels(
                row,
                target_fields=self.label_schema.field_to_entity.keys(),
            )

            resolved_words = resolve_name_address_nesting(word_candidates)
            resolved_words = resolve_total_amount_due_overlap(resolved_words)
            deduplicated_words = deduplicate_candidate_labels(resolved_words)

            spans, ignored_regions = build_spans(
                row["ocr_text"],
                deduplicated_words,
            )

            issues = find_remaining_issues(
                deduplicated_words,
                annotations,
            )

            for span in spans:
                start, end = span["start"], span["end"]

                if not (
                    0 <= start < end <= len(row["ocr_text"])
                    and row["ocr_text"][start:end] == span["text"]
                ):
                    issues.append(
                        {
                            "status": "invalid_offsets",
                            "field": span["label"],
                            "start": start,
                            "end": end,
                        }
                    )

            if issues:
                rejected.append(
                    {
                        "row_index": row_index,
                        "document_id": row["document_id"],
                        "issues": issues,
                    }
                )
            else:
                records.append(
                    {
                        "row_index": row_index,
                        "document_id": row["document_id"],
                        "text": row["ocr_text"],
                        "entities": spans,
                        "ignored_regions": ignored_regions,
                        "amount_due_suppressed_words": sum(
                            bool(word["suppressed_amount_due"])
                            for word in deduplicated_words
                        ),
                    }
                )

        return records, rejected


def prepare_docile_records(
    df: pd.DataFrame,
    label_schema: InvoiceLabelSchema | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build records using the default nine-field schema unless one is supplied."""
    builder = DocileRecordBuilder(
        label_schema=label_schema if label_schema is not None else InvoiceLabelSchema(),
    )
    return builder.build(df)
