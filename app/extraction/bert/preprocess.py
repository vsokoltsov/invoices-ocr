from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from app.extraction.bert.spans import (
    assign_candidate_labels,
    build_spans,
    deduplicate_candidate_labels,
    find_remaining_issues,
    resolve_name_address_nesting,
    resolve_total_amount_due_overlap,
)
from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.preprocess import DataFramePipeline, DecodeDocuments, DocumentStep


@dataclass
class AssignAnnotations(DocumentStep):
    """Project selected annotation boxes onto OCR words.

    Attributes:
        assign: Model-specific selection of invoice or line-item annotations.
    """

    assign: Callable
    output_column = "projection"

    def process(self, row: pd.Series) -> tuple[list, list]:
        """Return candidate word labels and their source annotations.

        Args:
            row: Document with decoded annotations and OCR words.

        Returns:
            Word candidates and annotation metadata.
        """
        return self.assign(row)


@dataclass
class ResolveCandidates(DocumentStep):
    """Apply one conflict rule to candidate word labels.

    Attributes:
        resolve: A rule that returns a new list of candidate words.
    """

    resolve: Callable
    output_column = "projection"

    def process(self, row: pd.Series) -> tuple[list, list]:
        """Keep annotation metadata while resolving word candidates.

        Args:
            row: Document with a projection from the preceding step.

        Returns:
            Resolved words and unchanged annotation metadata.
        """
        words, annotations = row.projection
        return self.resolve(words), annotations


class CreateSpans(DocumentStep):
    """Convert resolved word labels to character spans and ignored regions."""

    output_column = "span_data"

    def process(self, row: pd.Series) -> tuple[list, list]:
        """Create spans using offsets in the document text.

        Args:
            row: Document with resolved candidates and OCR text.

        Returns:
            Entity spans and ignored character regions.
        """
        return build_spans(row.ocr_text, row.projection[0])


class ValidateSpans(DocumentStep):
    """Report unmatched annotations, unresolved labels, and invalid offsets."""

    output_column = "issues"

    def process(self, row: pd.Series) -> list[dict]:
        """Collect all validation issues for one document.

        Args:
            row: Document with candidate metadata and constructed spans.

        Returns:
            An empty list for a valid document, or structured rejection issues.
        """
        issues = find_remaining_issues(*row.projection)
        for span in row.span_data[0]:
            start, end = span["start"], span["end"]
            if not (
                0 <= start < end <= len(row.ocr_text)
                and row.ocr_text[start:end] == span["text"]
            ):
                issues.append(
                    {
                        "status": "invalid_offsets",
                        "field": span["label"],
                        "start": start,
                        "end": end,
                    }
                )
        return issues


@dataclass
class InvoicePreprocessor:
    """Run ordered BERT steps to create validated character spans.

    Attributes:
        label_schema: Selected invoice fields and BIO label IDs.
    """

    label_schema: InvoiceLabelSchema = field(default_factory=InvoiceLabelSchema)

    def _assign_candidates(self, row):
        """Select invoice annotations for word projection.

        Args:
            row: Document with decoded OCR and annotation data.

        Returns:
            Candidate word labels and selected annotation metadata.
        """
        return assign_candidate_labels(
            row, target_fields=self.label_schema.field_to_entity.keys()
        )

    def pipeline(self) -> DataFramePipeline:
        """Compose decoding, annotation rules, span creation, and validation.

        Returns:
            An ordered pipeline executed by PipeFunc.
        """
        return DataFramePipeline(
            (
                DecodeDocuments(),
                AssignAnnotations(self._assign_candidates),
                ResolveCandidates(resolve_name_address_nesting),
                ResolveCandidates(resolve_total_amount_due_overlap),
                ResolveCandidates(deduplicate_candidate_labels),
                CreateSpans(),
                ValidateSpans(),
            )
        )

    def run(self, frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        """Preprocess documents and separate accepted records from rejections.

        Args:
            frame: Source documents with OCR text, words, and annotations.

        Returns:
            Accepted character-span records and structured rejection records.
        """
        prepared = self.pipeline().run(frame)
        records, rejected = [], []
        for index, row in prepared.iterrows():
            issues = (
                row.issues
                if not row.reason
                else [{"status": "invalid_document", "reason": row.reason}]
            )
            if issues:
                rejected.append(
                    {
                        "row_index": index,
                        "document_id": row.document_id,
                        "issues": issues,
                    }
                )
                continue
            spans, ignored_regions = row.span_data
            records.append(
                {
                    "row_index": index,
                    "document_id": row.document_id,
                    "text": row.ocr_text,
                    "entities": spans,
                    "ignored_regions": ignored_regions,
                    "amount_due_suppressed_words": sum(
                        bool(word["suppressed_amount_due"])
                        for word in row.projection[0]
                    ),
                }
            )
        return records, rejected
