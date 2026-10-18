from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
from transformers import PreTrainedTokenizerBase

from app.extraction.bert.alignment import DocumentAligner
from app.extraction.bert.fields.account import AccountNumberField
from app.extraction.bert.models import InvoiceTarget
from app.extraction.bert.spans import CandidateSpan, EntitySpan


class TokenizedBatch(TypedDict, total=False):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    token_type_ids: list[list[int]]
    labels: list[list[int]]


@dataclass(slots=True)
class TokenLabelBatchEncoder:
    """
    Converts invoice OCR text and its expected JSON into the numerical input
    that BERT needs for token classification training

    OCR text + expected JSON =>
    DocumentAligner finds character positions =>
    Tokenizer splits text into BERT tokens =>
    Each token gets 0, B-ENTITY or I-ENTITY =>
    Text tokens and labels become integer arrays
    """

    # Hugging Face tokenizer
    tokenizer: PreTrainedTokenizerBase
    # Finds where expected invoice values occur in the OCR text.
    aligner: DocumentAligner
    # Converts label names into integers
    label2id: Mapping[str, int]
    # BERT can process only a limited number of tokens at once
    # For standard BERT, this is usually 512 tokens.
    # If an invoice contains more than 512 tokens, 
    # it must be divided into multiple chunks.
    max_length: int = 512
    # Chunks overlap by 64 tokens
    window_stride: int = 64

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError(
                "max_length must be positive"
            )

        if not 0 <= self.window_stride < self.max_length:
            raise ValueError(
                "window_stride must be between 0 "
                "and max_length - 1"
            )

        if "O" not in self.label2id:
            raise ValueError(
                "label2id must contain the O label"
            )

    @staticmethod
    def _get_chunk_range(
        offsets: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """
        Receives the character offsets for every token in one chunk.
        An offset tells us where a token came from in the original text.

        Example:
            Text: "Invoice 12345"

            Possible offsets:
            [
                (0, 0),    # [CLS] special token
                (0, 7),    # Invoice
                (8, 11),   # 123
                (11, 13),  # ##45
                (0, 0),    # [SEP] special token
            ]

            (0, 7) means the token uses characters 0 through 6.
        """

        valid_offsets = [
            (start, end)
            for start, end in offsets
            if start != end
        ]

        if not valid_offsets:
            return None

        return (
            valid_offsets[0][0],
            valid_offsets[-1][1],
        )

    @staticmethod
    def _calculate_overlap(
        span: EntitySpan,
        chunk_range: tuple[int, int] | None,
    ) -> int:
        """
        Calculates how many characters of an entity occur inside a chunk.
        """

        if chunk_range is None:
            return 0

        chunk_start, chunk_end = chunk_range

        """
        Standard interval-overlap formula.
        Suppose:
            Entity: [80, 120)
            Chunk:  [0, 100)

        The overlapping part is:
            [80, 100)

        Its length is 20 characters.
        The calculation is:
            min(120, 100) - max(80, 0)
        which becomes:
            100 - 80
        which is:
            20
        max(0, ...) prevents a negative result when the ranges do not overlap.
        """
        return max(
            0,
            min(span.end, chunk_end)
            - max(span.start, chunk_start),
        )

    def _find_span_owners(
        self,
        spans: list[EntitySpan],
        chunk_ranges: list[
            tuple[int, int] | None
        ],
    ) -> dict[int, int]:
        """
        Assign each entity span to exactly one token chunk.

        This prevents entities in overlapping tokenizer windows
        from appearing more than once in training/evaluation.
        """

        owners: dict[int, int] = {}

        for span_index, span in enumerate(spans):

            """
            Example:
                Chunk:  [0, 100)
                Entity: [30, 50)
            The chunk fully contains the entity.
            But:
                Chunk:  [0, 100)
                Entity: [80, 120)

            does not fully contain it.
            """
            containing_chunks = [
                chunk_index
                for chunk_index, chunk_range
                in enumerate(chunk_ranges)
                if (
                    chunk_range is not None
                    and chunk_range[0] <= span.start
                    and span.end <= chunk_range[1]
                )
            ]

            # If one or more chunks fully contain the entity,
            # assign it to the first one.
            # It might appear completely in two 
            # chunks because of the stride overlap.
            # Choosing the first prevents duplicate training labels.
            if containing_chunks:
                owners[span_index] = (
                    containing_chunks[0]
                )
                continue
            
            # If no chunk completely contains the entity, 
            # calculate how much of it appears in every chunk.
            overlaps = [
                self._calculate_overlap(
                    span,
                    chunk_range,
                )
                for chunk_range in chunk_ranges
            ]

            # no chunks => nowhere to assign the entity.
            if not overlaps:
                continue

            best_chunk = int(np.argmax(overlaps))

            # Only assign the entity 
            # if the selected chunk actually overlaps it.
            if overlaps[best_chunk] > 0:
                owners[span_index] = best_chunk

        return owners

    def _label_chunk(
         self,
        offsets: list[tuple[int, int]],
        chunk_index: int,
        spans: list[EntitySpan],
        span_owners: Mapping[int, int],
        ignored_spans: list[CandidateSpan] | None = None,
    ) -> list[int]:
        """
        This method produces one label ID for every token in one chunk.
        For example:
            [0, 0, 1, 2, -100]
        could mean:
            Token 1 → O
            Token 2 → O
            Token 3 → B-INVOICE_NUMBER
            Token 4 → I-INVOICE_NUMBER
            Token 5 → ignored
        """

        labels: list[int] = []
        started_spans: set[int] = set()
        ignored_spans = ignored_spans or []

        # Process every token’s character range.
        for token_start, token_end in offsets:
            # Special and padding tokens.
            # A token with an empty range is usually:
            # * `[CLS]`
            # * `[SEP]`
            # * Padding
            if token_start == token_end:
                labels.append(-100)
                continue

            overlaps_ignored_span = any(
                token_start < ignored_span.end
                and token_end > ignored_span.start
                for ignored_span in ignored_spans
            )

            if overlaps_ignored_span:
                labels.append(-100)
                continue

            overlapping_span: (
                tuple[int, EntitySpan] | None
            ) = None

            for span_index, span in enumerate(spans):
                # Check whether the token range and entity range overlap.
                overlaps = (
                    token_start < span.end
                    and token_end > span.start
                )
                # Store the first matching entity and stop searching.
                # This is safe because InvoiceSpanAligner
                # prevents entity spans from overlapping each other.
                if overlaps:
                    overlapping_span = (
                        span_index,
                        span,
                    )
                    break
            
            # If the token is not inside an entity
            # give it the `O` label
            if overlapping_span is None:
                labels.append(
                    self.label2id["O"]
                )
                continue

            span_index, span = overlapping_span

            # Ignore duplicate span copies produced by
            # overlapping tokenizer windows.
            # The entity may appear in multiple overlapping chunks.
            # If this chunk is not the entity’s selected owner, ignore its token with -100.
            # This avoids counting or training on the entity twice.
            if span_owners.get(span_index) != chunk_index:
                labels.append(-100)
                continue

            prefix = (
                "B"
                if span_index not in started_spans
                else "I"
            )

            label_name = (
                f"{prefix}-{span.entity}"
            )

            # Check that the label exists in the configured label mapping.
            # This catches situations where you add a new entity to InvoiceTarget
            # but forget to add its BIO labels to label2id.
            if label_name not in self.label2id:
                # Stop with an informative error rather than 
                # silently assigning an incorrect label.
                raise KeyError(
                    f"Missing label in label2id: "
                    f"{label_name}"
                )

            # Convert the label string to a number and 
            # add it to the result.
            labels.append(
                self.label2id[label_name]
            )
            started_spans.add(span_index)

        return labels

    @staticmethod
    def _parse_target(
        value: object,
    ) -> InvoiceTarget:
        """
        Handles three possible input formats: 
        * InvoiceTarget
        * str
        * dict
        """

        if isinstance(value, InvoiceTarget):
            return value

        if isinstance(value, str):
            return InvoiceTarget.model_validate_json(
                value
            )

        return InvoiceTarget.model_validate(
            value
        )

    def _find_ignored_account_spans(
        self,
        *,
        text: str,
        target: InvoiceTarget,
    ) -> list[CandidateSpan]:
        account_number = (
            target
            .payment_instructions
            .account_number
        )

        # This document has a genuine JSON annotation.
        # The normal DocumentAligner will label it.
        if (
            account_number is not None
            and str(account_number).strip()
        ):
            return []

        # The OCR contains an account number, but JSON does
        # not annotate it. Ignore its tokens instead of
        # incorrectly labelling them as O.
        return [
            CandidateSpan(
                start=match.start("value"),
                end=match.end("value"),
            )
            for match in (
                AccountNumberField
                .alias_pattern()
                .finditer(text)
            )
        ]

    def __call__(
        self,
        examples: Mapping[str, list],
    ) -> TokenizedBatch:
        """
        Makes the object callable for passing it to Hugging Face
        """
        
        """
        The tokenizer declares the inputs expected by its model.
        For BERT, these are commonly:
            [
                "input_ids",
                "token_type_ids",
                "attention_mask",
            ]
        The code creates:
            {
                "input_ids": [],
                "token_type_ids": [],
                "attention_mask": [],
            }
        """
        output: dict[str, list[list[int]]] = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels": []
        }

        # Processing every document in the batch
        for text, json_data in zip(
            examples["ocred_text"],
            examples["json_data"],
        ):
            text = str(text)

            target = self._parse_target(
                value=json_data
            )

            # Finding entity character spans
            # Connects the target JSON values with 
            # their locations in the OCR text.
            # At this point, labels exist at the character level
            # They are not yet token labels.
            spans = self.aligner.find_spans(
                text=text,
                target=target,
            )
            ignored_spans = (
                self._find_ignored_account_spans(
                    text=text,
                    target=target,
                )
            )
            
            encoding = self.tokenizer(
                # Pass the OCR text into the Hugging Face tokenizer
                text,
                # The tokenizer must respect `max_length`.
                # Because overflow handling is enabled, 
                # text beyond the first chunk is placed into additional 
                # chunks rather than simply discarded.
                truncation=True,
                # Each chunk can contain at most this many tokens.
                max_length=self.max_length,
                # Neighbouring chunks overlap by this many tokens.
                stride=self.window_stride,
                # Create multiple chunks when the document is too long.
                # Without this, the text beyond `max_length` would be lost.
                return_overflowing_tokens=True,
                # Return each token’s original character location.
                # For example:
                #   "12345"
                # might become:
                #   "123"   → (16, 19)
                #   "##45"  → (19, 21)
                # These offsets are what connect character spans 
                # from `InvoiceSpanAligner` to BERT tokens.
                return_offsets_mapping=True,
            )

            # Remove offsets from `encoding` and save them separately.
            # Offsets are needed to build labels,
            # but they are not an input to BERT.
            offset_chunks = encoding.pop(
                "offset_mapping"
            )

            # The tokenizer can return metadata showing which original example produced each overflow chunk.
            # This encoder processes one document at a time inside the loop, so that metadata is unnecessary.
            # The second argument, `None`, means “do not raise an error if this key does not exist.”
            encoding.pop(
                "overflow_to_sample_mapping",
                None,
            )

            # Finding chunk character ranges
            # For each chunk, calculate the portion of the original text it covers.
            # Example:
            #   [
            #       (0, 1800),
            #       (1550, 3200),
            #   ]
            # The overlap from characters 1550–1800 comes from the tokenizer stride.
            chunk_ranges = [
                self._get_chunk_range(offsets)
                for offsets in offset_chunks
            ]

            # Selecting one owner per entity
            # Choose exactly one chunk responsible for each entity.
            # Example:
            #   {
            #       0: 0,
            #       1: 0,
            #       2: 1,
            #   }
            # This prevents duplicate entity labels in overlapping chunks.
            span_owners = self._find_span_owners(
                spans=spans,
                chunk_ranges=chunk_ranges,
            )

            # Creating labels for each chunk
            # Process every chunk created from this document.
            for chunk_index, offsets in enumerate(
                offset_chunks
            ):
                # Create one numerical label per token in this chunk.
                # Example:
                #   [-100, 0, 0, 1, 2, -100]
                labels = self._label_chunk(
                    offsets=offsets,
                    chunk_index=chunk_index,
                    spans=spans,
                    span_owners=span_owners,
                    ignored_spans=ignored_spans,
                )

                # Saving model inputs
                # Process each input required by BERT:
                # * input_ids
                # * attention_mask
                # * token_type_ids

                for name in (
                    self.tokenizer.model_input_names
                ):
                    # Take the input values for the current chunk and 
                    # add them to the output batch.
                    output[name].append(
                        encoding[name][chunk_index]
                    )

                # Add the labels belonging to the same chunk.
                # The positions line up:
                #   input_ids[0][5]
                # and:
                #   labels[0][5]
                # refer to the same token.

                output["labels"].append(labels)

        # Return all generated chunks and labels.
        # The result looks approximately like:
        # {
        #     "input_ids": [
        #         [101, 1999, 6764, 102, ...],
        #         [101, 2023, 2003, 102, ...],
        #     ],
        #     "attention_mask": [
        #         [1, 1, 1, 1, ...],
        #         [1, 1, 1, 1, ...],
        #     ],
        #     "token_type_ids": [
        #         [0, 0, 0, 0, ...],
        #         [0, 0, 0, 0, ...],
        #     ],
        #     "labels": [
        #         [-100, 0, 1, -100, ...],
        #         [-100, 0, 0, -100, ...],
        #     ],
        # }
        # Each row represents one chunk, not necessarily one original invoice.
        # One long invoice can produce several rows.
        return cast(TokenizedBatch, output)
