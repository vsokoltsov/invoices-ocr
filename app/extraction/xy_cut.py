import json
from dataclasses import dataclass
from statistics import median

import pandas as pd


@dataclass
class XYCut:
    """Reorder document blocks and optionally protect table bands.

    Attributes:
        min_gap_x: Minimum horizontal gap in normalized coordinates.
        min_gap_y: Minimum vertical gap in normalized coordinates.
        table_y_ranges_by_document: Optional document-to-page table ranges.
            None applies full-page XY-cut to every document.
            Missing documents in a supplied mapping retain their order.
    """

    min_gap_x: float = 0.04
    min_gap_y: float = 0.03
    table_y_ranges_by_document: (
        dict[str, dict[int, list[tuple[float, float]]]] | None
    ) = None

    def __post_init__(self):
        if not (0 < self.min_gap_x <= 1 and 0 < self.min_gap_y <= 1):
            raise ValueError("Gap thresholds must be greater than 0 and at most 1")

    def word_center_y(self, word):
        return (word["bbox"][1] + word["bbox"][3]) / 2

    def widest_gap(self, words, axis, minimum):
        """Find the widest empty gap along X or Y."""
        intervals = sorted(
            (word["bbox"][axis], word["bbox"][axis + 2]) for word in words
        )
        end = intervals[0][1]
        gaps = []

        for start, stop in intervals[1:]:
            if start - end >= minimum:
                gaps.append((start - end, (start + end) / 2))
            end = max(end, stop)

        return max(gaps)[1] if gaps else None

    def horizontal_order(self, words):
        """Group words into visual lines, then read each line left to right."""
        if not words:
            return []

        tolerance = median(word["bbox"][3] - word["bbox"][1] for word in words) * 0.5
        lines = []

        for word in sorted(
            words, key=lambda word: (self.word_center_y(word), word["bbox"][0])
        ):
            if (
                not lines
                or self.word_center_y(word)
                - median(self.word_center_y(item) for item in lines[-1])
                > tolerance
            ):
                lines.append([])

            lines[-1].append(word)

        return [
            word
            for line in lines
            for word in sorted(line, key=lambda word: word["bbox"][0])
        ]

    def xy_cut_order(self, words, min_gap_x, min_gap_y):
        """Separate blocks recursively, then order words within each block."""
        if len(words) <= 1:
            return words

        for axis, minimum in ((0, min_gap_x), (1, min_gap_y)):
            cut = self.widest_gap(words, axis, minimum)
            if cut is None:
                continue

            first = [word for word in words if word["bbox"][axis + 2] <= cut]
            second = [word for word in words if word["bbox"][axis] >= cut]

            if first and second and len(first) + len(second) == len(words):
                return self.xy_cut_order(
                    first, min_gap_x, min_gap_y
                ) + self.xy_cut_order(second, min_gap_x, min_gap_y)

        return self.horizontal_order(words)

    def reorder_words(self, row, table_y_ranges=None):
        """Return reordered word copies with rebuilt character offsets.

        Args:
            row: Document containing word dictionaries and their offsets.
            table_y_ranges: Optional page-to-table-interval mapping.
                None applies full-page XY-cut to every page.
                Missing pages in a supplied mapping retain their order.

        Returns:
            Reordered word dictionaries. The input row remains unchanged.
        """

        words = row["text_with_boxes"]
        if isinstance(words, str):
            words = json.loads(words)

        words = sorted(
            [dict(word) for word in words],
            key=lambda word: word["start"],
        )
        ordered = []

        for page in sorted({int(word["page"]) for word in words}):
            remaining = [word for word in words if int(word["page"]) == page]

            if table_y_ranges is not None and page not in table_y_ranges:
                ordered.extend(remaining)
                continue

            intervals = [] if table_y_ranges is None else sorted(table_y_ranges[page])

            previous_bottom = 0.0

            for top, bottom in intervals:
                if not 0 <= previous_bottom <= top < bottom <= 1:
                    raise ValueError("Table intervals must be valid and disjoint")
                previous_bottom = bottom

                before = [word for word in remaining if self.word_center_y(word) < top]
                table = [
                    word
                    for word in remaining
                    if top <= self.word_center_y(word) <= bottom
                ]

                ordered.extend(
                    self.xy_cut_order(
                        before,
                        self.min_gap_x,
                        self.min_gap_y,
                    )
                )
                ordered.extend(table)

                remaining = [
                    word for word in remaining if self.word_center_y(word) > bottom
                ]

            ordered.extend(
                self.xy_cut_order(
                    remaining,
                    self.min_gap_x,
                    self.min_gap_y,
                )
            )

        position = 0
        for word in ordered:
            word["start"] = position
            word["end"] = position + len(word["text"])
            position = word["end"] + 1

        return ordered

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reorder documents and rebuild matching OCR text and offsets."""
        result = frame.copy()
        documents = []

        for _, row in frame.iterrows():
            table_ranges = None

            if self.table_y_ranges_by_document is not None:
                table_ranges = self.table_y_ranges_by_document.get(
                    row["document_id"],
                    {},
                )

            documents.append(self.reorder_words(row, table_ranges))

        result["text_with_boxes"] = documents
        result["ocr_text"] = [
            " ".join(word["text"] for word in words) for words in documents
        ]
        return result
