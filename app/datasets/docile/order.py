from dataclasses import dataclass, replace
from statistics import median

from app.datasets.docile.models import OCRWord


@dataclass
class OCRReadingOrder:
    # Maximum vertical distance from the row center,
    # expressed as a fraction of the median word height.
    row_tolerance: float = 0.5

    def build(
        self,
        words: list[OCRWord],
    ) -> tuple[str, list[OCRWord]]:
        """Order words by page and visual row, then rebuild text offsets.

        This uses horizontal reading order. It does not detect columns
        or reconstruct table structure.
        """
        if not words:
            return "", []

        pages = {}

        for word in words:
            if word.text:
                pages.setdefault(word.page, []).append(word)

        ordered_words = []

        for page_index in sorted(pages):
            page_words = pages[page_index]

            heights = [
                word.bbox[3] - word.bbox[1]
                for word in page_words
                if word.bbox[3] > word.bbox[1]
            ]

            if not heights:
                raise ValueError(f"Page {page_index} has no valid word heights")

            tolerance = median(heights) * self.row_tolerance

            # Each row stores its words and vertical center positions.
            rows = []

            for word in sorted(
                page_words,
                key=lambda word: (
                    self._center_y(word),
                    word.bbox[0],
                ),
            ):
                center_y = self._center_y(word)

                if rows:
                    row = rows[-1]
                    row_center = median(row["centers"])
                else:
                    row = None

                if row is None or center_y - row_center > tolerance:
                    row = {"words": [], "centers": []}
                    rows.append(row)

                row["words"].append(word)
                row["centers"].append(center_y)

            for row in rows:
                ordered_words.extend(
                    sorted(
                        row["words"],
                        key=lambda word: word.bbox[0],
                    )
                )

        parts = []
        rebuilt_words = []
        position = 0

        for word in ordered_words:
            if parts:
                parts.append(" ")
                position += 1

            start = position
            parts.append(word.text)
            position += len(word.text)

            rebuilt_words.append(
                replace(
                    word,
                    start=start,
                    end=position,
                )
            )

        return "".join(parts), rebuilt_words

    @staticmethod
    def _center_y(word: OCRWord) -> float:
        return (word.bbox[1] + word.bbox[3]) / 2
