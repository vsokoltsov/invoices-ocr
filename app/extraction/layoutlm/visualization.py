"""Draw validation word errors on original invoice page images."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from app.extraction.layoutlm.converter import AttachmentToImageConverter


@dataclass
class ErrorPageRenderer:
    """Save a bounded set of gold and predicted label overlays.

    Attributes:
        converter: Loader for original cached or rendered page images.
        max_pages: Maximum number of pages saved after one validation epoch.
    """

    converter: AttachmentToImageConverter
    max_pages: int = 10

    @staticmethod
    def _panel(image: Image.Image, words: pd.DataFrame, column: str, color: str):
        """Draw labels for incorrect words without changing the source image."""
        panel = ImageOps.expand(image.copy(), border=(0, 32, 0, 0), fill="white")
        draw = ImageDraw.Draw(panel)
        draw.text((10, 8), column.replace("_", " "), fill=color)
        for word in words.to_dict("records"):
            box = json.loads(word["bbox"])
            x0, y0, x1, y1 = (
                box[0] * image.width / 1000,
                box[1] * image.height / 1000 + 32,
                box[2] * image.width / 1000,
                box[3] * image.height / 1000 + 32,
            )
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
            draw.text((x0, max(32, y0 - 12)), word[column], fill=color)
        return panel

    def save(self, pages, words: pd.DataFrame, errors: pd.DataFrame, output: Path):
        """Draw the pages with the most incorrect BIO word labels.

        Args:
            pages: Original page records with document IDs, PDF paths, and page numbers.
            words: Prediction report with boxes in the integer 0..1000 scale.
            errors: Strict entity error report used to select failed pages.
            output: Epoch diagnostic directory.

        Notes:
            Page numbers in overlays/index.csv are zero-based. Only words with
            incorrect BIO labels are drawn. Gold labels are on the left and
            predicted labels are on the right. Source images remain unchanged.
        """
        if self.max_pages <= 0 or errors.empty:
            return
        failed = set(zip(errors.document_id, errors.page, strict=True))
        wrong = words.loc[~words.correct]
        counts = (
            wrong.groupby(["document_id", "page"]).size().sort_values(ascending=False)
        )
        page_lookup = {
            (page["document_id"], page["page"]): (index, page)
            for index, page in enumerate(pages)
        }
        directory = output / "overlays"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = []
        for key in counts.index:
            if key not in failed:
                continue
            page_index, page = page_lookup[key]
            image = self.converter.load_page_image(page).copy()
            image.thumbnail((1200, 1600))
            selected = wrong.loc[wrong.document_id.eq(key[0]) & wrong.page.eq(key[1])]
            gold = self._panel(image, selected, "gold_label", "green")
            predicted = self._panel(image, selected, "predicted_label", "red")
            combined = Image.new("RGB", (gold.width * 2, gold.height), "white")
            combined.paste(gold, (0, 0))
            combined.paste(predicted, (gold.width, 0))
            filename = f"page_{page_index:04d}.png"
            combined.save(directory / filename)
            manifest.append({"document_id": key[0], "page": key[1], "image": filename})
            if len(manifest) >= self.max_pages:
                break
        pd.DataFrame(manifest).to_csv(directory / "index.csv", index=False)
