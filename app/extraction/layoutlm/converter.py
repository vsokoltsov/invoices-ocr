from dataclasses import dataclass

import pymupdf
from PIL import Image

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts


@dataclass
class AttachmentToImageConverter:
    """Render PDF pages and reuse RGB PNGs cached by document, page, and DPI."""

    artifacts: LayoutLMArtifacts

    def load_page_image(self, page, dpi=150) -> Image.Image:
        if int(page["page"]) < 0:
            raise ValueError("Page index must be nonnegative")
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        cached = self.artifacts.load_page_image(
            page["document_id"], int(page["page"]), dpi
        )
        if cached is not None:
            return cached

        with pymupdf.open(page["pdf_path"]) as document:
            pdf_page = document.load_page(int(page["page"]))
            pixmap = pdf_page.get_pixmap(
                dpi=dpi,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )

        self.artifacts.save_page_image(
            page["document_id"], int(page["page"]), dpi, image
        )
        return image
