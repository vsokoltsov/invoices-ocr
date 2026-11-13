from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image


@dataclass
class AttachmentToImageConverter:
    """Render PDF pages and reuse RGB PNGs cached by document, page, and DPI."""

    images_path: Path

    def __post_init__(self):
        self.images_path = Path(self.images_path)

    def load_page_image(self, page, dpi=150) -> Image.Image:
        if int(page["page"]) < 0:
            raise ValueError("Page index must be nonnegative")
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        cache_path = (
            self.images_path
            / f"dpi_{dpi}"
            / page["document_id"]
            / f"page_{int(page['page']):04d}.png"
        )

        if cache_path.is_file():
            with Image.open(cache_path) as image:
                return image.convert("RGB")

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

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(cache_path)
        return image
