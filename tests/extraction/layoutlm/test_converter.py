import numpy as np
import pytest

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.converter import AttachmentToImageConverter


def test_pdf_pages_render_as_rgb_and_cache_survives_closed_pdf(document_row, tmp_path):
    converter = AttachmentToImageConverter(
        LayoutLMArtifacts(output_dir=tmp_path, images_path=tmp_path / "cache")
    )
    page = {
        "document_id": document_row.document_id,
        "pdf_path": document_row.pdf_path,
        "page": 0,
    }
    image = converter.load_page_image(page, dpi=72)
    assert image.mode == "RGB"
    assert image.size == (144, 144)
    assert np.asarray(image).min() < 255
    cache = tmp_path / "cache" / "dpi_72" / "doc-1" / "page_0000.png"
    assert cache.is_file()
    # A cache hit must not need to reopen the source PDF.
    cached = converter.load_page_image({**page, "pdf_path": "missing.pdf"}, dpi=72)
    np.testing.assert_array_equal(np.asarray(image), np.asarray(cached))
    other_page = converter.load_page_image({**page, "page": 1}, dpi=144)
    assert other_page.size == (288, 288)
    assert (tmp_path / "cache" / "dpi_144" / "doc-1" / "page_0001.png").is_file()


@pytest.mark.parametrize(("page_index", "dpi"), [(-1, 72), (2, 72), (0, 0)])
def test_invalid_page_or_resolution_is_rejected(
    document_row, tmp_path, page_index, dpi
):
    converter = AttachmentToImageConverter(
        LayoutLMArtifacts(output_dir=tmp_path, images_path=tmp_path)
    )
    with pytest.raises(ValueError):
        converter.load_page_image(
            {
                "document_id": "doc",
                "pdf_path": document_row.pdf_path,
                "page": page_index,
            },
            dpi=dpi,
        )
