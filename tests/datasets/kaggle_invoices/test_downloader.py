import io
from contextlib import contextmanager
from zipfile import ZipFile

import httpx
import pytest

from app.datasets.kaggle_invoices import downloader as downloader_module
from app.datasets.kaggle_invoices.downloader import InvoiceDatasetDownloader


def zip_bytes():
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("batch/invoices.csv", "File Name\ninvoice.png\n")
    return buffer.getvalue()


def test_download_streams_archive_extracts_and_reuses_cache(tmp_path, monkeypatch):
    payload = zip_bytes()
    calls = []

    @contextmanager
    def stream(method, url, **kwargs):
        calls.append((method, url, kwargs))
        yield httpx.Response(
            200,
            content=payload,
            headers={"Content-Length": str(len(payload))},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(downloader_module.httpx, "stream", stream)
    downloader = InvoiceDatasetDownloader(
        str(tmp_path / "invoices"), str(tmp_path / "cache" / "data.zip")
    )
    assert downloader.download_and_extract() == tmp_path / "invoices"
    assert downloader.archive_path.read_bytes() == payload
    assert (downloader.destination / "batch/invoices.csv").is_file()
    assert downloader.download() == downloader.archive_path
    assert len(calls) == 1
    assert calls[0][2]["follow_redirects"] is True
    assert not downloader.archive_path.with_suffix(".zip.part").exists()


@pytest.mark.parametrize("failure", ["status", "stream"])
def test_failed_download_does_not_publish_or_leave_partial_archive(
    tmp_path, monkeypatch, failure
):
    @contextmanager
    def stream(method, url, **kwargs):
        if failure == "status":
            yield httpx.Response(403, request=httpx.Request(method, url))
        else:

            class BrokenResponse:
                headers = {}

                def raise_for_status(self):
                    pass

                def iter_bytes(self, **kwargs):
                    yield b"partial"
                    raise httpx.ReadError("connection lost")

            yield BrokenResponse()

    monkeypatch.setattr(downloader_module.httpx, "stream", stream)
    downloader = InvoiceDatasetDownloader(tmp_path / "out", tmp_path / "data.zip")
    with pytest.raises(httpx.HTTPError):
        downloader.download()
    assert not downloader.archive_path.exists()
    assert not downloader.archive_path.with_suffix(".zip.part").exists()
