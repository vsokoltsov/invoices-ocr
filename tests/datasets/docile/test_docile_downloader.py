from io import BytesIO
from zipfile import ZipFile

import httpx
import pytest

from app.datasets.docile.downloader import DocileDataset, DocileDatasetDownloader


@pytest.mark.parametrize("status", [200, 404, 403])
def test_download_stream_cache_and_http_error_redaction(tmp_path, monkeypatch, status):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("sample.txt", "sample")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            status, content=buffer.getvalue(), headers={"x-amz-request-id": "test-id"}
        )

    client_factory = httpx.Client
    client = client_factory(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        "app.datasets.docile.downloader.httpx.Client", lambda **kwargs: client
    )
    downloader = DocileDatasetDownloader(
        "private-token", target_dir=tmp_path, minimum_archive_bytes=0
    )
    assert "private-token" not in repr(downloader)
    if status != 200:
        with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
            downloader.download()
        assert "private-token" not in str(error.value)
        assert not list(tmp_path.glob("*.part"))
        return
    assert downloader.download() == tmp_path
    assert len(requests) == 1
    # A cached archive must be usable without making a request.
    with client_factory(transport=httpx.MockTransport(respond)) as cached_client:
        downloader._download_archive(
            cached_client, "annotated-trainval", "https://unused"
        )
    assert len(requests) == 1


def test_archive_selection_and_validation(tmp_path):
    downloader = DocileDatasetDownloader(
        "token/with space", dataset=DocileDataset.UNLABELED, without_pdfs=True
    )
    assert len(downloader.archive_names()) == 95
    assert downloader.archive_names()[-1] == "unlabeled-chunk-ocr-only-93"
    assert "token%2Fwith%20space" in downloader.download_urls()[0]
    assert (
        len(DocileDatasetDownloader("dummy", dataset="baselines").archive_names()) == 9
    )
    assert len(DocileDataset) == 222
    with pytest.raises(ValueError):
        DocileDatasetDownloader("")
    with pytest.raises(ValueError):
        DocileDatasetDownloader("x", minimum_archive_bytes=-1)
    with pytest.raises(ValueError):
        DocileDatasetDownloader("x", without_pdfs=True)
    invalid = tmp_path / "bad.zip"
    invalid.write_text("not a zip")
    with pytest.raises(ValueError, match="valid ZIP"):
        DocileDatasetDownloader("x", minimum_archive_bytes=0)._validate_archive(invalid)
    with pytest.raises(ValueError, match="minimum size"):
        DocileDatasetDownloader("x")._validate_archive(invalid)
