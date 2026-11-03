import json

import pandas as pd
import pytest

from app.datasets.kaggle_invoices.loader import InvoiceDatasetLoader


def test_loader_combines_batches_preserves_json_and_tracks_provenance(tmp_path):
    target = {"invoice": {"seller_name": "Acme"}}
    for batch in ("batch_a", "batch_b"):
        directory = tmp_path / batch
        (directory / "invoices").mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "File Name": "image.png",
                    "OCRed Text": "Seller Acme",
                    "Json Data": json.dumps(target),
                }
            ]
        ).to_csv(directory / "invoices.csv", index=False)
    frame = InvoiceDatasetLoader(tmp_path).load()
    assert len(frame) == 2
    assert frame["source_group"].tolist() == ["batch_a", "batch_b"]
    assert frame["ocred_text"].tolist() == ["Seller Acme"] * 2
    assert all(json.loads(value) == target for value in frame["json_data"])
    assert frame["path"].tolist() == [
        str(tmp_path / batch / "invoices/image.png") for batch in ("batch_a", "batch_b")
    ]
    assert frame["subdir"].tolist() == ["invoices"] * 2


def test_loader_skips_invalid_batches_and_raises_when_none_are_valid(tmp_path, capsys):
    pd.DataFrame({"File Name": ["image.png"]}).to_csv(
        tmp_path / "missing.csv", index=False
    )
    (tmp_path / "invalid").mkdir()
    pd.DataFrame({"wrong_column": ["image.png"]}).to_csv(
        tmp_path / "invalid.csv", index=False
    )
    with pytest.raises(RuntimeError, match="No valid CSV"):
        InvoiceDatasetLoader(tmp_path).load()
    output = capsys.readouterr().out
    assert "image directory not found" in output
    assert "'File Name' column not found" in output
