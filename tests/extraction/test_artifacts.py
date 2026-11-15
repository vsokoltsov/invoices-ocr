import numpy as np
import pandas as pd
import pytest
import torch

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts


@pytest.fixture(params=[BertArtifacts, LayoutLMArtifacts])
def artifacts(request, tmp_path):
    return request.param(tmp_path / "run")


def test_reading_missing_outputs_does_not_create_directories(artifacts):
    with pytest.raises(FileNotFoundError, match="Missing artifact"):
        artifacts.load_metrics()
    assert not artifacts.output_dir.exists()
    artifacts.prepare()
    artifacts.prepare()
    assert artifacts.output_dir.is_dir()


def test_persists_json_csv_and_weights_without_changing_their_contents(artifacts):
    artifacts.save_config({"model_name": "tiny", "labels": ["O", "B-NAME"]})
    artifacts.save_metrics({"f1": np.float64(0.75), "support": np.int64(3)})
    history = pd.DataFrame({"epoch": [1, 2], "loss": [0.5, 0.2]})
    artifacts.save_history(history)
    assert artifacts.load_config()["labels"] == ["O", "B-NAME"]
    assert artifacts.load_metrics() == {"f1": 0.75, "support": 3}
    pd.testing.assert_frame_equal(pd.read_csv(artifacts.history_path), history)
    path = artifacts.output_dir / "weights" / "saved.pt"
    artifacts.save_weights(path, {"weight": torch.tensor([1.0, 2.0])})
    torch.testing.assert_close(
        artifacts.load_weights(path)["weight"], torch.tensor([1.0, 2.0])
    )
    # Reject nonfinite JSON before replacing an existing report.
    with pytest.raises(ValueError):
        artifacts.save_metrics({"f1": float("nan")})
    assert artifacts.load_metrics() == {"f1": 0.75, "support": 3}


def test_publication_checks_file_types_and_run_membership(artifacts, tmp_path):
    with pytest.raises(FileNotFoundError):
        artifacts.publication_files()
    artifacts.validate_evaluation(
        artifacts.checkpoint_location, artifacts.test_metrics_path
    )
    with pytest.raises(ValueError, match="Checkpoint does not match"):
        artifacts.validate_evaluation(tmp_path / "another", artifacts.test_metrics_path)
    with pytest.raises(ValueError, match="publication directory"):
        artifacts.validate_evaluation(
            artifacts.checkpoint_location, tmp_path / "another-metrics.json"
        )
    # A directory at a required file path is not a valid artifact.
    artifacts.test_metrics_path.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        artifacts.load_metrics()


def test_bert_keeps_checkpoint_history_and_diagnostics_in_configured_locations(
    tmp_path,
):
    artifacts = BertArtifacts(
        tmp_path / "reports",
        checkpoint_dir=tmp_path / "weights",
        history_path=tmp_path / "history" / "training.csv",
        diagnostics_dir=tmp_path / "diagnostics",
    )
    artifacts.prepare()
    artifacts.save_manifest({"doc": "test"})
    assert artifacts.load_manifest() == {"doc": "test"}
    assert artifacts.manifest_path == tmp_path / "weights" / "split_manifest.json"
    assert artifacts.history_path.parent.is_dir()
    assert artifacts.diagnostics(2).output_dir == tmp_path / "diagnostics" / "epoch_002"
    assert not artifacts.has_records("test")
    artifacts.write_json(artifacts.records_path("test"), {"records": []})
    assert artifacts.has_records("test")
    for directory in (artifacts.checkpoint_dir, artifacts.checkpoint_dir / "debug"):
        with pytest.raises(ValueError, match="outside the checkpoint"):
            artifacts.debug_reports(directory)


def test_layoutlm_keeps_checkpoint_and_image_cache_outside_report_directory(tmp_path):
    artifacts = LayoutLMArtifacts(
        tmp_path / "reports",
        checkpoint_path=tmp_path / "weights" / "best.pt",
        images_path=tmp_path / "cache",
    )
    artifacts.prepare()
    artifacts.save_checkpoint({"weight": torch.tensor([3])})
    assert artifacts.load_checkpoint()["weight"].tolist() == [3]
    assert artifacts.page_image_path("doc", 2, 150) == (
        tmp_path / "cache" / "dpi_150" / "doc" / "page_0002.png"
    )
    assert artifacts.load_page_image("doc", 2, 150) is None
