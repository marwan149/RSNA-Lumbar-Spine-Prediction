from pathlib import Path

from rsna_lumbar_spine_prediction.config import RSNAConfig


def test_config_defaults() -> None:
    config = RSNAConfig.from_data_dir(
        data_dir=Path("/tmp/data"),
        output_path=Path("/tmp/submission.csv"),
        checkpoint_paths=[Path("/tmp/model.pt")],
    )
    assert config.batch_size == 1
    assert config.device in {"cpu", "cuda:0"}
    assert config.model_name == "edgenext_base.in21k_ft_in1k"
    assert len(config.conditions) == 5
    assert len(config.levels) == 5
