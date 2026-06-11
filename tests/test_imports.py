from rsna_lumbar_spine_prediction import RSNAConfig, RSNA24TestDataset, RSNA24Model, run_submission_prediction


def test_imports() -> None:
    assert RSNAConfig is not None
    assert RSNA24TestDataset is not None
    assert RSNA24Model is not None
    assert callable(run_submission_prediction)
