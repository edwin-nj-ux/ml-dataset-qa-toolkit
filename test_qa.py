import pandas as pd

from src.qa import run_qa


def test_run_qa_basic():
    df = pd.DataFrame({
        "item_id": [1, 2, 2],
        "label": ["a", None, "b"],
        "confidence": [0.9, 0.5, 1.1],
    })

    res = run_qa(df, min_confidence=0.6)
    assert res.n_rows == 3
    assert res.duplicate_ids == 1
    assert res.missing_label_rate > 0
    assert res.invalid_confidence >= 1
    assert res.below_min_confidence >= 1
