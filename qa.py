from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .metrics import (
    below_confidence_count,
    duplicates_count,
    imbalance_score,
    invalid_confidence_count,
    label_distribution,
    missing_rate,
)


@dataclass
class QAResults:
    n_rows: int
    columns: list
    id_col: str
    label_col: str
    confidence_col: str
    missing_label_rate: float
    duplicate_ids: int
    label_dist: Dict[str, int]
    imbalance: Dict[str, float]
    invalid_confidence: int
    below_min_confidence: int


def run_qa(
    df: pd.DataFrame,
    id_col: str = "item_id",
    label_col: str = "label",
    confidence_col: str = "confidence",
    min_confidence: Optional[float] = None,
) -> QAResults:
    if label_col not in df.columns:
        raise ValueError(f"Missing required column: {label_col}")

    dist = label_distribution(df, label_col)
    dist_dict = {str(k): int(v) for k, v in dist.to_dict().items()}

    return QAResults(
        n_rows=int(len(df)),
        columns=list(df.columns),
        id_col=id_col,
        label_col=label_col,
        confidence_col=confidence_col,
        missing_label_rate=missing_rate(df, label_col),
        duplicate_ids=duplicates_count(df, id_col),
        label_dist=dist_dict,
        imbalance=imbalance_score(dist),
        invalid_confidence=invalid_confidence_count(df, confidence_col),
        below_min_confidence=below_confidence_count(df, confidence_col, min_confidence),
    )
